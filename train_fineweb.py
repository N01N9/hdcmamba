import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm

# ⚡ 사용자 정의 HdcMamba 모델 임포트
from hdcmamba import HdcMamba9v3Model

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

# =========================================================================
# 1. 통합 Language Model Factory
# =========================================================================
class UniversalLM(nn.Module):
    def __init__(self, arch, scale, vocab_size=50277):
        super().__init__()
        self.arch = arch
        self.scale = scale
        
        configs = {
            "130M": {
                "Transformer": {"d_model": 768, "n_layers": 15, "n_heads": 12},
                "Mamba2":      {"d_model": 768, "n_layers": 25, "d_state": 64},
                "HdcMamba":    {"d_model": 768, "n_layers": 34, "d_state": 64, "n_heads": 12},
            },
            "360M": {
                "Transformer": {"d_model": 1024, "n_layers": 26, "n_heads": 16},
                "Mamba2":      {"d_model": 1024, "n_layers": 43, "d_state": 128},
                "HdcMamba":    {"d_model": 1024, "n_layers": 57, "d_state": 128, "n_heads": 16},
            },
            "1B": {
                "Transformer": {"d_model": 2048, "n_layers": 19, "n_heads": 32},
                "Mamba2":      {"d_model": 2048, "n_layers": 31, "d_state": 128},
                "HdcMamba":    {"d_model": 2048, "n_layers": 42, "d_state": 128, "n_heads": 32},
            }
        }
        
        cfg = configs[scale][arch]
        self.d_model = cfg["d_model"]
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        
        if arch == "HdcMamba":
            self.backbone = HdcMamba9v3Model(
                d_model=self.d_model, n_layers=cfg["n_layers"], d_state=cfg["d_state"], 
                num_slots=32, num_heads=cfg["n_heads"]
            )
        elif arch == "Transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=cfg["n_heads"], dim_feedforward=self.d_model*4, 
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=cfg["n_layers"])
        elif arch == "Mamba2":
            self.backbone = nn.ModuleList([
                Mamba2(d_model=self.d_model, d_state=cfg["d_state"]) for _ in range(cfg["n_layers"])
            ])

        self.norm_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        # ⚡ 핀셋 초기화: SSM 내부를 망가뜨리지 않도록 외부 레이어만 초기화
        self._init_weights(self.embedding)
        self._init_weights(self.lm_head)
        self._init_weights(self.norm_f)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.04)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.04)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        if self.arch == "Transformer":
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            x = self.backbone(x, mask=mask, is_causal=True)
        elif self.arch == "Mamba2":
            for layer in self.backbone: x = layer(x)
        else: x = self.backbone(x)
        return self.lm_head(self.norm_f(x))

# =========================================================================
# 2. 데이터 로더 (고속 토크나이징 + 셔플링)
# =========================================================================
class FineWebEduIterable(IterableDataset):
    def __init__(self, tokenizer, seq_len=8192, split="train"):
        # ⚡ 스트리밍 셔플링 추가
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split=split, streaming=True)
        self.dataset = self.dataset.shuffle(seed=42, buffer_size=10000)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        batch_texts = []
        for sample in self.dataset:
            batch_texts.append(sample["text"])
            if len(batch_texts) >= 64: # 64개 문장씩 묶어서 Rust 엔진으로 처리
                encodings = self.tokenizer(batch_texts, add_special_tokens=True)["input_ids"]
                for tokens in encodings: buffer.extend(tokens)
                batch_texts = []
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

# =========================================================================
# 3. 메인 학습 파이프라인
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True, choices=["Transformer", "Mamba2", "HdcMamba"])
    parser.add_argument("--scale", type=str, required=True, choices=["130M", "360M", "1B"])
    args = parser.parse_args()

    DEVICE, DTYPE, SEQ_LEN = "cuda", torch.bfloat16, 8192
    
    # ⚡ 최적화된 마이크로 배치 및 학습률 설정
    if args.scale == "130M":
        MICRO_BATCH, GRAD_ACCUM_STEPS, LR = 8, 4, 6e-4
        MAX_STEPS, WARMUP_STEPS = 20000, 100 # 빠른 학습 전환을 위해 워밍업 단축
    elif args.scale == "360M":
        MICRO_BATCH, GRAD_ACCUM_STEPS, LR = 4, 8, 4e-4
        MAX_STEPS, WARMUP_STEPS = 80000, 4000
    elif args.scale == "1B":
        MICRO_BATCH, GRAD_ACCUM_STEPS, LR = 2, 16, 2e-4
        MAX_STEPS, WARMUP_STEPS = 200000, 10000
    
    torch.backends.cuda.matmul.allow_tf32 = True
    run_name = f"{args.arch}-{args.scale}-FineWebEdu-Final"
    wandb.init(project="HdcMamba-Scaling-Laws", name=run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)
    train_loader = DataLoader(FineWebEduIterable(tokenizer, seq_len=SEQ_LEN), batch_size=MICRO_BATCH, num_workers=0)

    model = UniversalLM(arch=args.arch, scale=args.scale, vocab_size=len(tokenizer)).to(DEVICE).to(DTYPE)
    print(f"✅ Model Ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    step, data_iter = 0, iter(train_loader)
    progress_bar = tqdm(total=MAX_STEPS, desc=run_name)
    
    while step < MAX_STEPS:
        optimizer.zero_grad(set_to_none=True)
        micro_loss_acc = 0.0
        
        for _ in range(GRAD_ACCUM_STEPS):
            try: x, y = next(data_iter)
            except StopIteration: data_iter = iter(train_loader); x, y = next(data_iter)
            
            with torch.amp.autocast('cuda', dtype=DTYPE):
                logits = model(x.to(DEVICE))
                loss = loss_fn(logits.view(-1, logits.size(-1)).float(), y.to(DEVICE).view(-1))
                loss = loss / GRAD_ACCUM_STEPS
            
            loss.backward()
            micro_loss_acc += loss.item()
            
        # ⚡ 기울기 청진기 (Grad Norm) 계산 및 로깅
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        
        if step % 10 == 0:
            wandb.log({
                "train/loss": micro_loss_acc,
                "train/grad_norm": total_norm,
                "train/lr": scheduler.get_last_lr()[0],
                "train/tokens_seen": step * MICRO_BATCH * GRAD_ACCUM_STEPS * SEQ_LEN,
                "system/vram_GB": torch.cuda.max_memory_allocated() / 1e9
            }, step=step)
            progress_bar.set_postfix({"loss": f"{micro_loss_acc:.4f}", "norm": f"{total_norm:.2f}"})
        
        progress_bar.update(1)
        if step % 5000 == 0:
            torch.save(model.state_dict(), f"checkpoints/{run_name}_step_{step}.pt")

    wandb.finish()

if __name__ == "__main__": main()