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

# torch.autograd.set_detect_anomaly(True)


# =========================================================================
# 1. 통합 Language Model Factory (아키텍처 스위칭)
# =========================================================================
class UniversalLM(nn.Module):
    def __init__(self, arch, scale, vocab_size=32000):
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
            if Mamba2 is None:
                raise ImportError("mamba_ssm 패키지가 설치되지 않았습니다.")
            self.backbone = nn.ModuleList([
                Mamba2(d_model=self.d_model, d_state=cfg["d_state"]) for _ in range(cfg["n_layers"])
            ])

        self.norm_f = nn.LayerNorm(self.d_model)
        
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights(self.embedding)
        self._init_weights(self.lm_head)
        self._init_weights(self.norm_f)

    def _init_weights(self, module):
        """표준 LLM 가중치 초기화 (분산 0.02의 정규분포)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        if self.arch == "Transformer":
            seq_len = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.backbone(x, mask=mask, is_causal=True)
        elif self.arch == "Mamba2":
            for layer in self.backbone:
                x = layer(x)
        else: # HdcMamba
            x = self.backbone(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

# =========================================================================
# 2. 데이터 로더 (FineWeb-Edu 교체 완료)
# =========================================================================
class FineWebEduIterable(IterableDataset):
    def __init__(self, tokenizer, seq_len=8192, split="train"):
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        batch_texts = []
        for sample in self.dataset:
            batch_texts.append(sample["text"])
            
            # 🔥 핵심: 텍스트를 64개씩 모아서 초고속(Rust 엔진)으로 한 번에 토크나이징
            if len(batch_texts) >= 64:
                encodings = self.tokenizer(batch_texts, add_special_tokens=True)["input_ids"]
                for tokens in encodings:
                    buffer.extend(tokens)
                batch_texts = [] # 텍스트 버퍼 비우기
                
                # 모델 입력 크기(seq_len)만큼 버퍼가 차면 GPU로 발사
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

# =========================================================================
# 3. 메인 학습 파이프라인
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Architecture Training")
    parser.add_argument("--arch", type=str, required=True, choices=["Transformer", "Mamba2", "HdcMamba"])
    parser.add_argument("--scale", type=str, required=True, choices=["130M", "360M", "1B"])
    args = parser.parse_args()

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    SEQ_LEN = 8192
    
    if args.scale == "130M":
        MICRO_BATCH = 4
        GRAD_ACCUM_STEPS = 8
        LR = 2e-4
        MAX_STEPS = 20000
        WARMUP_STEPS = 1000
    elif args.scale == "360M":
        MICRO_BATCH = 4
        GRAD_ACCUM_STEPS = 8
        LR = 1e-4
        MAX_STEPS = 80000
        WARMUP_STEPS = 4000
    elif args.scale == "1B":
        MICRO_BATCH = 2
        GRAD_ACCUM_STEPS = 16
        LR = 5e-5
        MAX_STEPS = 200000
        WARMUP_STEPS = 10000
    
    torch.backends.cuda.matmul.allow_tf32 = True

    run_name = f"{args.arch}-{args.scale}-FineWebEdu"
    wandb.init(project="HdcMamba-Scaling-Laws", name=run_name, group=args.scale, config={
        "arch": args.arch, "scale": args.scale, "seq_len": SEQ_LEN,
        "global_batch_size": MICRO_BATCH * GRAD_ACCUM_STEPS, "lr": LR,
        "dataset": "FineWeb-Edu-100BT"
    })

    print(f"⏳ Initializing {run_name} Model with FineWeb-Edu...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)
    # ⚡ 클래스 이름 변경 반영
    train_dataset = FineWebEduIterable(tokenizer, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH, num_workers=0)

    model = UniversalLM(arch=args.arch, scale=args.scale, vocab_size=len(tokenizer))
    model.to(DEVICE).to(DTYPE)
    print(f"✅ Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS)
    loss_fn = nn.CrossEntropyLoss()

    print(f"🔥 Starting Training for {run_name} on A100...")
    model.train()
    
    step = 0
    accumulated_loss = 0.0
    progress_bar = tqdm(total=MAX_STEPS, desc=run_name)
    data_iter = iter(train_loader)
    
    while step < MAX_STEPS:
        optimizer.zero_grad(set_to_none=True)
        
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
                
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            with torch.amp.autocast('cuda', dtype=DTYPE):
                logits = model(x)
                # ⚡ 1. Loss 계산 직전에 logits를 float32로 캐스팅하여 오버플로우 방지
                loss = loss_fn(logits.view(-1, logits.size(-1)).float(), y.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
            
            loss.backward()
            accumulated_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        
        step += 1
        current_lr = scheduler.get_last_lr()[0]
        
        if step % 10 == 0:
            wandb.log({
                "train/loss": accumulated_loss,
                "train/lr": current_lr,
                "train/tokens_seen": step * MICRO_BATCH * GRAD_ACCUM_STEPS * SEQ_LEN,
                "system/vram_allocated_GB": torch.cuda.max_memory_allocated() / 1e9
            }, step=step)
            
            progress_bar.set_postfix({"loss": f"{accumulated_loss:.4f}", "lr": f"{current_lr:.2e}"})
            
        accumulated_loss = 0.0
        progress_bar.update(1)
        
        if step % 5000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{run_name}_step_{step}.pt")
            
    wandb.finish()

if __name__ == "__main__":
    main()