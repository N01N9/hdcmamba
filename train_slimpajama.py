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
from hdc_mamba_9_full import HdcMamba9v3Model

# (선택) 비교군 모델 라이브러리 (설치된 경우에만 동작하도록 예외 처리)
try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

# =========================================================================
# 1. 통합 Language Model Factory (아키텍처 스위칭)
# =========================================================================
class UniversalLM(nn.Module):
    def __init__(self, arch, scale, vocab_size=32000):
        super().__init__()
        self.arch = arch
        self.scale = scale
        
        # 앞서 정밀하게 맞춘 체급별 스펙
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
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        
        # 2. Backbone (아키텍처 분기)
        if arch == "HdcMamba":
            self.backbone = HdcMamba9v3Model(
                d_model=self.d_model, 
                n_layers=cfg["n_layers"], 
                d_state=cfg["d_state"], 
                num_slots=32, 
                num_heads=cfg["n_heads"]
            )
        elif arch == "Transformer":
            # PyTorch 기본 Causal Transformer
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=cfg["n_heads"], dim_feedforward=self.d_model*4, 
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=cfg["n_layers"])
        elif arch == "Mamba2":
            if Mamba2 is None:
                raise ImportError("mamba_ssm 패키지가 설치되지 않았습니다. (pip install mamba-ssm)")
            self.backbone = nn.ModuleList([
                Mamba2(d_model=self.d_model, d_state=cfg["d_state"]) for _ in range(cfg["n_layers"])
            ])

        self.norm_f = nn.LayerNorm(self.d_model)
        
        # 3. LM Head & Weight Tying
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        if self.arch == "Transformer":
            # Causal Mask 생성
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
# 2. 데이터 로더 (기존과 동일)
# =========================================================================
class SlimPajamaIterable(IterableDataset):
    def __init__(self, tokenizer, seq_len=8192, split="train"):
        self.dataset = load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            tokens = self.tokenizer(sample["text"], add_special_tokens=True)["input_ids"]
            buffer.extend(tokens)
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

    # ⚡ A100(80GB) 환경 맞춤 동적 하이퍼파라미터 (OOM 방지)
    # 모델이 커질수록 한 번에 올릴 수 있는 마이크로 배치를 줄이고, 누적(Accumulation)을 늘립니다.
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    SEQ_LEN = 8192
    
# 기존 고정 설정이었던 MAX_STEPS와 WARMUP_STEPS를 체급별로 동적 할당
    if args.scale == "130M":
        MICRO_BATCH = 16
        GRAD_ACCUM_STEPS = 2   # Global Batch = 32 (약 26만 토큰/스텝)
        LR = 6e-4
        MAX_STEPS = 20000      # 🎯 목표: 약 52억(5.2B) 토큰 학습
        WARMUP_STEPS = 1000    # MAX_STEPS의 약 5%
    elif args.scale == "360M":
        MICRO_BATCH = 6
        GRAD_ACCUM_STEPS = 5   # Global Batch = 30 (약 24만 토큰/스텝)
        LR = 3e-4
        MAX_STEPS = 80000      # 🎯 목표: 약 196억(19.6B) 토큰 학습
        WARMUP_STEPS = 4000
    elif args.scale == "1B":
        MICRO_BATCH = 2
        GRAD_ACCUM_STEPS = 16  # Global Batch = 32 (약 26만 토큰/스텝)
        LR = 2e-4
        MAX_STEPS = 200000     # 🎯 목표: 약 524억(52.4B) 토큰 학습
        WARMUP_STEPS = 10000
    
    torch.backends.cuda.matmul.allow_tf32 = True

    # W&B 동적 이름 부여 (예: HdcMamba-360M)
    run_name = f"{args.arch}-{args.scale}"
    wandb.init(project="HdcMamba-Scaling-Laws", name=run_name, group=args.scale, config={
        "arch": args.arch, "scale": args.scale, "seq_len": SEQ_LEN,
        "global_batch_size": MICRO_BATCH * GRAD_ACCUM_STEPS, "lr": LR
    })

    print(f"⏳ Initializing {run_name} Model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    train_dataset = SlimPajamaIterable(tokenizer, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH, num_workers=2)

    model = UniversalLM(arch=args.arch, scale=args.scale, vocab_size=tokenizer.vocab_size)
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
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
            
            loss.backward()
            accumulated_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        step += 1
        current_lr = scheduler.get_last_lr()[0]
        
        if step % 10 == 0:
            # ⚡ 논문용 핵심 로깅 (Loss, LR, 실제 처리 토큰 수)
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