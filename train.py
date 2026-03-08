import os
import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import HdcMamba9v3 from hdcmamba package
from hdcmamba import HdcMamba9v3Block as HdcMambaBlock

# ──────────────────────────────────────────────────────────────
# 1. Hyperparameters (100M Parameter Scale / GPT-2 Small 체급)
# ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# --------------------------------------------------------
# 🔹 Transformer Configs (~85.1M Params)
# --------------------------------------------------------
D_MODEL_TF = 768      # HeadDim = 768 / 12 = 64
N_LAYERS_TF = 12      
N_HEADS_TF = 12       

# --------------------------------------------------------
# 🔹 HdcMamba-9v3 Configs (~86.7M Params)
# 트랜스포머와 레이어 수를 12개로 동일하게 맞추고,
# 파라미터 수를 맞추기 위해 D_MODEL을 1152로 스케일업
# --------------------------------------------------------
D_MODEL_HDC = 1280    # HeadDim = 1280 / 20 = 64 (Power of 2 fixed)
N_LAYERS_HDC = 12     
N_HEADS_HDC = 20      
D_STATE_HDC = 128     

# --------------------------------------------------------
# 🔹 Training & Task Configs
# --------------------------------------------------------
LR = 4e-4             # 100M 체급이므로 학습률을 살짝 낮춤 (4e-3 -> 4e-4)
BATCH_SIZE = 16       # 100M 모델의 VRAM OOM 방지를 위해 배치 16
SEQ_LEN = 1024        # 진정한 LM 테스트를 위한 긴 문맥 (청크 128의 배수)
TRAIN_STEPS = 3000    # 100M 모델이 추세를 보여주기에 충분한 스텝
LOG_INTERVAL = 100
GENERATE_INTERVAL = 500 # 500 스텝마다 모델이 창작한 대사 출력

print(f"🚀 TinyShakespeare Language Model Benchmark (~100M Scale)")
print(f"Seq Len: {SEQ_LEN}, Batch Size: {BATCH_SIZE}, Steps: {TRAIN_STEPS}")

# ──────────────────────────────────────────────────────────────
# 2. Data Loading (TinyShakespeare)
# ──────────────────────────────────────────────────────────────
DATA_PATH = 'input.txt'
if not os.path.exists(DATA_PATH):
    print("Downloading TinyShakespeare dataset...")
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        f.write(requests.get(url).text)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars) # 셰익스피어 데이터셋은 약 65개의 문자를 사용합니다.
print(f"Vocab Size: {VOCAB_SIZE} characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% Train, 10% Val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix]) # 다음 토큰 예측(Next Token)
    return x.to(DEVICE), y.to(DEVICE)

# ──────────────────────────────────────────────────────────────
# 3. Model Wrappers & Causal Transformer
# ──────────────────────────────────────────────────────────────
# ⚡ 중요: 트랜스포머가 미래를 보지 못하도록 Causal Masking 강제
class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # 1. 현재 시퀀스 길이(L)에 맞는 Causal Mask 생성 (미래 시점 차단)
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        
        # 2. attn_mask 파라미터로 마스크 전달
        attn_out, _ = self.attn(
            self.ln_1(x), self.ln_1(x), self.ln_1(x),
            attn_mask=causal_mask,
            need_weights=False, 
            is_causal=True
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

class LMHeadModel(nn.Module):
    def __init__(self, backbone, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = backbone
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight # Weight tying
        
    def forward(self, x):
        h = self.embedding(x)
        out = self.backbone(h)
        aux_loss = 0.0
        if isinstance(out, tuple):
            h, aux_loss = out
            if aux_loss is None: aux_loss = 0.0
        else:
            h = out
            
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, aux_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -SEQ_LEN:] # Context Window 넘지 않게 자름
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # 마지막 토큰의 예측값만 가져옴
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# ──────────────────────────────────────────────────────────────
# 4. Training Loop
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('val')
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(X)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), Y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def train_model(model_name, model, steps, lr=4e-4):
    print(f"\n[Training] {model_name}...")
    model = model.to(DEVICE).to(DTYPE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch('train')
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, aux_loss = model(x)
            base_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss = base_loss + 0.01 * aux_loss
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % LOG_INTERVAL == 0 or step == 1:
            val_loss = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"  Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
            t0 = time.time()
            
        if step % GENERATE_INTERVAL == 0:
            print(f"\n--- 🗣️ {model_name} Generation (Step {step}) ---")
            context = torch.tensor((encode("O God, ")), dtype=torch.long, device=DEVICE).unsqueeze(0)
            generated = model.generate(context, max_new_tokens=150)
            print(decode(generated[0].tolist()))
            print("-" * 50 + "\n")
            t0 = time.time()

# ──────────────────────────────────────────────────────────────
# 5. Main Execution
# ──────────────────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # --- 1. Transformer (Baseline ~19M) ---
    class TFBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL_TF)
            self.layers = nn.ModuleList([CausalTransformerBlock(D_MODEL_TF, N_HEADS_TF) for _ in range(N_LAYERS_TF)])
        def forward(self, x):
            positions = torch.arange(x.size(1), device=x.device)
            x = x + self.pos_emb(positions).unsqueeze(0)
            for l in self.layers: x = l(x)
            return x
    
    tf_model = LMHeadModel(TFBackbone(), D_MODEL_TF, VOCAB_SIZE)
    print(f"\n[Model] Transformer (Causal) | Layers: {N_LAYERS_TF} | Params: {count_parameters(tf_model)/1e6:.2f}M")
    train_model("Transformer", tf_model, TRAIN_STEPS, lr=LR)
    del tf_model; torch.cuda.empty_cache()
    
    # --- 2. HdcMamba-9v3 (Optimized ~20M) ---
    class HdcBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                HdcMambaBlock(d_model=D_MODEL_HDC, d_state=D_STATE_HDC, num_heads=N_HEADS_HDC, chunk_size=128)
                for _ in range(N_LAYERS_HDC)
            ])
        def forward(self, x):
            total_aux_loss = 0.0
            for l in self.layers:
                x, aux = l(x)
                if aux is not None: total_aux_loss += aux
            return x, total_aux_loss
            
    hdc_model = LMHeadModel(HdcBackbone(), D_MODEL_HDC, VOCAB_SIZE)
    print(f"\n[Model] HdcMamba-9v3 | Layers: {N_LAYERS_HDC} | Params: {count_parameters(hdc_model)/1e6:.2f}M")
    train_model("HdcMamba-9v3", hdc_model, TRAIN_STEPS, lr=LR)
    
if __name__ == "__main__":
    main()