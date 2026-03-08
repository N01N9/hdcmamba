import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        d_inner = d_model * 2
        # Mamba-2 표준 투영 크기
        self.in_proj = nn.Linear(d_model, d_inner * 2 + 2 * d_state + d_model, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

class HdcMambaBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        # HdcMamba 융합 투영 크기
        self.in_proj = nn.Linear(d_model, d_state * 3 + d_model * 4, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)
        self.out_proj = nn.Linear(d_state + d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model); self.group_norm = nn.GroupNorm(8, d_model)

class LanguageModel(nn.Module):
    def __init__(self, arch_type, vocab_size, d_model, n_layers, d_state=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if arch_type == "Transformer":
            self.layers = nn.ModuleList([TransformerBlock(d_model) for _ in range(n_layers)])
        elif arch_type == "Mamba2":
            self.layers = nn.ModuleList([Mamba2Block(d_model, d_state) for _ in range(n_layers)])
        elif arch_type == "HdcMamba":
            self.layers = nn.ModuleList([HdcMambaBlock(d_model, d_state) for _ in range(n_layers)])
            
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # ⚡ Weight Tying 적용 (실제 LLM 표준)
        self.lm_head.weight = self.embedding.weight

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6

VOCAB_SIZE = 32000

# ⚡ 정밀 튜닝된 아키텍처 스펙
configs = {
    "130M": {
        "Transformer": {"d_model": 768, "n_layers": 15},
        "Mamba2":      {"d_model": 768, "n_layers": 25, "d_state": 64},
        "HdcMamba":    {"d_model": 768, "n_layers": 34, "d_state": 64}, # 34층의 깊은 추론
    },
    "360M": {
        "Transformer": {"d_model": 1024, "n_layers": 26},
        "Mamba2":      {"d_model": 1024, "n_layers": 43, "d_state": 128},
        "HdcMamba":    {"d_model": 1024, "n_layers": 57, "d_state": 128}, # 57층의 극강 뎁스
    },
    "1B": {
        "Transformer": {"d_model": 2048, "n_layers": 19},
        "Mamba2":      {"d_model": 2048, "n_layers": 31, "d_state": 128},
        "HdcMamba":    {"d_model": 2048, "n_layers": 42, "d_state": 128},
    }
}

print("="*50)
for scale, archs in configs.items():
    print(f"🚀 --- {scale} Scale Models ---")
    for arch_type, kwargs in archs.items():
        model = LanguageModel(arch_type, VOCAB_SIZE, **kwargs)
        print(f"[{arch_type:<11}] L={kwargs['n_layers']:<2} | Params: {model.count_params():.1f} M")
print("="*50)