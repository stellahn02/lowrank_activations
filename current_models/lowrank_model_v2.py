import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer

wandb.init(project="lowrank_model_llama32_like")

# ---------------- RMSNorm (same style as original scratch) ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Same formula as your original scratch model
        norm = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.weight * x / (norm + self.eps)

# ---------------- RoPE (baseline-style) ----------------

def apply_rope(x, seq_dim=1, rope_theta=10000):
    """
    x: (B, S, H, hd) with seq_dim = 1
    """
    b, s, nh, hd = x.shape
    device, dtype = x.device, x.dtype
    pos = torch.arange(s, device=device, dtype=dtype)               # (S,)
    inv = 1.0 / (rope_theta ** (torch.arange(0, hd, 2, device=device, dtype=dtype) / hd))
    freqs = torch.outer(pos, inv)                                   # (S, hd/2)
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(0).unsqueeze(2)                             # (1, S, 1, hd/2)
    cos = cos.unsqueeze(0).unsqueeze(2)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rope = torch.zeros_like(x)
    x_rope[..., ::2] = x1 * cos - x2 * sin
    x_rope[..., 1::2] = x2 * cos + x1 * sin
    return x_rope

# ---------------- Low-rank embedding ----------------

class LowRankEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, k, Vk):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Vk: (d_model, k)
        self.Vk = nn.Parameter(Vk, requires_grad=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)      # (B, S, d_model)
        return x @ self.Vk             # (B, S, k)

# ---------------- GQA Attention in low-rank space ----------------

class GQAttention_LR(nn.Module):
    def __init__(self, k, n_heads, n_kv_heads, attn_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = k // n_heads

        self.q_proj = nn.Linear(k, k, bias=False)
        self.k_proj = nn.Linear(k, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(k, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(k, k, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        x: (B, S, K)
        """
        B, S, K = x.shape
        H, H_kv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)      # (B, H, S, hd)
        k = self.k_proj(x).view(B, S, H_kv, hd).transpose(1, 2)   # (B, H_kv, S, hd)
        v = self.v_proj(x).view(B, S, H_kv, hd).transpose(1, 2)   # (B, H_kv, S, hd)

        q = apply_rope(q)
        k = apply_rope(k)

        if H_kv != H:
            k = k.repeat_interleave(H // H_kv, dim=1)             # (B, H, S, hd)
            v = v.repeat_interleave(H // H_kv, dim=1)             # (B, H, S, hd)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)  # (B, H, S, S)

        # Causal mask
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))

        # Stable softmax
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)                                # (B, H, S, hd)
        z = z.transpose(1, 2).contiguous().view(B, S, K)         # (B, S, K)
        return self.out_proj(z)

# ---------------- FFN (GELU, like your original LR version) ----------------

class LlamaBlock_LR(nn.Module):
    def __init__(self, k, ffn_dim, n_heads, n_kv_heads, attn_dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(k)
        self.attn = GQAttention_LR(k, n_heads, n_kv_heads, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(k)
        self.ffn = nn.Sequential(
            nn.Linear(k, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, k)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ---------------- Low-rank Llama-3.2-like model ----------------

class Llama3_1B_LowRank(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=2048,
        k=384,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim=8192,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k = k

        self.embed = None   # will be set via set_lowrank_embedding

        self.layers = nn.ModuleList([
            LlamaBlock_LR(k, ffn_dim, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(k)
        self.lm_head = nn.Linear(k, vocab_size, bias=False)

    def set_lowrank_embedding(self, Vk):
        """
        Vk: (d_model, k) on any device; will be moved to model device.
        """
        device = next(self.parameters()).device
        Vk = Vk.to(device)
        self.embed = LowRankEmbedding(self.vocab_size, self.d_model, self.k, Vk).to(device)

    def forward(self, input_ids):
        x = self.embed(input_ids)        # (B, S, k)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)        # (B, S, vocab_size)
        return logits

# ---------------- TRAINING & LOGGING ----------------

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    seq_len = 24
    batch = 4

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers power modern language models.",
        "Low-rank approximations save GPU memory.",
        "Learning rate schedules help neural nets converge."
    ]

    d_model = 2048
    k = 384           # low-rank dimension
    n_layers = 16
    n_heads = 32
    n_kv_heads = 8
    ffn_dim = 8192

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Collect basis Vk from random embeddings on device ---
    embed_tmp = nn.Embedding(vocab_size, d_model).to(device)
    acts = []
    for _ in range(16):
        ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
        a = embed_tmp(ids).reshape(-1, d_model).cpu()  # keep acts on CPU for PCA
        acts.append(a)
    acts = torch.cat(acts, dim=0)
    with torch.no_grad():
        U, S, V = torch.pca_lowrank(acts, q=k, center=True)
        Vk = V[:, :k].contiguous()   # still on CPU for now

    # --- Build low-rank Llama model, move to device, then set embedding ---
    model = Llama3_1B_LowRank(
        vocab_size=vocab_size,
        d_model=d_model,
        k=k,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_dim=ffn_dim,
    ).to(device)
    model.set_lowrank_embedding(Vk)  # will move Vk and embedding to device

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    steps = 50

    for step in range(steps):
        toks = tokenizer(
            sentences * ((batch + len(sentences) - 1) // len(sentences)),
            padding="max_length",
            truncation=True,
            max_length=seq_len + 1,
            return_tensors="pt",
        )
        all_ids = toks["input_ids"][:batch].to(device)
        input_ids = all_ids[:, :-1]
        targets = all_ids[:, 1:]

        logits = model(input_ids)
        B, S, Vocab = logits.shape
        loss = ce_loss(
            logits.view(B * S, Vocab),
            targets.contiguous().view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        mem_alloc = mem_peak = mem_reserved = 0.0
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e6
            mem_peak = torch.cuda.max_memory_allocated() / 1e6
            mem_reserved = torch.cuda.memory_reserved() / 1e6

        wandb.log({
            "loss": float(loss.item()),
            "learning_rate": lr,
            "grad_norm": float(grad_norm),
            "mem_allocated_MB": mem_alloc,
            "mem_peak_MB": mem_peak,
            "mem_reserved_MB": mem_reserved,
            "step": step,
        })

        print(
            f"step={step:02d} | loss={loss.item():.4f} | lr={lr:.2e} | "
            f"grad_norm={float(grad_norm):.2f} | "
            f"mem_alloc={mem_alloc:.1f}MB mem_peak={mem_peak:.1f}MB "
            f"mem_reserved={mem_reserved:.1f}MB"
        )

    print("DONE. Check W&B dashboard for learning curves.")

if __name__ == "__main__":
    main()
