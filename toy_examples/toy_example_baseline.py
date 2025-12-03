import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaEmbeddings(nn.Module):
    def __init__(self, num_embeddings, d_model):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings, d_model)
    def forward(self, input_ids):
        return self.word_embeddings(input_ids)  # [batch, seq, d_model]

class VanillaAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x_ln = self.ln1(x)
        attn_output, _ = self.self_attn(x_ln, x_ln, x_ln)
        x = x + attn_output
        x = x + self.ff(self.ln2(x))
        return x

class VanillaToyTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, num_layers):
        super().__init__()
        self.embed = VanillaEmbeddings(num_tokens, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, d_model))  # max seq 128
        self.layers = nn.ModuleList([VanillaAttentionBlock(d_model) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, num_tokens)
    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos_embed[:, :input_ids.size(1)]
        for layer in self.layers:
            x = layer(x)
        logits = self.output_proj(x)
        return logits

def main():
    batch = 4
    seq = 16
    vocab = 100
    d_model = 64    # full model size
    num_layers = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = torch.randint(0, vocab, (batch, seq)).to(device)
    model = VanillaToyTransformer(vocab, d_model, num_layers).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logits = model(input_ids)
    print('Logits:', logits.shape)
    loss = logits.mean()
    loss.backward()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
        print(f"Peak CUDA mem: {peak_mem:.1f} MB")
    else:
        print("Not running on CUDA — memory tracking unavailable.")

if __name__ == "__main__":
    main()
