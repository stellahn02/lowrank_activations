import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model name may change, check Hugging Face page for exact name
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
class LowRankEmbeddings(nn.Module):
    def __init__(self, num_embeddings, d_model, k, Vk):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings, d_model)
        self.Vk = nn.Parameter(Vk, requires_grad=False)
    def forward(self, input_ids):
        emb = self.word_embeddings(input_ids)   # [batch, seq, d_model]
        return emb @ self.Vk                   # [batch, seq, k]

class SimpleLowRankAttentionBlock(nn.Module):
    def __init__(self, k, n_heads=2):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        self.q_proj = nn.Linear(k, k)
        self.k_proj = nn.Linear(k, k)
        self.v_proj = nn.Linear(k, k)
        self.out_proj = nn.Linear(k, k)
        self.ln1 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, k*4),
            nn.ReLU(),
            nn.Linear(k*4, k)
        )
        self.ln2 = nn.LayerNorm(k)
    def forward(self, x):
        x_ln = self.ln1(x)
        Q = self.q_proj(x_ln)
        K = self.k_proj(x_ln)
        V = self.v_proj(x_ln)
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.k ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V
        x = x + self.out_proj(attn_output)
        x = x + self.ff(self.ln2(x))
        return x

class LowRankToyTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, k, num_layers, Vk):
        super().__init__()
        self.embed = LowRankEmbeddings(num_tokens, d_model, k, Vk)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, k))  # max seq 128
        self.layers = nn.ModuleList([SimpleLowRankAttentionBlock(k) for _ in range(num_layers)])
        self.output_proj = nn.Linear(k, num_tokens)  # For logits
    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos_embed[:, :input_ids.size(1)]
        for layer in self.layers:
            x = layer(x)
        logits = self.output_proj(x)  # [batch, seq, vocab]
        return logits

def main():
    batch = 4
    seq = 16
    vocab = 100
    d_model = 64
    k = 8
    num_layers = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Vk = torch.randn(d_model, k).to(device)
    input_ids = torch.randint(0, vocab, (batch, seq)).to(device)
    model = LowRankToyTransformer(vocab, d_model, k, num_layers, Vk).to(device)

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
