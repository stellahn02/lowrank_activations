import math
import torch
import torch.nn as nn
import wandb
from torch.utils.data import random_split, DataLoader

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    get_cosine_schedule_with_warmup,
)

from alpaca_dataset import build_alpaca_dataset


def evaluate_hf(model, dev_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss  # mean over valid tokens in this batch

            valid = (labels != -100).sum().item()
            total_loss += loss.item() * valid
            total_tokens += valid

    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss)
    return mean_loss, ppl

# ---------------- CONFIG ----------------

MODEL_ID = "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "hybrid_lowrank_adapters"

BATCH_SIZE = 8
STEPS = 3000
LR = 3e-4
WARMUP_STEPS = 300
MAX_LEN = 256

VAL_FRAC = 0.05
VAL_EVERY = 20
ACCUM_STEPS = 4

ACT_RANK = 8   # activation bottleneck
# LORA_RANK = 8     # LoRA rank

# ---------------- LoRA ----------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_out, d_in = base.weight.shape
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)

        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(x))

# class LoRAOnlyLinear(nn.Module):
#     def __init__(self, base: nn.Linear, r: int, alpha: float):
#         super().__init__()
#         self.lora = LoRALinear(base, r=r, alpha=alpha)

#     def forward(self, x):
#         return self.lora(x)


# -------- Activation Low-Rank Adapter --------
# class ActivationLowRankLinear(nn.Module):
#     """
#     Frozen Linear + trainable low-rank activation adapter
#     """

#     def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
#         super().__init__()

#         self.base = base
#         self.base.weight.requires_grad_(False)
#         if self.base.bias is not None:
#             self.base.bias.requires_grad_(False)

#         d_out = base.out_features
#         d_in = base.in_features

#         # low-rank activation adapter
#         self.U = nn.Linear(d_out, rank, bias=False)
#         self.V = nn.Linear(rank, d_out, bias=False)

#         self.act = nn.SiLU()
#         self.scaling = alpha / rank

#         # important init
#         nn.init.zeros_(self.V.weight)
#         nn.init.kaiming_uniform_(self.U.weight, a=math.sqrt(5))

#     def forward(self, x):
#         h = self.base(x)              # frozen weights
#         z = self.act(self.U(h))       # NO detach
#         delta = self.scaling * self.V(z)
#         return h + delta

class LowRankActivationScalingLinear(nn.Module):
    """
    Frozen Linear + low-rank multiplicative activation adapter
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d = base.out_features
        self.U = nn.Linear(d, rank, bias=False)
        self.V = nn.Linear(rank, d, bias=False)

        self.act = nn.SiLU()

        nn.init.zeros_(self.V.weight)

    def forward(self, x):
        h = self.base(x)
        gate = 1.0 + self.V(self.act(self.U(h)))
        return h * gate



# ---------------- Wrapping ----------------

# def wrap_linears(module: nn.Module, act_rank: int, lora_rank: int):
#     for name, child in list(module.named_children()):
#         if isinstance(child, nn.Linear):
#             setattr(
#                 module,
#                 name,
#                 HybridAdapterLinear(child, act_rank, lora_rank),
#             )
#         else:
#             wrap_linears(child, act_rank, lora_rank)

def wrap_linears_activation_only(
    module: nn.Module,
    act_rank: int,
):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LowRankActivationScalingLinear(child, act_rank),
            )
        else:
            wrap_linears_activation_only(child, act_rank)

# def wrap_linears_activation_only(
#     module: nn.Module,
#     act_rank: int,
#     prefix: str = "",
# ):
#     """
#     Apply LowRankActivationScalingLinear ONLY to MLP projections:
#       - mlp.gate_proj
#       - mlp.up_proj
#       - mlp.down_proj
#     """
#     for name, child in list(module.named_children()):
#         full_name = f"{prefix}.{name}" if prefix else name

#         if isinstance(child, nn.Linear):
#             if any(
#                 key not in full_name
#                 for key in (
#                     "mlp.gate_proj",
#                     "mlp.up_proj",
#                     "mlp.down_proj",
#                 )
#             ):
#                 setattr(
#                     module,
#                     name,
#                     LowRankActivationScalingLinear(child, act_rank),
#                 )
#         else:
#             wrap_linears_activation_only(
#                 child,
#                 act_rank,
#                 prefix=full_name,
#             )



# ---------------- Model Wrapper ----------------

class LlamaActivationAdapters(nn.Module):
    def __init__(self, model_id: str, act_rank: int, tokenizer):
        super().__init__()

        base = LlamaForCausalLM.from_pretrained(model_id)

        if len(tokenizer) > base.get_input_embeddings().num_embeddings:
            base.resize_token_embeddings(len(tokenizer))

        for p in base.parameters():
            p.requires_grad = False

        wrap_linears_activation_only(base.model, act_rank)

        self.base = base

    def forward(self, **kwargs):
        return self.base(**kwargs)


# ---------------- Evaluation ----------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, use_cache=False)
        loss = out.loss
        valid = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * valid
        total_tokens += valid

    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    return mean_loss, math.exp(mean_loss)

# ---------------- Main ----------------

def main():
    wandb.init(project=PROJECT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = build_alpaca_dataset(
        tokenizer,
        max_len=MAX_LEN,
        mask_prompt=True,
    )

    n_dev = int(len(dataset) * VAL_FRAC)
    train_ds, dev_ds = random_split(dataset, [len(dataset) - n_dev, n_dev])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)

    model = LlamaActivationAdapters(
        MODEL_ID,
        act_rank=ACT_RANK,
        tokenizer=tokenizer,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        WARMUP_STEPS,
        STEPS,
    )

    step = 0
    optimizer.zero_grad()

    for epoch in range(1000):
        for i, batch in enumerate(train_loader):
            if step >= STEPS:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch, use_cache=False).loss
            (loss / ACCUM_STEPS).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                # torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                lr = scheduler.get_last_lr()[0]

                mem_alloc = mem_peak = mem_reserved = 0.0
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e6
                    mem_peak = torch.cuda.max_memory_allocated() / 1e6
                    mem_reserved = torch.cuda.memory_reserved() / 1e6

                # log loss * ACCUM_STEPS to undo scaling
                wandb.log({
                    "loss": float(loss.item() * ACCUM_STEPS),
                    "learning_rate": LR,
                    "grad_norm": float(grad_norm),
                    "mem_allocated_MB": mem_alloc,
                    "mem_peak_MB": mem_peak,
                    "mem_reserved_MB": mem_reserved,
                    "step": step,
                })

                # periodic validation
                if step % VAL_EVERY == 0 and step > 0:
                    val_loss, val_ppl = evaluate_hf(model, dev_loader, device)
                    print(f"[VAL] step={step:04d} | val_loss={val_loss:.6f} | val_ppl={val_ppl:.2f}")
                    wandb.log({
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                        "step": step,
                    })

                print(
                    f"step={step:04d} | loss={(loss.item()*ACCUM_STEPS):.6f} | lr={lr:.2e} | "
                    f"grad_norm={float(grad_norm):.2f} | "
                    f"mem_alloc={mem_alloc:.1f}MB mem_peak={mem_peak:.1f}MB "
                    f"mem_reserved={mem_reserved:.1f}MB"
                )

                step += 1


        if step >= STEPS:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()
