import math
import torch
import torch.nn as nn
import wandb
from torch.utils.data import random_split, DataLoader

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)

from peft import IA3Config, get_peft_model, TaskType

from alpaca_dataset import build_alpaca_dataset

# ---------------- CONFIG ----------------

MODEL_ID = "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "hybrid_lowrank_adapters"

BATCH_SIZE = 8
STEPS = 3000
LR = 2e-4
WARMUP_STEPS = 300
MAX_LEN = 256

VAL_FRAC = 0.05
VAL_EVERY = 20
ACCUM_STEPS = 4

# ---------------- EVAL ----------------

@torch.no_grad()
def evaluate_hf(model, dev_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dev_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            **batch,
            use_cache=False,
        )

        loss = outputs.loss
        valid = (batch["labels"] != -100).sum().item()

        total_loss += loss.item() * valid
        total_tokens += valid

    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    return mean_loss, math.exp(mean_loss)

# ---------------- MAIN ----------------

def main():
    wandb.init(project=PROJECT_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    full_dataset = build_alpaca_dataset(
        tokenizer,
        max_len=MAX_LEN,
        mask_prompt=True,
    )

    n_dev = max(1, int(VAL_FRAC * len(full_dataset)))
    n_train = len(full_dataset) - n_dev

    train_ds, dev_ds = random_split(
        full_dataset,
        [n_train, n_dev],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)

    # ---------------- Base Model ----------------

    model = LlamaForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)

    # ---------------- IA³ ----------------

    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            # Attention
            "k_proj",
            "v_proj",
            # MLP
            "up_proj",
            "down_proj",
        ],
        feedforward_modules=[
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, ia3_config)
    model.print_trainable_parameters()
    model.train()

    # Optimizer (only IA³ params are trainable)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=STEPS,
    )

    step = 0
    optimizer.zero_grad()

    # ---------------- Training Loop ----------------

    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= STEPS:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                **batch,
                use_cache=False,
            )

            loss = outputs.loss / ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                mem_alloc = mem_peak = mem_reserved = 0.0
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e6
                    mem_peak = torch.cuda.max_memory_allocated() / 1e6
                    mem_reserved = torch.cuda.memory_reserved() / 1e6

                wandb.log({
                    "loss": float(loss.item() * ACCUM_STEPS),
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm),
                    "mem_allocated_MB": mem_alloc,
                    "mem_peak_MB": mem_peak,
                    "mem_reserved_MB": mem_reserved,
                    "step": step,
                })

                if step % VAL_EVERY == 0 and step > 0:
                    val_loss, val_ppl = evaluate_hf(model, dev_loader, device)
                    wandb.log({
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                        "step": step,
                    })
                    print(
                        f"[VAL] step={step:04d} | "
                        f"loss={val_loss:.4f} | ppl={val_ppl:.2f}"
                    )

                print(
                    f"step={step:04d} | loss={(loss.item()*ACCUM_STEPS):.6f} | "
                    f"lr={lr:.2e} | grad_norm={float(grad_norm):.2f} | "
                    f"mem_alloc={mem_alloc:.1f}MB mem_peak={mem_peak:.1f}MB"
                )

                step += 1

        if step >= STEPS:
            break

    print("DONE")

if __name__ == "__main__":
    main()
