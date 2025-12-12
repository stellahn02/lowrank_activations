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

from alpaca_dataset import build_alpaca_dataset, get_alpaca_tokenizer


# ---------------- CONFIG ----------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROJECT_NAME = "hf_llama32_with_val_alpaca"

BATCH_SIZE = 8
STEPS = 3000
LR = 3e-5
WARMUP_STEPS = 300
MAX_LEN = 256

VAL_FRAC = 0.05
VAL_EVERY = 100
ACCUM_STEPS = 4   # gradient accumulation


# ---------------- EVAL ----------------

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


# ---------------- MAIN ----------------

def main():
    wandb.init(project=PROJECT_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Tokenizer & HF config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Full Alpaca dataset using your helper
    alpaca_tok = get_alpaca_tokenizer("distilbert-base-uncased")
    full_dataset = build_alpaca_dataset(
        alpaca_tok,
        max_len=MAX_LEN,
        mask_prompt=True,
    )

    # 3. Random train/dev split
    n_total = len(full_dataset)
    n_dev = max(1, int(VAL_FRAC * n_total))
    n_train = n_total - n_dev
    train_ds, dev_ds = random_split(
        full_dataset, [n_train, n_dev],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)

    batch0 = next(iter(train_loader))
    print("labels[0][:40]:", batch0["labels"][0][:40])
    print("unique labels:", torch.unique(batch0["labels"]))
    print("ignore_index in CE:", -100)

    # 4. HF Llama model
    config = LlamaConfig.from_pretrained(MODEL_ID)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=None if device == "cpu" else "auto",
    )
    if device == "cpu":
        model.to(device)
    model.train()

    # 5. Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=STEPS,
    )

    # 6. Training loop with gradient accumulation + periodic validation
    step = 0              # optimizer steps
    optimizer.zero_grad()

    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= STEPS:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss

            # scale by ACCUM_STEPS for accumulation
            loss = loss / ACCUM_STEPS
            loss.backward()

            # only update every ACCUM_STEPS mini-batches
            if (batch_idx + 1) % ACCUM_STEPS == 0:
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
                    "learning_rate": lr,
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

    print("DONE. Check W&B for HF train/val curves.")


if __name__ == "__main__":
    main()
