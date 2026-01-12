# import math
# import torch
# import torch.nn as nn
# import wandb
# from torch.utils.data import random_split, DataLoader

# from transformers import (
#     AutoTokenizer,
#     LlamaForCausalLM,
#     LlamaConfig,
#     get_cosine_schedule_with_warmup,
# )

# from peft import LoraConfig, get_peft_model, TaskType

# from alpaca_dataset import build_alpaca_dataset

# # ---------------- CONFIG ----------------

# MODEL_ID = "meta-llama/Llama-3.2-1B"
# PROJECT_NAME = "hybrid_lowrank_adapters"

# BATCH_SIZE = 8
# STEPS = 3000
# LR = 2e-4          # LoRA can use slightly higher LR
# WARMUP_STEPS = 300
# MAX_LEN = 256

# VAL_FRAC = 0.05
# VAL_EVERY = 20
# ACCUM_STEPS = 4

# lora_rank = 8

# # ---------------- EVAL ----------------

# def evaluate_hf(model, dev_loader, device):
#     model.eval()
#     total_loss = 0.0
#     total_tokens = 0

#     with torch.no_grad():
#         for batch in dev_loader:
#             input_ids = batch["input_ids"].to(device)
#             labels = batch["labels"].to(device)
#             attention_mask = batch["attention_mask"].to(device)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 use_cache=False,
#             )

#             loss = outputs.loss
#             valid = (labels != -100).sum().item()
#             total_loss += loss.item() * valid
#             total_tokens += valid

#     model.train()
#     mean_loss = total_loss / max(1, total_tokens)
#     return mean_loss, math.exp(mean_loss)

# # ---------------- MAIN ----------------

# def main():
#     wandb.init(project=PROJECT_NAME)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # Dataset
#     full_dataset = build_alpaca_dataset(
#         tokenizer,
#         max_len=MAX_LEN,
#         mask_prompt=True,
#     )

#     n_dev = max(1, int(VAL_FRAC * len(full_dataset)))
#     n_train = len(full_dataset) - n_dev
#     train_ds, dev_ds = random_split(
#         full_dataset,
#         [n_train, n_dev],
#         generator=torch.Generator().manual_seed(42),
#     )

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)

#     # Base model
#     model = LlamaForCausalLM.from_pretrained(MODEL_ID)
#     model.to(device)

#     # ---------------- LoRA ----------------
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=lora_rank,
#         lora_alpha=lora_rank,
#         lora_dropout=0.05,
#         target_modules=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#             "gate_proj",
#             "up_proj",
#             "down_proj",
#         ],
#         bias="none",
#     )

#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
#     model.train()

#     # Optimizer only sees LoRA params
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LR,
#         weight_decay=0.0,
#     )

#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=WARMUP_STEPS,
#         num_training_steps=STEPS,
#     )

#     step = 0
#     optimizer.zero_grad()

#     for epoch in range(1000):
#         for batch_idx, batch in enumerate(train_loader):
#             if step >= STEPS:
#                 break

#             input_ids = batch["input_ids"].to(device)
#             labels = batch["labels"].to(device)
#             attention_mask = batch["attention_mask"].to(device)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 use_cache=False,
#             )

#             loss = outputs.loss / ACCUM_STEPS
#             loss.backward()

#             if (batch_idx + 1) % ACCUM_STEPS == 0:
#                 grad_norm = torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), 1.0
#                 )
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#                 lr = scheduler.get_last_lr()[0]

#                 mem_alloc = mem_peak = mem_reserved = 0.0
#                 if torch.cuda.is_available():
#                     mem_alloc = torch.cuda.memory_allocated() / 1e6
#                     mem_peak = torch.cuda.max_memory_allocated() / 1e6
#                     mem_reserved = torch.cuda.memory_reserved() / 1e6

#                 # log loss * ACCUM_STEPS to undo scaling
#                 wandb.log({
#                     "loss": float(loss.item() * ACCUM_STEPS),
#                     "learning_rate": LR,
#                     "grad_norm": float(grad_norm),
#                     "mem_allocated_MB": mem_alloc,
#                     "mem_peak_MB": mem_peak,
#                     "mem_reserved_MB": mem_reserved,
#                     "step": step,
#                 })

#                 # periodic validation
#                 if step % VAL_EVERY == 0 and step > 0:
#                     val_loss, val_ppl = evaluate_hf(model, dev_loader, device)
#                     print(f"[VAL] step={step:04d} | val_loss={val_loss:.6f} | val_ppl={val_ppl:.2f}")
#                     wandb.log({
#                         "val_loss": val_loss,
#                         "val_perplexity": val_ppl,
#                         "step": step,
#                     })

#                 print(
#                     f"step={step:04d} | loss={(loss.item()*ACCUM_STEPS):.6f} | lr={lr:.2e} | "
#                     f"grad_norm={float(grad_norm):.2f} | "
#                     f"mem_alloc={mem_alloc:.1f}MB mem_peak={mem_peak:.1f}MB "
#                     f"mem_reserved={mem_reserved:.1f}MB"
#                 )

#                 step += 1

#         if step >= STEPS:
#             break

#     print("DONE")

# if __name__ == "__main__":
#     main()


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
    LlamaForSequenceClassification
)

from peft import LoraConfig, get_peft_model, TaskType

from datasets import load_dataset

# ---------------- CONFIG ----------------

MODEL_ID = "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "hybrid_lowrank_adapters"

BATCH_SIZE = 8
STEPS = 3000
LR = 2e-4          # LoRA can use slightly higher LR
WARMUP_STEPS = 300
MAX_LEN = 256

VAL_FRAC = 0.05
VAL_EVERY = 20
ACCUM_STEPS = 4

LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

# ---------------- DATASET ----------------

def build_boolq_dataset(tokenizer, max_len):
    ds = load_dataset("boolq")

    def preprocess(ex):
        text = f"Question: {ex['question']}\nPassage: {ex['passage']}"
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(ex["answer"]),  # 0/1 for sequence classification
        }

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
    return ds

def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [pad_id]*(max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0]*(max_len - len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ---------------- EVAL ----------------

@torch.no_grad()
def evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0

    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return correct / total

# ---------------- MAIN ----------------

def main():
    wandb.init(project=PROJECT_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    boolq_ds = build_boolq_dataset(tokenizer, MAX_LEN)
    train_loader = DataLoader(
        boolq_ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    dev_loader = DataLoader(
        boolq_ds["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # Model + LoRA
    base_model = LlamaForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    # Optimizer only sees LoRA params
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

            loss = outputs.loss / ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                mem_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_peak  = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_reserved = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0

                # evaluate only every 50 steps
                acc = None
                if step % 50 == 0 and step > 0:
                    acc = evaluate(model, dev_loader, device)

                log_dict = {
                    "loss": float(loss.item() * ACCUM_STEPS),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "mem_allocated_MB": mem_alloc,
                    "mem_peak_MB": mem_peak,
                    "mem_reserved_MB": mem_reserved,
                    "step": step,
                }
                if acc is not None:
                    log_dict["boolq_acc"] = acc

                wandb.log(log_dict)
                print(f"Step {step:04d} | Loss {loss.item()*ACCUM_STEPS:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} | MemAlloc {mem_alloc:.1f}MB" + (f" | Acc {acc:.4f}" if acc is not None else ""))

                step += 1


        if step >= STEPS:
            break

    print("DONE")

if __name__ == "__main__":
    main()

