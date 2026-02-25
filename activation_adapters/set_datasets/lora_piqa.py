import math
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForSequenceClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# ---------------- CONFIG ----------------
MODEL_ID = "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "llama_piqa_peft"

BATCH_SIZE = 8
STEPS = 3000
LR = 5e-4
WARMUP_STEPS = 300
MAX_LEN = 256
ACCUM_STEPS = 4  # gradient accumulation

LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
torch.cuda.empty_cache()

# ---------------- DATASET ----------------
def build_piqa_pair_dataset(tokenizer, max_len):
    ds = load_dataset("piqa")

    def preprocess(ex):
        goal = ex["goal"]
        sol1 = ex["sol1"]
        sol2 = ex["sol2"]

        # Two separate sequences
        text1 = f"Goal: {goal}\nSolution: {sol1}"
        text2 = f"Goal: {goal}\nSolution: {sol2}"

        enc1 = tokenizer(text1, truncation=True, max_length=max_len, padding=False)
        enc2 = tokenizer(text2, truncation=True, max_length=max_len, padding=False)

        return {
            "input_ids_0": enc1["input_ids"],
            "attention_mask_0": enc1["attention_mask"],
            "input_ids_1": enc2["input_ids"],
            "attention_mask_1": enc2["attention_mask"],
            "labels": int(ex["label"]),  # 0 means sol1, 1 means sol2
        }

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
    return ds


def collate_piqa_pairs(batch, pad_id):
    def pad(seqs, pad_value):
        maxlen = max(len(s) for s in seqs)
        return [s + [pad_value] * (maxlen - len(s)) for s in seqs]

    ids0 = pad([x["input_ids_0"] for x in batch], pad_id)
    msk0 = pad([x["attention_mask_0"] for x in batch], 0)
    ids1 = pad([x["input_ids_1"] for x in batch], pad_id)
    msk1 = pad([x["attention_mask_1"] for x in batch], 0)
    labels = [x["labels"] for x in batch]

    return {
        "input_ids_0": torch.tensor(ids0, dtype=torch.long),
        "attention_mask_0": torch.tensor(msk0, dtype=torch.long),
        "input_ids_1": torch.tensor(ids1, dtype=torch.long),
        "attention_mask_1": torch.tensor(msk1, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def option_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, 2] from LlamaForSequenceClassification(num_labels=2)
    return: [B] scalar score (logit difference)
    """
    return logits[:, 1] - logits[:, 0]

# ---------------- EVALUATION ----------------
@torch.no_grad()
def evaluate_piqa_pairs(model, dev_loader, device):
    model.eval()
    correct = total = 0

    for batch in dev_loader:
        labels = batch["labels"].to(device)

        logits0 = model(
            input_ids=batch["input_ids_0"].to(device),
            attention_mask=batch["attention_mask_0"].to(device),
        ).logits
        logits1 = model(
            input_ids=batch["input_ids_1"].to(device),
            attention_mask=batch["attention_mask_1"].to(device),
        ).logits

        score0 = option_score_from_logits(logits0)  # [B]
        score1 = option_score_from_logits(logits1)  # [B]
        pair_logits = torch.stack([score0, score1], dim=1)  # [B, 2]

        preds = pair_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return correct / total

# ---------------- MAIN ----------------
def main():
    wandb.init(project=PROJECT_NAME, group="lars")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    piqa_ds = build_piqa_pair_dataset(tokenizer, MAX_LEN)
    train_loader = DataLoader(
        piqa_ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_piqa_pairs(b, tokenizer.pad_token_id),
    )
    dev_loader = DataLoader(
        piqa_ds["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_piqa_pairs(b, tokenizer.pad_token_id),
    )

    # Model + LoRA
    base_model = LlamaForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(base_model, lora_config)
    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=STEPS,
    )

    for n, p in model.named_parameters():
        if "U.weight" in n or "V.weight" in n or p.requires_grad==True:
            print(n, p.requires_grad)

    # Training loop
    step = 0
    optimizer.zero_grad()
    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= STEPS:
                break

            ids0 = batch["input_ids_0"].to(device)
            msk0 = batch["attention_mask_0"].to(device)
            ids1 = batch["input_ids_1"].to(device)
            msk1 = batch["attention_mask_1"].to(device)
            labels = batch["labels"].to(device)

            # forward both options
            logits0 = model(input_ids=ids0, attention_mask=msk0).logits  # [B,2]
            logits1 = model(input_ids=ids1, attention_mask=msk1).logits  # [B,2]

            score0 = option_score_from_logits(logits0)  # [B]
            score1 = option_score_from_logits(logits1)  # [B]
            pair_logits = torch.stack([score0, score1], dim=1)  # [B,2]

            loss = F.cross_entropy(pair_logits, labels) / ACCUM_STEPS
            loss.backward()

            # inside the accumulation step
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                mem_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_peak  = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_reserved = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0

                # evaluate only every 50 steps
                acc = None
                if step % 50 == 0 and step > 0:
                    acc = evaluate_piqa_pairs(model, dev_loader, device)

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
                    log_dict["piqa_acc"] = acc

                wandb.log(log_dict)
                print(f"Step {step:04d} | Loss {loss.item()*ACCUM_STEPS:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} | MemAlloc {mem_alloc:.1f}MB" + (f" | Acc {acc:.4f}" if acc is not None else ""))

                step += 1

        if step >= STEPS:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()


