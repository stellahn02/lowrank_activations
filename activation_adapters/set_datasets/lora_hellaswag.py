import os
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
PROJECT_NAME = "llama_hellaswag_peft"

BATCH_SIZE = 16
STEPS = 3000
LR = 5e-4
WARMUP_STEPS = 300
MAX_LEN = 256
ACCUM_STEPS = 4  # gradient accumulation

LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

# ---------------- HELPERS ----------------
def option_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [N, 2] (binary seq-cls head)
    return: [N] scalar score
    """
    return logits[:, 1] - logits[:, 0]


def normalized_option_score(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    logits: [N, 2]
    lengths: [N] lengths of the sequences (float)
    Returns: [N] length-normalized scores
    """
    raw_score = option_score_from_logits(logits)   # [N]
    return raw_score / lengths


# ---------------- DATASET ----------------
def build_hellaswag_4way_dataset(tokenizer, max_len):
    ds = load_dataset("hellaswag")

    def preprocess(ex):
        ctx = ex.get("ctx", ex.get("context"))
        endings = ex["endings"]
        
        # Handle potential bad label data
        try:
            label = int(ex["label"])
        except (ValueError, TypeError):
            return None 

        out = {"labels": label}
        for i in range(4):
            text = f"Context: {ctx}\nEnding: {endings[i]}"
            enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            out[f"input_ids_{i}"] = enc["input_ids"]
            out[f"attention_mask_{i}"] = enc["attention_mask"]
            out[f"len_{i}"] = len(enc["input_ids"])
        return out


    original_cols = ds["train"].column_names
    ds = ds.map(preprocess, remove_columns=original_cols)
    ds = ds.filter(lambda x: x is not None)
    
    return ds


def collate_hellaswag_4way(batch, pad_id):
    batch = [x for x in batch if x is not None]
    
    if len(batch) == 0:
        return None 

    def pad(seqs, pad_value):
        maxlen = max(len(s) for s in seqs)
        return [s + [pad_value] * (maxlen - len(s)) for s in seqs]

    out = {}
    for i in range(4):
        ids = pad([x[f"input_ids_{i}"] for x in batch], pad_id)
        msk = pad([x[f"attention_mask_{i}"] for x in batch], 0)
        out[f"input_ids_{i}"] = torch.tensor(ids, dtype=torch.long)
        out[f"attention_mask_{i}"] = torch.tensor(msk, dtype=torch.long)
        out[f"len_{i}"] = torch.tensor([x[f"len_{i}"] for x in batch], dtype=torch.float)

    out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return out



def pack_4way_batch(batch, device, pad_id):
    """
    Pack 4 options per example into a single batch with global padding.
    """
    input_ids_list = []
    attention_mask_list = []
    lengths_list = []

    # Find the global max length across all 4 option groups
    max_t = 0
    for i in range(4):
        max_t = max(max_t, batch[f"input_ids_{i}"].size(1))

    # Pad each group to that global max length
    for i in range(4):
        ids = batch[f"input_ids_{i}"]
        msk = batch[f"attention_mask_{i}"]
        B, T = ids.shape
        
        if T < max_t:
            padding_len = max_t - T
            # Pad ids with pad_id, msk with 0
            ids = torch.cat([ids, torch.full((B, padding_len), pad_id, dtype=torch.long)], dim=1)
            msk = torch.cat([msk, torch.zeros((B, padding_len), dtype=torch.long)], dim=1)
        
        input_ids_list.append(ids)
        attention_mask_list.append(msk)
        lengths_list.append(batch[f"len_{i}"])

    ids = torch.cat(input_ids_list, dim=0).to(device)    # [B*4, max_t]
    msk = torch.cat(attention_mask_list, dim=0).to(device)  # [B*4, max_t]
    lens = torch.cat(lengths_list, dim=0).to(device)     # [B*4]

    return ids, msk, lens


# ---------------- EVALUATION ----------------
@torch.no_grad()
def evaluate_hellaswag_4way(model, dev_loader, device, pad_id):
    model.eval()
    correct = total = 0

    for batch in dev_loader:
        labels = batch["labels"].to(device)
        B = labels.size(0)

        ids, msk, lens = pack_4way_batch(batch, device, pad_id)             # [B*4, T], [B*4, T], [B*4]
        logits = model(input_ids=ids, attention_mask=msk).logits     # [B*4, 2]
        scores = normalized_option_score(logits, lens)               # [B*4]

        scores_4way = scores.view(4, B).transpose(0, 1).contiguous() # [B, 4]
        preds = scores_4way.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += B

    model.train()
    return correct / total


# ---------------- MAIN ----------------
def main():
    wandb.init(project=PROJECT_NAME, group="lars_norm_packed")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    hs_ds = build_hellaswag_4way_dataset(tokenizer, MAX_LEN)
    train_loader = DataLoader(
        hs_ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_hellaswag_4way(b, tokenizer.pad_token_id),
    )
    dev_loader = DataLoader(
        hs_ds["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_hellaswag_4way(b, tokenizer.pad_token_id),
    )

    # Base model
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
    model.to(torch.bfloat16)
    if hasattr(model, "score"):
        model.score.to(torch.bfloat16)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable params: {sum(p.numel() for p in trainable):,} / "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=STEPS,
    )

    # (optional) sanity-print trainable LARS params
    for n, p in model.named_parameters():
        if "U.weight" in n or "V.weight" in n or p.requires_grad:
            print(n, p.requires_grad)

    # Training loop
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= STEPS:
                break

            labels = batch["labels"].to(device)
            B = labels.size(0)

            ids, msk, lens = pack_4way_batch(batch, device, tokenizer.pad_token_id)        # [B*4, T], [B*4, T], [B*4]
            logits = model(input_ids=ids, attention_mask=msk).logits
            scores = normalized_option_score(logits, lens)          # [B*4]

            scores_4way = scores.view(4, B).transpose(0, 1).contiguous()  # [B, 4]

            loss = F.cross_entropy(scores_4way, labels) / ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                mem_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_reserved = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0

                acc = None
                if step % 50 == 0 and step > 0:
                    acc = evaluate_hellaswag_4way(model, dev_loader, device, tokenizer.pad_token_id)

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
                    log_dict["hellaswag_acc_norm"] = acc

                wandb.log(log_dict)
                print(
                    f"Step {step:04d} | Loss {loss.item()*ACCUM_STEPS:.4f} "
                    f"| LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} "
                    f"| MemAlloc {mem_alloc:.1f}MB"
                    + (f" | Acc_norm {acc:.4f}" if acc is not None else "")
                )

                step += 1

        if step >= STEPS:
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
