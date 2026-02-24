import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    LARSConfig,
    AdaLoraConfig,
    IA3Config,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model,
    TaskType,
)
import wandb
from tqdm import tqdm

# -------------------------
# Argument Parser
# -------------------------


#8, 256
#8, 512
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--peft_method", type=str, default="lora",
                        choices=["lora", "adalora", "ia3", "prefix", "prompt", "bitfit", "lars", "full"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    # parser.add_argument("--lr", type=float, default=2e-5) #FT
    # parser.add_argument("--lr", type=float, default=1e-4) #LARS, LoRA
    # parser.add_argument("--lr", type=float, default=5e-3) #IA3
    parser.add_argument("--lr", type=float, default=1e-3) #prefix
    # parser.add_argument("--lr", type=float, default=5e-2) #prompt
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--project", type=str, default="boolq_llama_peft")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--dataset",  type=str, default="boolq",
                        choices=["boolq", "piqa"])

    return parser.parse_args()


# -------------------------
# Dataset
# -------------------------
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

def build_piqa_dataset(tokenizer, max_len):
    ds = load_dataset("lighteval/piqa")

    def preprocess(ex):
        text = (
            f"Goal: {ex['goal']}\n"
            f"Solution 1: {ex['sol1']}\n"
            f"Solution 2: {ex['sol2']}"
        )

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": ex["label"],  # already 0 or 1
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



# -------------------------
# PEFT Config Factory
# -------------------------
def get_peft_config(method):

    if method == "lora":
        return LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
        )

    elif method == "adalora":
        return AdaLoraConfig(
            r=8,
            target_r=4,
            init_r=8,
            lora_alpha=16,
            task_type=TaskType.SEQ_CLS,
            total_step=1500
        )

    elif method == "ia3":
        return IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=["k_proj", "v_proj", "down_proj"],
        )

    elif method == "prefix":
        return PrefixTuningConfig(
            num_virtual_tokens=20,
            task_type=TaskType.SEQ_CLS,
        )

    elif method == "prompt":
        return PromptTuningConfig(
            num_virtual_tokens=20,
            task_type=TaskType.SEQ_CLS,
        )

    elif method == "bitfit":
        return None  # handled manually

    elif method == "lars":
        return LARSConfig(
        task_type=TaskType.SEQ_CLS,   # sequence classification
        target_modules= "all-linear",
        fan_in_fan_out=False,              # use fan-in scaling, optional
        rank=8,
        learned_pooling=False,    
    )


def get_dataset(dataset_name, tokenizer, max_len):
    if dataset_name == "boolq":
        return build_boolq_dataset(tokenizer, max_len)
    elif dataset_name == "piqa":
        return build_piqa_dataset(tokenizer, max_len)
    


# -------------------------
# BitFit Setup
# -------------------------
def apply_bitfit(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# -------------------------
# Evaluation
# -------------------------
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


# -------------------------
# Train
# -------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=args.project, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token


    # dataset = build_boolq_dataset(tokenizer, args.max_len)
    dataset = get_dataset(args.dataset, tokenizer, args.max_len)

    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size,
                              shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),)
    val_loader = DataLoader(dataset["validation"], batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),)

    model = LlamaForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        num_labels=2,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    if args.checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Apply PEFT
    if args.peft_method == "bitfit":
        model = apply_bitfit(model)
    elif args.peft_method == "full":
        pass  # fine-tune all parameters, no adapter
    else:
        peft_config = get_peft_config(args.peft_method)
        model = get_peft_model(model, peft_config)

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    wandb.log({
        "trainable_params": trainable_params,
        "total_params": total_params,
    })

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    total_steps = 3000
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps,
    )


    model.train()
    # Training loop
    step = 0
    optimizer.zero_grad()
    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= 3000:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.accum_steps
            loss.backward()

            # inside the accumulation step
            if (batch_idx + 1) % args.accum_steps == 0:
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
                    acc = evaluate(model, val_loader, device)

                log_dict = {
                    "loss": float(loss.item() * args.accum_steps),
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
                print(f"Step {step:04d} | Loss {loss.item()*args.accum_steps:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} | MemPeak {mem_peak:.1f}MB" + (f" | Acc {acc:.4f}" if acc is not None else ""))

                step += 1

        if step >= 3000:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()


