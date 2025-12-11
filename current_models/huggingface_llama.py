import torch
import torch.nn as nn
import wandb

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    get_cosine_schedule_with_warmup,
)

# ---------------- CONFIG ----------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # or "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "baseline_model_llama32_hf"

SEQ_LEN = 24
BATCH_SIZE = 4
STEPS = 40
LR = 1e-4
WARMUP_STEPS = 5

# ---------------- MAIN ------------------

def main():
    wandb.init(project=PROJECT_NAME)

    # 1. Tokenizer & config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig.from_pretrained(MODEL_ID)

    # 2. Model (exact Llama architecture + pretrained weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.float32 if device == "cuda" else torch.float32,
        device_map=None if device == "cpu" else "auto",
    )
    if device == "cpu":
        model.to(device)
    model.train()

    # 3. Same toy corpus
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers power modern language models.",
        "Low-rank approximations save GPU memory.",
        "Learning rate schedules help neural nets converge."
    ]

    # 4. Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=STEPS,
    )
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 5. Training loop (very similar structure to your scratch loop)
    for step in range(STEPS):
        # Repeat sentences to fill the batch
        repeat_factor = (BATCH_SIZE + len(sentences) - 1) // len(sentences)
        batch_sentences = (sentences * repeat_factor)[:BATCH_SIZE]

        toks = tokenizer(
            batch_sentences,
            padding="max_length",
            truncation=True,
            max_length=SEQ_LEN + 1,  # +1 for next-token target
            return_tensors="pt",
        )
        input_ids_full = toks["input_ids"].to(device)
        attention_mask_full = toks["attention_mask"].to(device)

        # Shift for causal LM
        input_ids = input_ids_full[:, :-1]
        targets = input_ids_full[:, 1:]
        attention_mask = attention_mask_full[:, :-1]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # (B, S, vocab)

        vocab_size = logits.size(-1)
        loss = ce_loss(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # Memory logging (same idea as your script)
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

    print("DONE. Check W&B dashboard for metrics and memory curves.")

if __name__ == "__main__":
    main()
