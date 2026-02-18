import math
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.backends.cuda import sdp_kernel
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LARSConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
import sys, site

print("PY:", sys.version)
print("EXE:", sys.executable)
print("SITES:", site.getsitepackages())

# ---------------- CONFIG ----------------
MODEL_ID = "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "llama_boolq_peft_quantized"

BATCH_SIZE = 8
STEPS = 3000
LR = 5e-4
WARMUP_STEPS = 300
MAX_LEN = 256
ACCUM_STEPS = 4  # gradient accumulation

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
    input_ids = [x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0] * (max_len - len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ---------------- EVALUATION ----------------
@torch.no_grad()
def evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0

    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return correct / total

# ---------------- HELPERS ----------------
def get_embed_weight(obj):
    # Works for raw HF model + PEFT-wrapped model
    if hasattr(obj, "get_input_embeddings") and obj.get_input_embeddings() is not None:
        return obj.get_input_embeddings().weight
    # fallback paths
    if hasattr(obj, "model") and hasattr(obj.model, "embed_tokens"):
        return obj.model.embed_tokens.weight
    if hasattr(obj, "base_model") and hasattr(obj.base_model, "model"):
        # common llama path (PEFT)
        return obj.base_model.model.model.embed_tokens.weight
    raise RuntimeError("Could not locate embed_tokens weight")

def print_embed_dtype(obj, tag):
    w = get_embed_weight(obj)
    print(tag, w.dtype, tuple(w.shape))

# ---------------- MAIN ----------------
def main():
    wandb.init(project=PROJECT_NAME, group="lars_4bit")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Tokenizer (FIXED) ----------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    # Ensure PAD exists for batch_size > 1 in LlamaForSequenceClassification
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pad_id = tokenizer.pad_token_id
    print("pad_token:", tokenizer.pad_token, "pad_id:", pad_id)
    assert pad_id is not None, "pad_token_id is None; cannot do batch_size>1 seq cls"

    # ---------------- Dataset ----------------
    boolq_ds = build_boolq_dataset(tokenizer, MAX_LEN)
    train_loader = DataLoader(
        boolq_ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    dev_loader = DataLoader(
        boolq_ds["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ---------------- Quantized model (8bit) ----------------
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
    )

    # 4-bit quantization
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",           
    #     bnb_4bit_use_double_quant=True,      
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    base_model = LlamaForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        quantization_config=bnb_config,
        device_map={"": 0},          # single GPU
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # ---------------- CRITICAL FIX: set pad_token_id on model config ----------------
    base_model.config.pad_token_id = pad_id
    # Optional consistency
    base_model.config.eos_token_id = tokenizer.eos_token_id

    print("base_model q_proj type:", type(base_model.model.layers[0].self_attn.q_proj))
    print("base_model.config.pad_token_id:", base_model.config.pad_token_id)

    # ---------------- LARS adapter config ----------------
    lars_config = LARSConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules="all-linear",
        fan_in_fan_out=True,
        rank=8,
        block_size=32,
        init_lars_weights=True,
    )

    # ---------------- prepare (k-bit training) ----------------
    base_model = prepare_model_for_kbit_training(base_model)
    print_embed_dtype(base_model, "after prepare")

    # OPTIONAL: cast embed_tokens back to bf16 (prepare_model_for_kbit_training often casts to fp32)
    base_model.model.embed_tokens = base_model.model.embed_tokens.to(torch.float16)
    print_embed_dtype(base_model, "after cast-back")

    def print_attn_debug(tag, m):
        cfg = m.config
        print(tag, "use_cache=", cfg.use_cache,
            "attn_impl=", getattr(cfg, "_attn_implementation", None),
            "output_attentions=", getattr(cfg, "output_attentions", None))

    # ---------------- wrap with PEFT ----------------
    model = get_peft_model(base_model, lars_config)

    # (belt + suspenders) ensure wrapper config also has pad id
    model.config.pad_token_id = pad_id
    print_attn_debug("[before GC]", base_model)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    print_attn_debug("[after  GC]", base_model)

    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())

    # ---------------- cast classifier head to bf16 (fixes eval matmul mismatch) ----------------
    base = model.base_model.model  # underlying LlamaForSequenceClassification

    if hasattr(base, "score") and base.score is not None:
        base.score = base.score.to(torch.bfloat16)

    # If PEFT wrapped score (ModulesToSaveWrapper), cast wrapped module too
    # from peft.utils.other import ModulesToSaveWrapper
    # for _, mod in model.named_modules():
    #     if isinstance(mod, ModulesToSaveWrapper):
    #         for k in list(mod.modules_to_save.keys()):
    #             mod.modules_to_save[k] = mod.modules_to_save[k].to(torch.bfloat16)

    print("score dtype:", getattr(base, "score", None).weight.dtype if hasattr(base, "score") else None)

    model.train()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

    # ---------------- Optimizer + scheduler ----------------
    optimizer = AdamW8bit(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=STEPS,
    )

    for n, p in model.named_parameters():
        if "U.weight" in n or "V.weight" in n or p.requires_grad:
            print(n, p.requires_grad)

    # ---------------- Training loop ----------------
    step = 0
    optimizer.zero_grad()
    printed_embed_dtype = False

    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= STEPS:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if step == 0 and batch_idx == 0:   # run once
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                # FORWARD
                print("[probe] batch seq_len:", input_ids.shape[1])
                print("[probe] baseline allocated MB:", torch.cuda.memory_allocated()/1e6)
                with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                torch.cuda.synchronize()
                fwd_peak = torch.cuda.max_memory_allocated() / 1e6
                print(f"[probe] peak after fwd MB: {fwd_peak:.1f}")

                # reset so backward peak is measured separately
                torch.cuda.reset_peak_memory_stats()

                # BACKWARD
                (out.loss / ACCUM_STEPS).backward()

                torch.cuda.synchronize()
                bwd_peak = torch.cuda.max_memory_allocated() / 1e6
                print(f"[probe] peak during bwd MB: {bwd_peak:.1f}")

                optimizer.zero_grad(set_to_none=True)
                continue

            # if not printed_embed_dtype:
            #     printed_embed_dtype = True

            #     def hook(mod, inp, out):
            #         print("embed out dtype:", out.dtype)

            #     h = model.base_model.model.model.embed_tokens.register_forward_hook(hook)

            #     with torch.autocast("cuda", dtype=torch.bfloat16):
            #         _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            #     h.remove()
            #     optimizer.zero_grad(set_to_none=True)
            #     continue

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / ACCUM_STEPS

            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                mem_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_peak  = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_reserved = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0

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
                print(
                    f"Step {step:04d} | Loss {loss.item()*ACCUM_STEPS:.4f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} | "
                    f"MemAlloc {mem_alloc:.1f}MB" + (f" | Acc {acc:.4f}" if acc is not None else "")
                )

                step += 1

        if step >= STEPS:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()
