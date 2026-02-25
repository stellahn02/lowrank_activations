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
    AutoModelForSequenceClassification,
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
from utils import ( 
    build_boolq_dataset, boolq_collate_fn, boolq_evaluate, boolq_forward_step,
    build_piqa_dataset, piqa_collate_fn, piqa_forward_step, piqa_evaluate,
    build_hellaswag_dataset, hellaswag_collate_fn, hellaswag_forward_step, hellaswag_evaluate
) 
# -------------------------
# Argument Parser
# -------------------------


#8, 256
#8, 512
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--peft_method", type=str, default="lora",
                        choices=["lora", "adalora", "ia3", "prefix", "prompt", "bitfit", "lars", "full"])
    parser.add_argument("--model_name", type=str, default="llama", 
                        choices=["llama", "qwen"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4) #LARS, LoRA
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--project", type=str, default="boolq_llama_peft")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--dataset",  type=str, default="boolq",
                        choices=["boolq", "piqa", "hellaswag"])

    return parser.parse_args()



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
            total_step=1500,
            target_modules="all-linear",
        )

    elif method == "ia3":
        return IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules="all-linear",
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
    dataset, evaluate_fn, collate_fn, forward_step = None, None, None, None
    
    if dataset_name == "boolq":
        dataset = build_boolq_dataset(tokenizer, max_len)
        evaluate_fn = boolq_evaluate
        collate_fn = boolq_collate_fn
        forward_step = boolq_forward_step
        
    if dataset_name == "piqa":
        dataset = build_piqa_dataset(tokenizer, max_len)
        evaluate_fn = piqa_evaluate
        collate_fn = piqa_collate_fn
        forward_step = piqa_forward_step

    if dataset_name == "hellaswag":
        dataset = build_hellaswag_dataset(tokenizer, max_len)
        evaluate_fn = hellaswag_evaluate
        collate_fn = hellaswag_collate_fn
        forward_step = hellaswag_forward_step     
    
    return dataset, evaluate_fn, collate_fn, forward_step

def get_lr(method):
    if method == "lora" or method == "lars":
        return 1e-4
    if method == "prompt":
        return 1e-5
    if method == "prefix":
        return 1e-5
    elif method == "adalora":
        return 5e-4
    elif method == "ia3":
        return 5e-3
    elif method == "full":
        return 2e-5
    else:
        raise ValueError(f"Unknown PEFT method: {method}")    


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
# Train
# -------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=args.project, config=vars(args))

    MODEL_MAP = {
        "llama": "meta-llama/Llama-3.2-1B",
        "qwen": "Qwen/Qwen2.5-7B-Instruct"
    }
    model_id = MODEL_MAP[args.model_name]

    if args.model_name == "llama":
        model = LlamaForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
        )
    else: #qwen
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # dataset = build_boolq_dataset(tokenizer, args.max_len)
    dataset, evaluate_fn, collate_fn, forward_step = get_dataset(args.dataset, tokenizer, args.max_len)

    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size,
                              shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),)
    val_loader = DataLoader(dataset["validation"], batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),)

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

    if args.model_name == "qwen":
        model.to(torch.bfloat16)
        if hasattr(model, "score"):
            model.score.to(torch.bfloat16)

    args.lr = get_lr(args.peft_method)
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
    torch.cuda.reset_peak_memory_stats()  # reset memory stats at the start of training
    torch.cuda.empty_cache()  # clear any cached memory
    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= 1500:
                break

            loss = forward_step(args.accum_steps, model, batch, device)
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
                    acc = evaluate_fn(model, val_loader, device)

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
                    log_dict["acc"] = acc

                wandb.log(log_dict)
                print(f"Step {step:04d} | Loss {loss.item()*args.accum_steps:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} | MemPeak {mem_peak:.1f}MB" + (f" | Acc {acc:.4f}" if acc is not None else ""))

                step += 1

        if step >= 1500:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()


