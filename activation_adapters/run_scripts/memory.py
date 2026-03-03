import os
import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForSequenceClassification,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
    LlamaForCausalLM,
    AutoModelForCausalLM
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
    measure_latency,
    build_boolq_dataset, boolq_collate_fn, boolq_evaluate, boolq_forward_step,
    build_piqa_dataset, piqa_collate_fn, piqa_forward_step, piqa_evaluate,
    build_hellaswag_dataset, hellaswag_collate_fn, hellaswag_forward_step, hellaswag_evaluate,
    build_siqa_dataset, siqa_collate_fn, siqa_forward_step, siqa_evaluate,
    build_arcc_dataset, arcc_collate_fn, arcc_evaluate, arcc_forward_step,
    build_quality_dataset, quality_collate_fn, quality_forward_step, quality_evaluate,
    build_qasper_dataset, build_hotpotqa_dataset, build_multidoc2dial_dataset,
    longqa_binary_evaluate, longqa_binary_forward_step, longqa_collate_fn,
    build_subject_dataset, collate_fn_subject, pack_10way_batch, mmlu_forward_step, evaluate_mmlu,
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
    parser.add_argument("--total_steps", type=int, default=1500)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--project", type=str, default="boolq_llama_peft")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--dataset",  type=str, default="boolq",
                        choices=["boolq", "piqa", "hellaswag", "siqa", "arc_c", "quality", "qasper", "hotpotqa", "multidoc2dial", "business", "biology", "law", "economics", "history", "physics","health", "math", "computer science"])
    parser.add_argument("--num_samples", type=int, default=-1, help="Num training samples (-1 for full)")
    parser.add_argument("--zero_shot", action="store_true", help="Run eval and exit")

    return parser.parse_args()



# -------------------------
# PEFT Config Factory
# -------------------------
def get_peft_config(method, rank, total_steps):

    if method == "lora":
        return LoraConfig(
            r=rank,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
        )

    elif method == "adalora":
        return AdaLoraConfig(
            r=rank,
            target_r=4,
            init_r=8,
            lora_alpha=16,
            task_type=TaskType.SEQ_CLS,
            total_step=total_steps,
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
        rank=rank,
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

    if dataset_name in ["biology", "business", "law", "economics", "history", "physics", "health", "math", "computer science"]:
        dataset = build_subject_dataset(tokenizer, max_len, dataset_name)
        evaluate_fn = evaluate_mmlu
        collate_fn = collate_fn_subject
        forward_step = mmlu_forward_step    
    
    if dataset_name == "siqa":
        dataset = build_siqa_dataset(tokenizer, max_len)
        evaluate_fn = siqa_evaluate
        collate_fn = siqa_collate_fn
        forward_step = siqa_forward_step

    if dataset_name == "arc_c":
        dataset = build_arcc_dataset(tokenizer, max_len)
        evaluate_fn = arcc_evaluate
        collate_fn = arcc_collate_fn
        forward_step = arcc_forward_step
    
    if dataset_name == "quality":
        dataset = build_quality_dataset(tokenizer, max_len)
        evaluate_fn = quality_evaluate
        collate_fn = quality_collate_fn
        forward_step = quality_forward_step

    elif dataset_name in ["qasper", "hotpotqa", "multidoc2dial"]:
        if dataset_name == "qasper":
            dataset = build_qasper_dataset(tokenizer, max_len)
        elif dataset_name == "hotpotqa":
            dataset = build_hotpotqa_dataset(tokenizer, max_len)
        else:
            dataset = build_multidoc2dial_dataset(tokenizer, max_len)
        
        evaluate_fn = longqa_binary_evaluate
        collate_fn = longqa_collate_fn
        forward_step = longqa_binary_forward_step
                
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=args.project, config=vars(args))

    MODEL_MAP = {
        "llama": "meta-llama/Llama-3.2-1B",
        "qwen": "Qwen/Qwen2.5-7B-Instruct"
    }
    model_id = MODEL_MAP[args.model_name]

    if args.dataset in ["biology", "business", "law", "economics", "history", "physics", "health", "math", "computer science"]:
        if args.model_name == "llama":
            model = LlamaForCausalLM.from_pretrained(
                model_id,
            )
        else: #qwen
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
            )
    else:
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

    if args.num_samples > 0:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(min(args.num_samples, len(dataset["train"]))))
        print(f"Dataset sampled to {len(dataset['train'])} examples.")
    
    actual_num_samples = len(dataset["train"])
    print(f">>> Training on {actual_num_samples} samples.")
    wandb.log({"num_samples": actual_num_samples})

    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size,
                              shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),)
    val_loader = DataLoader(dataset["validation"], batch_size=args.batch_size,
                            shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),)

    model.config.pad_token_id = tokenizer.pad_token_id

    # zero-shot
    if args.zero_shot:
        model.to(device)
        model.eval()
        acc = evaluate_fn(model, val_loader, device)
        wandb.log({"acc": acc, "num_samples": 0})
        return 
    
    # dynamic training steps
    steps_per_epoch = len(train_loader) // args.accum_steps
    if steps_per_epoch == 0: 
        steps_per_epoch = 1

    # if 0 < args.num_samples < 5000:
    #     max_steps = steps_per_epoch * args.epochs
    # else:
    #     max_steps = 1500
    
    # print(f">>> Training for {max_steps} total steps.")

    if args.checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Apply PEFT
    if args.peft_method == "bitfit":
        model = apply_bitfit(model)
    elif args.peft_method == "full":
        pass  # fine-tune all parameters, no adapter
    else:
        peft_config = get_peft_config(args.peft_method, args.rank, args.total_steps)
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

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=args.total_steps,
    )


    model.train()
    # Training loop
    step = 0
    total_tokens = 0
    latency, inf_tps = None, None
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()  # reset memory stats at the start of training
    torch.cuda.empty_cache()  # clear any cached memory

    # adaptive_eval = max(5, max_steps // 5)

    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            
            if step >= args.total_steps:
                break

            if args.dataset in ["biology", "business", "law", "economics", "history", "physics", "health", "math", "computer science"]:
                pad_id = model.config.pad_token_id
                ids, _, _ = pack_10way_batch(batch, device, pad_id)
                total_tokens += ids.numel()
            elif "input_ids" in batch:
                total_tokens += batch["input_ids"].numel()
            else:
                choice_keys = [k for k in batch.keys() if k.startswith("input_ids_")]
                for key in choice_keys:
                    total_tokens += batch[key].numel()

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
                
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

                # evaluate only every 50 steps
                acc = None

                if step % 50 == 0 and step > 0:
                    acc = evaluate_fn(model, val_loader, device)
                    if step % 100 == 0 and latency is None and args.dataset in ["boolq"]:
                        latency, inf_tps = measure_latency(model, val_loader, device)


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
                    log_dict["tokens_per_sec"] = tokens_per_sec
                if latency is not None:
                    log_dict["inference_latency_sec"] = latency
                    log_dict["inference_tokens_per_sec"] = inf_tps

                wandb.log(log_dict)
                print(f"Step {step:04d} | Loss {loss.item()*args.accum_steps:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} | MemPeak {mem_peak:.1f}MB" + (f" | Acc {acc:.4f}" if acc is not None else ""))

                step += 1
                total_tokens = 0

        if step >= args.total_steps:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()

