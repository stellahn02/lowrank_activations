
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random

# ---------------- BoolQ ----------------
def build_boolq_dataset(tokenizer, max_len):
    ds = load_dataset("boolq")

    def preprocess(ex):
        text = f"Question: {ex['question']}\nPassage: {ex['passage']}"
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False, # Dynamic Padding
            # padding="max_length", # Static Padding
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(ex["answer"]),  # 0/1 for sequence classification
        }

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
    return ds

def boolq_collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [pad_id]*(max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0]*(max_len - len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

@torch.no_grad()
def boolq_evaluate(model, dev_loader, device):
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

def boolq_forward_step(accum_steps, model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss / accum_steps


# ---------------- PIQA ----------------

def build_piqa_dataset(tokenizer, max_len):
    ds = load_dataset("piqa")

    def preprocess(ex):
        goal = ex["goal"]
        sol1 = ex["sol1"]
        sol2 = ex["sol2"]

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

def piqa_collate_fn(batch, pad_id):
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


def piqa_forward_step(accum_steps, model, batch, device):
    ids0, msk0 = batch["input_ids_0"].to(device), batch["attention_mask_0"].to(device)
    ids1, msk1 = batch["input_ids_1"].to(device), batch["attention_mask_1"].to(device)
    labels = batch["labels"].to(device)
    
    logits0 = model(input_ids=ids0, attention_mask=msk0).logits
    logits1 = model(input_ids=ids1, attention_mask=msk1).logits
    
    # score = logit[1] - logit[0]
    scores = torch.stack([logits0[:, 1] - logits0[:, 0], 
                            logits1[:, 1] - logits1[:, 0]], dim=1)
    return F.cross_entropy(scores, labels) / accum_steps

@torch.no_grad()
def piqa_evaluate(model, dev_loader, device):
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

# ---------------- HellaSwag ----------------

def build_hellaswag_dataset(tokenizer, max_len):
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


def hellaswag_collate_fn(batch, pad_id):
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

    ids = torch.cat(input_ids_list, dim=0).to(device)   
    msk = torch.cat(attention_mask_list, dim=0).to(device)  
    lens = torch.cat(lengths_list, dim=0).to(device)    

    return ids, msk, lens

def hellaswag_forward_step(accum_steps, model, batch, device):
    pad_id = model.config.pad_token_id
    ids, msk, lens = pack_4way_batch(batch, device, pad_id)
    B = batch["labels"].size(0)
    labels = batch["labels"].to(device)
    
    logits = model(input_ids=ids, attention_mask=msk).logits
    # Normalized score: (logit[1] - logit[0]) / length
    scores = (logits[:, 1] - logits[:, 0]) / lens
    scores_4way = scores.view(4, B).transpose(0, 1).contiguous()
    return F.cross_entropy(scores_4way, labels) / accum_steps

@torch.no_grad()
def hellaswag_evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0
    pad_id = model.config.pad_token_id
    
    for batch in dev_loader:
        labels = batch["labels"].to(device)
        B = labels.size(0)

        ids, msk, lens = pack_4way_batch(batch, device, pad_id)         
        logits = model(input_ids=ids, attention_mask=msk).logits    
        

        scores = (logits[:, 1] - logits[:, 0]) / lens               
        scores_4way = scores.view(4, B).transpose(0, 1).contiguous()
        
        preds = scores_4way.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B

    model.train()
    return correct / total



# ---------------- MMLU ----------------

def build_subject_dataset(tokenizer, max_len, subject):
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    biology_test = ds["test"].filter(lambda x: x["category"] == subject)

    split = biology_test.train_test_split(test_size=0.1)
    train_ds, val_ds = split["train"], split["test"]

    ds_filtered = DatasetDict({"train": train_ds, "validation": val_ds})

    def preprocess(ex):
        question = ex["question"]
        label = int(ex["answer_index"])
        choices = ex["options"]
        choices = choices[:10] + [""] * (10 - len(choices))  # ensure 10 choices

        shuffled_idx = list(range(10))
        random.shuffle(shuffled_idx)
        shuffled_choices = [choices[i] for i in shuffled_idx]
        label = shuffled_idx.index(label)

        out = {"labels": label}
        for i in range(10):
            # Prompt with context
            prompt = (
                f"You are a highly knowledgeable expert in {subject}. Read the question carefully and choose the best answer. Explain your reasoning briefly before answering."
                f"Question: {question}\nChoices:\n" +
                "\n".join([f"{j+1}. {c}" for j, c in enumerate(shuffled_choices)]) +
                "\nAnswer:"
            )
            full_text = prompt + f" {shuffled_choices[i]}"

            enc_full = tokenizer(full_text, truncation=True, max_length=max_len, padding="max_length")
            enc_prompt = tokenizer(prompt, truncation=True, max_length=max_len, padding=False)
            prompt_len = len(enc_prompt["input_ids"])

            out[f"input_ids_{i}"] = enc_full["input_ids"]
            out[f"attention_mask_{i}"] = enc_full["attention_mask"]
            out[f"prompt_len_{i}"] = prompt_len
        return out

    for split_name in ["train", "validation"]:
        ds_filtered[split_name] = ds_filtered[split_name].map(
            preprocess,
            remove_columns=ds_filtered[split_name].column_names
        ).filter(lambda x: x is not None)

    return ds_filtered

# -----------------------------
# Collate function
# -----------------------------
def collate_fn_subject(batch, pad_id):
    def pad(seqs):
        maxlen = max(len(s) for s in seqs)
        return [s + [pad_id]*(maxlen-len(s)) for s in seqs]

    out = {}
    for i in range(10):
        ids = pad([x[f"input_ids_{i}"] for x in batch])
        msk = pad([x[f"attention_mask_{i}"] for x in batch])
        out[f"input_ids_{i}"] = torch.tensor(ids, dtype=torch.long)
        out[f"attention_mask_{i}"] = torch.tensor(msk, dtype=torch.long)
        out[f"prompt_len_{i}"] = torch.tensor([x[f"prompt_len_{i}"] for x in batch], dtype=torch.long)
    out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return out

# -----------------------------
# Pack 10-way batch for model
# -----------------------------
def pack_10way_batch(batch, device, pad_id):
    input_ids_list, attention_mask_list, prompt_len_list = [], [], []
    max_len = max(batch[f"input_ids_{i}"].size(1) for i in range(10))

    for i in range(10):
        ids, msk = batch[f"input_ids_{i}"], batch[f"attention_mask_{i}"]
        B, T = ids.shape
        if T < max_len:
            pad_len = max_len - T
            ids = torch.cat([ids, torch.full((B, pad_len), pad_id, dtype=ids.dtype)], dim=1)
            msk = torch.cat([msk, torch.zeros((B, pad_len), dtype=msk.dtype)], dim=1)
        input_ids_list.append(ids)
        attention_mask_list.append(msk)
        prompt_len_list.append(batch[f"prompt_len_{i}"])

    ids = torch.cat(input_ids_list, dim=0).to(device)
    msk = torch.cat(attention_mask_list, dim=0).to(device)
    prompt_lens = torch.cat(prompt_len_list, dim=0).to(device)
    return ids, msk, prompt_lens

# -----------------------------
# Forward + LoRA loss
# -----------------------------
def mmlu_forward_step(accum_steps, model, batch, device):
    pad_id = model.config.pad_token_id
    ids, msk, prompt_lens = pack_10way_batch(batch, device, pad_id)
    B = batch["labels"].size(0)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=ids, attention_mask=msk)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Compute NLL per choice
    losses_per_choice = []
    for i in range(10):
        choice_losses = []
        for b in range(B):
            idx = b + i*B
            start = prompt_lens[idx] - 1
            tokens = shift_labels[idx, start:] != pad_id
            masked_tokens = token_log_probs[idx, start:][tokens]
            if masked_tokens.numel() > 0:
                choice_losses.append(-masked_tokens.sum() / masked_tokens.numel())
            else:
                choice_losses.append(torch.tensor(0.0, device=device, requires_grad=True))
        losses_per_choice.append(torch.stack(choice_losses))
    nlls = torch.stack(losses_per_choice, dim=1)  # [B, 10]
    scores = -nlls
    # return F.cross_entropy(scores, labels) / accum_steps

    targets = F.one_hot(labels, num_classes=10).float()
    targets = targets * 0.9 + 0.1 / 10  # 0.9 confidence + 0.1 smoothing
    loss = -(F.log_softmax(scores, dim=-1) * targets).sum(dim=-1).mean()
    return loss / accum_steps

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_mmlu(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    pad_id = model.config.pad_token_id

    for batch in dataloader:
        B = batch["labels"].size(0)
        labels = batch["labels"].to(device)
        ids, msk, prompt_lens = pack_10way_batch(batch, device, pad_id)

        outputs = model(input_ids=ids, attention_mask=msk)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # NLL per choice
        losses_per_choice = []
        for i in range(10):
            choice_losses = []
            for b in range(B):
                idx = b + i*B
                start = prompt_lens[idx] - 1
                masked_tokens = token_log_probs[idx, start:][shift_labels[idx, start:] != pad_id]
                if masked_tokens.numel() == 0:
                    choice_losses.append(torch.tensor(0.0, device=device))
                else:
                    choice_losses.append(-masked_tokens.mean())
            losses_per_choice.append(torch.stack(choice_losses))
        nlls = torch.stack(losses_per_choice, dim=1)
        preds = nlls.argmin(dim=1)
        correct += (preds == labels).sum().item()
        total += B

    acc = correct / total
    # print(f"Subject Accuracy: {acc*100:.2f}%")
    model.train()
    return acc