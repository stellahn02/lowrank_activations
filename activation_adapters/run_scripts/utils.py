import torch
from datasets import load_dataset
import torch.nn.functional as F

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