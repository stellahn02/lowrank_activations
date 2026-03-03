
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random
import time

def measure_latency(model, dataloader, device, num_batches=20):
    model.eval()
    total_time = 0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            torch.cuda.synchronize()
            start = time.time()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)

            torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)

            if "input_ids" in batch:
                total_tokens += batch["input_ids"].numel()

    avg_latency = total_time / num_batches
    tokens_per_sec = total_tokens / total_time

    model.train()
    return avg_latency, tokens_per_sec

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

        score0 = logits0[:, 1] - logits0[:, 0]
        score1 = logits1[:, 1] - logits1[:, 0]
        pair_logits = torch.stack([score0, score1], dim=1)

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


# ---------------- SIQA ----------------

def build_siqa_dataset(tokenizer, max_len):
    # dataset: context, question, answerA, answerB, answerC
    ds = load_dataset("baber/social_i_qa")

    def preprocess(ex):
        ctx = ex["context"]
        q = ex["question"]
        options = [ex["answerA"], ex["answerB"], ex["answerC"]]
        
        # labels are strings "1", "2", "3". Convert to 0, 1, 2.
        try:
            label = int(ex["label"]) - 1
        except:
            return None

        out = {"labels": label}
        for i in range(3):
            text = f"Context: {ctx}\nQuestion: {q}\nAnswer: {options[i]}"
            enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            out[f"input_ids_{i}"] = enc["input_ids"]
            out[f"attention_mask_{i}"] = enc["attention_mask"]
            out[f"len_{i}"] = len(enc["input_ids"])
        return out

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
    ds = ds.filter(lambda x: x is not None)
    return ds

def siqa_collate_fn(batch, pad_id):
    def pad(seqs, pad_val):
        maxlen = max(len(s) for s in seqs)
        return [s + [pad_val] * (maxlen - len(s)) for s in seqs]

    out = {}
    for i in range(3):
        ids = pad([x[f"input_ids_{i}"] for x in batch], pad_id)
        msk = pad([x[f"attention_mask_{i}"] for x in batch], 0)
        out[f"input_ids_{i}"] = torch.tensor(ids, dtype=torch.long)
        out[f"attention_mask_{i}"] = torch.tensor(msk, dtype=torch.long)
        out[f"len_{i}"] = torch.tensor([x[f"len_{i}"] for x in batch], dtype=torch.float)

    out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return out

def pack_3way_batch(batch, device, pad_id):
    """Modified packing for 3-option SIQA"""
    input_ids_list, attention_mask_list, lengths_list = [], [], []
    max_t = max(batch[f"input_ids_{i}"].size(1) for i in range(3))

    for i in range(3):
        ids, msk = batch[f"input_ids_{i}"], batch[f"attention_mask_{i}"]
        B, T = ids.shape
        if T < max_t:
            ids = torch.cat([ids, torch.full((B, max_t - T), pad_id, dtype=torch.long)], dim=1)
            msk = torch.cat([msk, torch.zeros((B, max_t - T), dtype=torch.long)], dim=1)
        input_ids_list.append(ids)
        attention_mask_list.append(msk)
        lengths_list.append(batch[f"len_{i}"])

    return (torch.cat(input_ids_list, dim=0).to(device), 
            torch.cat(attention_mask_list, dim=0).to(device), 
            torch.cat(lengths_list, dim=0).to(device))

def siqa_forward_step(accum_steps, model, batch, device):
    pad_id = model.config.pad_token_id
    ids, msk, lens = pack_3way_batch(batch, device, pad_id)
    labels = batch["labels"].to(device)
    B = labels.size(0)
    
    logits = model(input_ids=ids, attention_mask=msk).logits
    scores = (logits[:, 1] - logits[:, 0]) / lens
    scores_3way = scores.view(3, B).transpose(0, 1).contiguous()
    return F.cross_entropy(scores_3way, labels) / accum_steps

@torch.no_grad()
def siqa_evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0
    pad_id = model.config.pad_token_id
    for batch in dev_loader:
        labels = batch["labels"].to(device)
        B = labels.size(0)
        ids, msk, lens = pack_3way_batch(batch, device, pad_id)
        logits = model(input_ids=ids, attention_mask=msk).logits
        scores = (logits[:, 1] - logits[:, 0]) / lens
        preds = scores.view(3, B).transpose(0, 1).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B
    model.train()
    return correct / total

# ---------------- ARC-Challenge ----------------

def build_arcc_dataset(tokenizer, max_len):
    ds = load_dataset("ai2_arc", "ARC-Challenge")

    def preprocess(ex):
        question = ex["question"]
        options = ex["choices"]["text"]
        labels_raw = ex["choices"]["label"]
        answer_key = ex["answerKey"]
        
        # Map 'A'-'E' or '1'-'5' to 0-4
        label_map = {str(l): i for i, l in enumerate(labels_raw)}
        try:
            label = label_map[str(answer_key)]
        except KeyError:
            return None 

        out = {"labels": label}
        # ARC-c is mostly 4 options, but some are 5. We prep for 5.
        for i in range(5):
            if i < len(options):
                text = f"Question: {question}\nAnswer: {options[i]}"
                enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
                out[f"input_ids_{i}"] = enc["input_ids"]
                out[f"attention_mask_{i}"] = enc["attention_mask"]
                out[f"len_{i}"] = len(enc["input_ids"])
            else:
                # Fill dummies for missing 5th options
                out[f"input_ids_{i}"] = []
                out[f"attention_mask_{i}"] = []
                out[f"len_{i}"] = 0.0
        return out

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
    ds = ds.filter(lambda x: x is not None)
    return ds

def arcc_collate_fn(batch, pad_id):
    def pad(seqs, pad_val):
        if not seqs or all(len(s) == 0 for s in seqs): return None
        maxlen = max(len(s) for s in seqs)
        return [s + [pad_val] * (maxlen - len(s)) for s in seqs]

    out = {}
    for i in range(5):
        raw_ids = [x[f"input_ids_{i}"] for x in batch]
        padded_ids = pad(raw_ids, pad_id)
        if padded_ids:
            out[f"input_ids_{i}"] = torch.tensor(padded_ids, dtype=torch.long)
            out[f"attention_mask_{i}"] = torch.tensor(pad([x[f"attention_mask_{i}"] for x in batch], 0), dtype=torch.long)
            out[f"len_{i}"] = torch.tensor([x[f"len_{i}"] for x in batch], dtype=torch.float)
        else:
            out[f"input_ids_{i}"] = None

    out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return out

def pack_5way_batch(batch, device, pad_id):
    input_ids_list, attention_mask_list, lengths_list = [], [], []
    active_indices = [i for i in range(5) if batch[f"input_ids_{i}"] is not None]
    max_t = max(batch[f"input_ids_{i}"].size(1) for i in active_indices)

    for i in range(5):
        if i in active_indices:
            ids = batch[f"input_ids_{i}"].to(device)
            msk = batch[f"attention_mask_{i}"].to(device)
            lens = batch[f"len_{i}"].to(device)
            
            B, T = ids.shape
            if T < max_t:
                padding_len = max_t - T
                ids = torch.cat([ids, torch.full((B, padding_len), pad_id, device=device, dtype=torch.long)], dim=1)
                msk = torch.cat([msk, torch.zeros((B, padding_len), device=device, dtype=torch.long)], dim=1)
            
            input_ids_list.append(ids)
            attention_mask_list.append(msk)
            lengths_list.append(lens) 
        else:
            B = batch["labels"].size(0)
            input_ids_list.append(torch.full((B, max_t), pad_id, device=device, dtype=torch.long))
            attention_mask_list.append(torch.zeros((B, max_t), device=device, dtype=torch.long))
            lengths_list.append(torch.ones(B, device=device, dtype=torch.float))

    return (torch.cat(input_ids_list, dim=0), 
            torch.cat(attention_mask_list, dim=0), 
            torch.cat(lengths_list, dim=0))

def arcc_forward_step(accum_steps, model, batch, device):
    pad_id = model.config.pad_token_id
    ids, msk, lens = pack_5way_batch(batch, device, pad_id)
    labels = batch["labels"].to(device)
    B = labels.size(0)
    
    logits = model(input_ids=ids, attention_mask=msk).logits
    # Using your standard binary logit difference
    scores = (logits[:, 1] - logits[:, 0]) / (lens + 1e-8)
    scores = torch.clamp(scores, min=-100, max=100)
    scores_5way = scores.view(5, B).transpose(0, 1).contiguous()
    return F.cross_entropy(scores_5way, labels) / accum_steps

@torch.no_grad()
def arcc_evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0
    pad_id = model.config.pad_token_id
    for batch in dev_loader:
        labels = batch["labels"].to(device)
        B = labels.size(0)
        ids, msk, lens = pack_5way_batch(batch, device, pad_id)
        logits = model(input_ids=ids, attention_mask=msk).logits
        scores = (logits[:, 1] - logits[:, 0]) / lens
        preds = scores.view(5, B).transpose(0, 1).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B
    model.train()
    return correct / total

import torch
from datasets import load_dataset
import torch.nn.functional as F

# ---------------- Common Helpers ----------------
def pad_sequence(seqs, pad_val):
    maxlen = max(len(s) for s in seqs)
    return [s + [pad_val] * (maxlen - len(s)) for s in seqs]

# ---------------- QuALITY ----------------
def build_quality_dataset(tokenizer, max_len):
    ds = load_dataset("emozilla/quality", "default")
    def preprocess(ex):
        question = ex["question"]
        article = ex["article"]
        out = {"labels": ex["answer"]}
        for i in range(4):
            text = f"Context: {article}\nQuestion: {question}\nAnswer: {ex['options'][i]}"
            enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            out[f"input_ids_{i}"] = enc["input_ids"]
            out[f"attention_mask_{i}"] = enc["attention_mask"]
            out[f"len_{i}"] = len(enc["input_ids"])
        return out
    return ds.map(preprocess, remove_columns=ds["train"].column_names)

def quality_collate_fn(batch, pad_id):
    out = {"labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long)}
    for i in range(4):
        out[f"input_ids_{i}"] = torch.tensor(pad_sequence([x[f"input_ids_{i}"] for x in batch], pad_id), dtype=torch.long)
        out[f"attention_mask_{i}"] = torch.tensor(pad_sequence([x[f"attention_mask_{i}"] for x in batch], 0), dtype=torch.long)
        out[f"len_{i}"] = torch.tensor([x[f"len_{i}"] for x in batch], dtype=torch.float)
    return out

def quality_forward_step(accum_steps, model, batch, device):
    pad_id = model.config.pad_token_id
    ids, msk, lens = pack_4way_batch(batch, device, pad_id)
    B = batch["labels"].size(0)
    
    logits = model(input_ids=ids, attention_mask=msk).logits
    scores = (logits[:, 1] - logits[:, 0]) / (lens + 1e-8)
    
    scores_4way = scores.view(4, B).transpose(0, 1).contiguous()
    return F.cross_entropy(scores_4way, batch["labels"].to(device)) / accum_steps

@torch.no_grad()
def quality_evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0
    pad_id = model.config.pad_token_id
    for batch in dev_loader:
        B = batch["labels"].size(0)
        ids, msk, lens = pack_4way_batch(batch, device, pad_id)
        
        logits = model(input_ids=ids, attention_mask=msk).logits
        scores = (logits[:, 1] - logits[:, 0]) / lens
        preds = scores.view(4, B).transpose(0, 1).argmax(dim=1)
        
        correct += (preds == batch["labels"].to(device)).sum().item()
        total += B
    model.train()
    return correct / total

# ---------------- Qasper ----------------
def build_qasper_dataset(tokenizer, max_len):
    ds = load_dataset("allenai/qasper")
    def preprocess(ex):
        question = ex["question"]
        full_text = " ".join([" ".join(p) for p in ex["full_text"]["paragraphs"]])
        enc = tokenizer(f"Question: {question}\nContext: {full_text}", truncation=True, max_length=max_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], 
                "labels": 0 if ex["answers"][0]["answer"][0]["unanswerable"] else 1}
    return ds.map(preprocess, remove_columns=ds["train"].column_names)

# ---------------- HotpotQA  ----------------
def build_hotpotqa_dataset(tokenizer, max_len):
    ds = load_dataset("hotpot_qa", "distractor")
    def preprocess(ex):
        question = ex["question"]
        full_ctx = " ".join([" ".join(p) for p in ex["context"]["sentences"]])
        enc = tokenizer(f"Question: {question}\nContext: {full_ctx}", truncation=True, max_length=max_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": 1 if ex["answer"].lower() == "yes" else 0}
    return ds.map(preprocess, remove_columns=ds["train"].column_names)

# ---------------- MultiDoc2Dial  ----------------
def build_multidoc2dial_dataset(tokenizer, max_len):
    ds = load_dataset("multidoc2dial", "multidoc2dial")
    def preprocess(ex):
        enc = tokenizer(f"Question: {ex['question']}\nDoc: {ex['context']}", truncation=True, max_length=max_len)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": 0}
    return ds.map(preprocess, remove_columns=ds["train"].column_names)

# ---------------- Shared Logic for Binary Long-QA ----------------
def longqa_collate_fn(batch, pad_id):
    # binary collation for HotpotQA, Qasper, MultiDoc2Dial, BoolQ
    max_l = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [pad_id]*(max_l - len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0]*(max_l - len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def longqa_binary_forward_step(accum_steps, model, batch, device):
     # forward step for HotpotQA, Qasper, MultiDoc2Dial"
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss / accum_steps

@torch.no_grad()
def longqa_binary_evaluate(model, dev_loader, device):
    # Evaluation for HotpotQA, Qasper, MultiDoc2Dial
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
