
import torch
import torch.nn.functional as F
import string
import json
import re
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random
import time
import collections
from collections import Counter


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
        options = ex["options"]
        label = int(ex["answer"])

        out = {"labels": label}

        old_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        try:
            for i in range(4):
                prompt = f"{article}\n\nQuestion: {question}\nAnswer:"
                enc_prompt = tokenizer(prompt, add_special_tokens=False)

                ans_text = " " + options[i]
                enc_ans = tokenizer(ans_text, add_special_tokens=False)

                max_prompt = max_len - len(enc_ans["input_ids"]) - 1
                if max_prompt < 32:
                    max_prompt = 32

                prompt_ids = enc_prompt["input_ids"][-max_prompt:]
                input_ids = prompt_ids + enc_ans["input_ids"]
                input_ids = input_ids[:max_len]

                out[f"input_ids_{i}"] = input_ids
                out[f"attention_mask_{i}"] = [1] * len(input_ids)
                out[f"prompt_len_{i}"] = min(len(prompt_ids), len(input_ids))
        finally:
            tokenizer.truncation_side = old_side

        return out

    return ds.map(
        preprocess,
        remove_columns=ds["train"].column_names,
        num_proc=12,
    )


def quality_collate_fn(batch, pad_id):
    def pad_sequence(seqs, pad_val):
        maxlen = max(len(s) for s in seqs)
        return [s + [pad_val] * (maxlen - len(s)) for s in seqs]

    out = {
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    }

    for i in range(4):
        ids_list = [x[f"input_ids_{i}"] for x in batch]
        msk_list = [x[f"attention_mask_{i}"] for x in batch]

        out[f"input_ids_{i}"] = torch.tensor(
            pad_sequence(ids_list, pad_id), dtype=torch.long
        )
        out[f"attention_mask_{i}"] = torch.tensor(
            pad_sequence(msk_list, 0), dtype=torch.long
        )
        out[f"prompt_len_{i}"] = torch.tensor(
            [x[f"prompt_len_{i}"] for x in batch], dtype=torch.long
        )

    return out


def quality_pack_4way_batch(batch, device, pad_id):
    input_ids_list = []
    attention_mask_list = []

    max_t = max(batch[f"input_ids_{i}"].size(1) for i in range(4))

    for i in range(4):
        ids = batch[f"input_ids_{i}"]
        msk = batch[f"attention_mask_{i}"]
        B, T = ids.shape

        if T < max_t:
            padding_len = max_t - T
            ids = torch.cat(
                [ids, torch.full((B, padding_len), pad_id, dtype=torch.long)], dim=1
            )
            msk = torch.cat(
                [msk, torch.zeros((B, padding_len), dtype=torch.long)], dim=1
            )

        input_ids_list.append(ids)
        attention_mask_list.append(msk)

    ids = torch.cat(input_ids_list, dim=0).to(device)   # [4B, T]
    msk = torch.cat(attention_mask_list, dim=0).to(device)  # [4B, T]
    return ids, msk


def quality_forward_step(accum_steps, model, batch, device):
    pad_id = model.config.pad_token_id
    ids, msk = quality_pack_4way_batch(batch, device, pad_id)

    B = batch["labels"].size(0)
    labels = batch["labels"].to(device)
    prompt_lens = torch.cat([batch[f"prompt_len_{i}"] for i in range(4)], dim=0).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=msk).logits   # [4B, T, V]

    shift_logits = logits[:, :-1, :]
    shift_ids = ids[:, 1:]                  # tokens being predicted
    shift_msk = msk[:, 1:].float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    tok_logp = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)  # [4B, T-1]

    Tm1 = shift_ids.size(1)
    positions = torch.arange(Tm1, device=device).unsqueeze(0)  # [1, T-1]

    # answer starts at token index prompt_len in ids space,
    # so in shifted space it starts at prompt_len - 1
    answer_mask = (positions >= (prompt_lens.unsqueeze(1) - 1)).float()
    answer_mask = answer_mask * shift_msk

    sum_logp = (tok_logp * answer_mask).sum(dim=1)      # [4B]
    cnt = answer_mask.sum(dim=1).clamp_min(1.0)         # [4B]
    nll = -(sum_logp / cnt)                             # [4B]

    scores = (-nll).view(4, B).transpose(0, 1).contiguous()   # [B, 4]
    loss = F.cross_entropy(scores, labels)

    return loss / accum_steps


@torch.no_grad()
def quality_evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0
    pad_id = model.config.pad_token_id

    with torch.inference_mode():
        for batch in dev_loader:
            ids, msk = quality_pack_4way_batch(batch, device, pad_id)

            B = batch["labels"].size(0)
            labels = batch["labels"].to(device)
            prompt_lens = torch.cat([batch[f"prompt_len_{i}"] for i in range(4)], dim=0).to(device)

            logits = model(input_ids=ids, attention_mask=msk).logits

            shift_logits = logits[:, :-1, :]
            shift_ids = ids[:, 1:]
            shift_msk = msk[:, 1:].float()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            tok_logp = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)

            Tm1 = shift_ids.size(1)
            positions = torch.arange(Tm1, device=device).unsqueeze(0)
            answer_mask = (positions >= (prompt_lens.unsqueeze(1) - 1)).float()
            answer_mask = answer_mask * shift_msk

            sum_logp = (tok_logp * answer_mask).sum(dim=1)
            cnt = answer_mask.sum(dim=1).clamp_min(1.0)
            nll = -(sum_logp / cnt)

            scores = (-nll).view(4, B).transpose(0, 1).contiguous()
            preds = scores.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += B

            del logits, shift_logits, log_probs, tok_logp, nll, scores

    model.train()
    return correct / total

# ---------------- RACE ----------------

def build_race_dataset(tokenizer, max_len, subset="all", limit_train=None, limit_val=None):
    ds = load_dataset("race", subset)

    if limit_train is not None and limit_train > 0:
        ds["train"] = ds["train"].select(range(min(limit_train, len(ds["train"]))))

    if limit_val is not None and limit_val > 0:
        ds["validation"] = ds["validation"].select(range(min(limit_val, len(ds["validation"]))))

    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    def preprocess(ex):
        article = ex["article"]
        question = ex["question"]
        options = ex["options"]
        answer = ex["answer"]

        if options is None or len(options) != 4:
            return {"keep": False}

        if answer not in answer_map:
            return {"keep": False}

        label = answer_map[answer]

        out = {
            "labels": label,
            "question_text": question,
            "options_text": options,
            "gold_answer_letter": answer,
            "keep": True,
        }

        old_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        try:
            prompt = (
                f"Passage:\n{article}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
            enc_prompt = tokenizer(prompt, add_special_tokens=False)

            for i in range(4):
                ans_text = " " + options[i]
                enc_ans = tokenizer(ans_text, add_special_tokens=False)

                max_prompt = max_len - len(enc_ans["input_ids"])
                max_prompt = max(32, max_prompt)

                prompt_ids = enc_prompt["input_ids"][-max_prompt:]
                input_ids = (prompt_ids + enc_ans["input_ids"])[:max_len]

                out[f"input_ids_{i}"] = input_ids
                out[f"attention_mask_{i}"] = [1] * len(input_ids)
                out[f"prompt_len_{i}"] = min(len(prompt_ids), len(input_ids))
        finally:
            tokenizer.truncation_side = old_side

        return out

    processed = {}
    for split_name in ["train", "validation"]:
        processed[split_name] = ds[split_name].map(
            preprocess,
            remove_columns=ds[split_name].column_names,
            load_from_cache_file=False,
        )
        processed[split_name] = processed[split_name].filter(
            lambda x: x["keep"],
            load_from_cache_file=False,
        )
        if "keep" in processed[split_name].column_names:
            processed[split_name] = processed[split_name].remove_columns(["keep"])

    return processed


def race_collate_fn(batch, pad_id):
    def pad_sequence(seqs, pad_val):
        maxlen = max(len(s) for s in seqs)
        return [s + [pad_val] * (maxlen - len(s)) for s in seqs]

    out = {
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        "question_text": [x["question_text"] for x in batch],
        "options_text": [x["options_text"] for x in batch],
        "gold_answer_letter": [x["gold_answer_letter"] for x in batch],
    }

    for i in range(4):
        ids_list = [x[f"input_ids_{i}"] for x in batch]
        msk_list = [x[f"attention_mask_{i}"] for x in batch]

        out[f"input_ids_{i}"] = torch.tensor(
            pad_sequence(ids_list, pad_id), dtype=torch.long
        )
        out[f"attention_mask_{i}"] = torch.tensor(
            pad_sequence(msk_list, 0), dtype=torch.long
        )
        out[f"prompt_len_{i}"] = torch.tensor(
            [x[f"prompt_len_{i}"] for x in batch], dtype=torch.long
        )

    return out


def race_pack_4way_batch(batch, device, pad_id):
    input_ids_list = []
    attention_mask_list = []

    max_t = max(batch[f"input_ids_{i}"].size(1) for i in range(4))

    for i in range(4):
        ids = batch[f"input_ids_{i}"]
        msk = batch[f"attention_mask_{i}"]
        B, T = ids.shape

        if T < max_t:
            padding_len = max_t - T
            ids = torch.cat(
                [ids, torch.full((B, padding_len), pad_id, dtype=torch.long)], dim=1
            )
            msk = torch.cat(
                [msk, torch.zeros((B, padding_len), dtype=torch.long)], dim=1
            )

        input_ids_list.append(ids)
        attention_mask_list.append(msk)

    ids = torch.cat(input_ids_list, dim=0).to(device)   # [4B, T]
    msk = torch.cat(attention_mask_list, dim=0).to(device)  # [4B, T]
    return ids, msk


def compute_race_scores(model, batch, device):
    """
    Returns average answer-token NLL per option: [B, 4]
    Lower is better.
    """
    pad_id = model.config.pad_token_id
    ids, msk = race_pack_4way_batch(batch, device, pad_id)

    B = batch["labels"].size(0)
    prompt_lens = torch.cat([batch[f"prompt_len_{i}"] for i in range(4)], dim=0).to(device)

    logits = model(input_ids=ids, attention_mask=msk).logits   # [4B, T, V]

    shift_logits = logits[:, :-1, :]
    shift_ids = ids[:, 1:]
    shift_msk = msk[:, 1:].float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    tok_logp = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)  # [4B, T-1]

    Tm1 = shift_ids.size(1)
    positions = torch.arange(Tm1, device=device).unsqueeze(0)

    # answer starts at original token index prompt_len
    # shifted positions correspond to original token index = pos + 1
    answer_mask = (positions + 1 >= prompt_lens.unsqueeze(1)).float()
    answer_mask = answer_mask * shift_msk

    sum_logp = (tok_logp * answer_mask).sum(dim=1)
    cnt = answer_mask.sum(dim=1).clamp_min(1.0)
    nll = -(sum_logp / cnt)   # [4B]

    scores = nll.view(4, B).transpose(0, 1).contiguous()   # [B, 4], lower is better
    return scores


def race_forward_step(accum_steps, model, batch, device):
    labels = batch["labels"].to(device)
    scores = compute_race_scores(model, batch, device)   # lower = better
    loss = F.cross_entropy(-scores, labels)
    return loss / accum_steps


@torch.no_grad()
def race_evaluate(model, dev_loader, device, verbose=False):
    model.eval()
    correct = total = 0

    answer_letters = ["A", "B", "C", "D"]

    with torch.inference_mode():
        for batch in dev_loader:
            labels = batch["labels"].to(device)
            scores = compute_race_scores(model, batch, device)   # [B, 4], lower better
            preds = scores.argmin(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if verbose:
                ex_nll = scores.detach().cpu()
                for i in range(labels.size(0)):
                    print("=" * 100)
                    print("QUESTION:")
                    print(batch["question_text"][i])
                    print("\nOPTIONS:")
                    for j, opt in enumerate(batch["options_text"][i]):
                        marker = ""
                        if j == labels[i].item():
                            marker += " [GOLD]"
                        if j == preds[i].item():
                            marker += " [PRED]"
                        print(f"  {answer_letters[j]}. {opt}  nll={ex_nll[i, j].item():.4f}{marker}")

                    print(f"\nGOLD: {answer_letters[labels[i].item()]}")
                    print(f"PRED: {answer_letters[preds[i].item()]}")

    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def race_evaluate_verbose(model, dev_loader, device, max_print=10):
    model.eval()
    correct = total = 0

    pred_counts = Counter()
    gold_counts = Counter()

    printed = 0

    for batch in dev_loader:
        labels = batch["labels"].to(device)
        scores = compute_race_scores(model, batch, device)   # [B, 4], lower is better
        preds = scores.argmin(dim=1)

        B = labels.size(0)

        correct += (preds == labels).sum().item()
        total += B

        for i in range(B):
            pred_idx = preds[i].item()
            gold_idx = labels[i].item()

            pred_counts[pred_idx] += 1
            gold_counts[gold_idx] += 1

            if printed < max_print:
                print("=" * 100)
                print("QUESTION:")
                print(batch["question_text"][i])
                print("\nOPTIONS:")
                for j, opt in enumerate(batch["options_text"][i]):
                    marker = ""
                    if j == gold_idx:
                        marker += " [GOLD]"
                    if j == pred_idx:
                        marker += " [PRED]"
                    print(f"  {j} {opt}{marker}")

                print("\nSCORES (lower is better):")
                ex_scores = scores[i].detach().cpu().tolist()
                for j, s in enumerate(ex_scores):
                    print(f"  {j}: {s:.4f}")

                print(f"\nGOLD: {gold_idx} ({batch['gold_answer_letter'][i]})")
                print(f"PRED: {pred_idx}")
                printed += 1

    acc = correct / max(total, 1)

    print("\n" + "=" * 100)
    print(f"RACE accuracy: {acc:.4f}")
    print("Prediction counts:", dict(pred_counts))
    print("Gold counts:", dict(gold_counts))

    total_preds = sum(pred_counts.values())
    if total_preds > 0:
        print("Prediction fractions:")
        for k in range(4):
            print(f"  option {k}: {pred_counts[k] / total_preds:.4f}")

    model.train()
    return acc, pred_counts, gold_counts

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
                "\nID:"
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