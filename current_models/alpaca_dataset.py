from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_alpaca_tokenizer(model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def format_alpaca_example(row):
    instr = row["instruction"]
    inp = row.get("input", "")
    out = row["output"]

    if inp.strip():
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\n{inp}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr}\n\n"
            "### Response:\n"
        )
    return prompt, out

# In alpaca_dataset.py
def build_alpaca_dataset(tokenizer, max_len=256, mask_prompt=False):
    alpaca = load_dataset("yahma/alpaca-cleaned")["train"]

    def encode_row(row):
        prompt, answer = format_alpaca_example(row)
        full_text = prompt + answer

        toks = tokenizer(
            full_text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )

        input_ids = toks["input_ids"]
        labels = input_ids.copy()  # no -100 anywhere

        # TEMP: disable masking completely
        # if mask_prompt: ...

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": toks["attention_mask"],
        }

    encoded = alpaca.map(encode_row, remove_columns=alpaca.column_names)
    encoded.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    return encoded


def get_alpaca_dataloader(
    model_name="distilbert-base-uncased",
    max_len=256,
    batch_size=4,
    shuffle=True,
    mask_prompt=False,
):
    tokenizer = get_alpaca_tokenizer(model_name)
    dataset = build_alpaca_dataset(tokenizer, max_len=max_len, mask_prompt=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return tokenizer, loader
