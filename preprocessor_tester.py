import os
import glob
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer

train_input_dir = r".\train_examples"
model_name = "mt5-coref-pytorch/link-append-xxl"

train_files = glob.glob(os.path.join(train_input_dir, "*.csv"))

train_dataset = load_dataset("csv", data_files=train_files)["train"]
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
def preprocess_function(batch):
    inputs = [example for example in batch["input"]]
    outputs = [example if not pd.isna(example) else "" for example in batch["output"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Tokenize outputs and use them as labels
    labels = tokenizer(
        outputs,
        return_tensors="pt",
        max_length=384,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Apply preprocessing to datasets
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

for row in train_dataset.select(range(100)):
    print(row)
    print(tokenizer.decode(row["labels"], skip_special_tokens=True))