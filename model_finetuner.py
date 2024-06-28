
import os
import glob
import pandas as pd
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, Trainer, TrainingArguments
from transformers.integrations import HfDeepSpeedConfig
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model finetuner.')
    parser.add_argument('--train_d', type=str, help='Path to train examples csv directory.')
    parser.add_argument('--dev_d', type=str, help='Path to validation examples csv directory.')
    parser.add_argument('--cp_d', type=str, help='Path to output directory for checkpoints.')

    return parser.parse_known_args()

def main():
    args, _ = parse_arguments()
    train_input_dir = os.path.abspath(args.train_d).rstrip("/\\")
    dev_input_dir = os.path.abspath(args.dev_d).rstrip("/\\")
    checkpoint_dir = os.path.abspath(args.cp_d).rstrip("/\\")
    model_name = "mt5-coref-pytorch/link-append-xxl"

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
        },
        "fp16": {
            "enabled": False,
        },
        "bf16": {
            "enabled": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_num_steps": 100,
            },
        },
        "activation_checkpointing": {
            "partition_activations": True,
        },
        "steps_per_print": 20,
        "wall_clock_breakdown": False,
    }

    # Avoids race condition errors
    def load_with_retry(files_path, retries=4, delay=5):
        for attempt in range(retries):
            try:
                return load_dataset("csv", data_files=files_path)
            except Exception as e:
                print(f"Load attempt: {attempt}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise e

    # Load datasets
    train_files = glob.glob(os.path.join(train_input_dir, "*.csv"))
    dev_files = glob.glob(os.path.join(dev_input_dir, "*.csv"))

    train_dataset = load_with_retry(train_files)["train"]
    eval_dataset = load_with_retry(dev_files)["train"]

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
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        learning_rate=1e-3,
        warmup_steps=100,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        save_steps=0.5,
        save_only_model=True,
        save_total_limit=1,
        prediction_loss_only=True,
        deepspeed=deepspeed_config,
        do_eval=True,
        evaluation_strategy="epoch",
        fp16=False,
        bf16=True
    )

    # Load model
    model_config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=model_config)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()
    print("Done.")

if __name__ == "__main__":
    main()
