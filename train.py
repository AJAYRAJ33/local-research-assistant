"""
Task 01 - Fine-tuning Script
Model  : microsoft/Phi-3-mini-4k-instruct
Method : QLoRA (4-bit quantization + LoRA adapters)
Trainer: HuggingFace TRL SFTTrainer

Hardware target: 8GB RAM, CPU-only
Run overnight: python train.py
"""

import os
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_from_disk

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./outputs"
ADAPTER_DIR = "./lora-adapter"
MAX_SEQ_LENGTH = 512       # keep low to save RAM
LORA_R = 16                # LoRA rank — higher = more capacity, more RAM
LORA_ALPHA = 32            # alpha = 2× rank is a safe default
LORA_DROPOUT = 0.05
BATCH_SIZE = 1             # MUST be 1 on 8GB RAM
GRAD_ACCUM = 8             # effective batch = 8
EPOCHS = 3
LR = 2e-4
# ──────────────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit QLoRA mode...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model):
    print(f"Applying LoRA: rank={LORA_R}, alpha={LORA_ALPHA}")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(ADAPTER_DIR).mkdir(exist_ok=True)

    print("Loading datasets...")
    train_dataset = load_from_disk("data/train")
    val_dataset = load_from_disk("data/val")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,           # CPU doesn't support fp16 training
        bf16=False,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,  # safer for CPU training
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    print("\nStarting training (this will take several hours on CPU)...")
    print("Consider running this overnight.\n")
    trainer.train()

    print("Saving LoRA adapter...")
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}/")


if __name__ == "__main__":
    train()
