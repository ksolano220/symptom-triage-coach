"""LoRA fine-tune of Qwen2.5-1.5B-Instruct on (symptom, schema-valid JSON) pairs.

Reads data/processed/train.jsonl and val.jsonl produced by generate_data.py.
Wraps each row in the chat template with the pre-visit system prompt. Trains
a LoRA adapter targeting attention + MLP projections in 4-bit (QLoRA).

Designed for a single T4 GPU (Colab free tier), ~15-20 minutes for ~500 pairs.
"""

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "outputs/lora_weights"


def format_example(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(train_path, val_path, tokenizer):
    train_rows = [format_example(r, tokenizer) for r in load_jsonl(train_path)]
    val_rows = [format_example(r, tokenizer) for r in load_jsonl(val_path)]
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def attach_lora(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def train(train_path, val_path, hub_repo_id=None):
    model, tokenizer = load_base_model()
    model = attach_lora(model)
    train_ds, val_ds = build_dataset(train_path, val_path, tokenizer)

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch",
        report_to="none",
        max_length=768,
        packing=False,
        dataset_text_field="text",
        push_to_hub=bool(hub_repo_id),
        hub_model_id=hub_repo_id,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    if hub_repo_id:
        trainer.push_to_hub()
    return trainer


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    train(
        train_path=root / "data" / "processed" / "train.jsonl",
        val_path=root / "data" / "processed" / "val.jsonl",
    )
