from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from promptforge.config import OptimizerConfig
from promptforge.data.optimizer_generate import generate_optimizer_dataset
from promptforge.utils.device import (
    describe_device,
    get_device,
    require_gpu,
    set_seed,
    training_precision_flags,
)


def _enable_gpu_speedups() -> None:
    """Best-effort CUDA knobs for faster training on consumer GPUs."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def _split_df(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_ratio), random_state=seed
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _load_base_model_and_tokenizer(config: OptimizerConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if (config.use_fp16 and torch.cuda.is_available()) else None
    if config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    quant_config = None
    if config.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    return model, tokenizer


def _tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_seq_length: int,
) -> DatasetDict:
    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["training_text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )


class _CausalLMCollator(DataCollatorForLanguageModeling):
    """Pad labels with -100 to match input_ids (transformers 5.x batching)."""

    def torch_call(self, examples):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def train_optimizer(config: OptimizerConfig, regenerate_dataset: bool = False) -> dict[str, Any]:
    """Phase-2 LoRA SFT training. GPU-first."""
    require_gpu()
    set_seed(config.seed)
    _enable_gpu_speedups()

    device = get_device(prefer_gpu=config.prefer_gpu)
    print("Device:", describe_device(device))
    if device.type != "cuda":
        print("WARNING: CUDA not available — LoRA training will be very slow on CPU.")

    dataset_path = Path(config.dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if regenerate_dataset or not dataset_path.exists():
        df = generate_optimizer_dataset(
            num_examples=config.num_examples, seed=config.seed
        )
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    print(f"Optimizer dataset size: {len(df)}")
    train_df, val_df, test_df = _split_df(
        df,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    print(f"Splits → train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    raw = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )

    model, tokenizer = _load_base_model_and_tokenizer(config)

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    tokenized = _tokenize_dataset(raw, tokenizer, config.max_seq_length)
    precision = training_precision_flags(
        prefer_gpu=config.prefer_gpu,
        use_fp16=config.use_fp16,
        use_bf16=config.use_bf16,
    )

    train_size = len(tokenized["train"])
    steps_per_epoch = max(
        1,
        train_size
        // (
            config.per_device_train_batch_size
            * config.gradient_accumulation_steps
        ),
    )
    total_steps = steps_per_epoch * config.num_train_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=config.dataloader_num_workers,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    data_collator = _CausalLMCollator(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_dir = Path(config.final_model_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    meta = {
        "base_model_name": config.base_model_name,
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules,
        },
        "max_seq_length": config.max_seq_length,
        "max_new_tokens": config.max_new_tokens,
    }
    with open(final_dir / "optimizer_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    config.save_yaml(final_dir / "training_config.yaml")

    # Keep a small test sample for offline smoke checks
    test_df.head(50).to_csv(final_dir / "test_sample.csv", index=False)

    payload = {
        "train": dict(train_result.metrics),
        "validation": eval_metrics,
        "device": describe_device(device),
        "final_model_dir": str(final_dir),
        "base_model_name": config.base_model_name,
    }
    metrics_path = Path(config.output_dir) / "optimizer_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved LoRA adapter bundle → {final_dir}")
    print(f"Saved metrics → {metrics_path}")
    return payload


def load_optimizer_model(
    adapter_dir: str | Path,
    device: torch.device | None = None,
    prefer_gpu: bool = True,
):
    """Load base causal LM + LoRA adapter for inference."""
    adapter_dir = Path(adapter_dir)
    meta_path = adapter_dir / "optimizer_config.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_name = meta["base_model_name"]
    else:
        # PEFT adapter_config.json usually has base model
        peft_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = peft_cfg.get("base_model_name_or_path")
        if not base_name:
            raise FileNotFoundError(
                f"Could not resolve base model from {adapter_dir}"
            )
        meta = {"base_model_name": base_name, "max_new_tokens": 512}

    device = device or get_device(prefer_gpu=prefer_gpu)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    if device.type != "cuda":
        model.to(device)
    return model, tokenizer, meta
