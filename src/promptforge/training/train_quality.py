from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

from promptforge.config import QualityScorerConfig
from promptforge.data.generate import generate_dataset, summarize_dataset
from promptforge.data.prepare import (
    dataframe_to_dataset_dict,
    split_dataframe,
    tokenize_and_label,
)
from promptforge.models.quality import PromptForgeQualityModel, save_quality_bundle
from promptforge.training.metrics import (
    compute_metrics,
    overall_from_dimensions,
    per_dimension_report,
)
from promptforge.training.trainer import PromptForgeTrainer
from promptforge.utils.device import (
    describe_device,
    get_device,
    require_gpu,
    set_seed,
    training_precision_flags,
)


def build_training_args(config: QualityScorerConfig) -> TrainingArguments:
    precision = training_precision_flags(
        prefer_gpu=config.prefer_gpu,
        use_fp16=config.use_fp16,
        use_bf16=config.use_bf16,
    )
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )


def prepare_data(
    config: QualityScorerConfig,
    regenerate: bool = False,
) -> tuple[pd.DataFrame, Any]:
    dataset_path = Path(config.dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if regenerate or not dataset_path.exists():
        df = generate_dataset(num_examples=config.num_examples, seed=config.seed)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    summary = summarize_dataset(df)
    print("Dataset summary:", json.dumps(summary, indent=2, default=str))

    train_df, val_df, test_df = split_dataframe(
        df,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    print(
        f"Splits → train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    raw = dataframe_to_dataset_dict(train_df, val_df, test_df)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized = tokenize_and_label(
        raw,
        tokenizer=tokenizer,
        max_length=config.max_length,
        label_columns=config.label_names,
    )
    return df, (tokenizer, tokenized, train_df, val_df, test_df)


def train_quality_scorer(
    config: QualityScorerConfig,
    regenerate_dataset: bool = False,
) -> dict[str, Any]:
    """Full Phase-1 training loop. GPU-first."""
    require_gpu()
    set_seed(config.seed)

    device = get_device(prefer_gpu=config.prefer_gpu)
    print("Device:", describe_device(device))
    if device.type != "cuda":
        print(
            "WARNING: CUDA not available — training will be slow on CPU. "
            "Use Colab/GPU runtime for Phase 1."
        )

    _, packed = prepare_data(config, regenerate=regenerate_dataset)
    tokenizer, tokenized, _, _, _ = packed

    model = PromptForgeQualityModel(
        model_name=config.model_name,
        num_labels=config.num_labels,
        dropout=config.dropout,
        dimension_loss_weight=config.dimension_loss_weight,
        quality_loss_weight=config.quality_loss_weight,
        label_names=config.label_names,
    )
    model.to(device)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    training_args = build_training_args(config)

    trainer = PromptForgeTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])

    pred_out = trainer.predict(tokenized["test"])
    raw_preds = pred_out.predictions
    labels = pred_out.label_ids
    predictions = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds

    dim_report = per_dimension_report(predictions, labels, config.label_names)
    overall = overall_from_dimensions(predictions, labels)

    print("\n=== Validation ===")
    print(json.dumps(eval_metrics, indent=2))
    print("\n=== Test per-dimension ===")
    print(json.dumps(dim_report, indent=2))
    print("\n=== Test overall (mean of dims) ===")
    print(json.dumps(overall, indent=2))

    final_dir = Path(config.final_model_dir)
    save_quality_bundle(model, tokenizer, final_dir)
    config.save_yaml(final_dir / "training_config.yaml")

    metrics_path = Path(config.output_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": dict(train_result.metrics),
        "validation": eval_metrics,
        "test_per_dimension": dim_report,
        "test_overall": overall,
        "device": describe_device(device),
        "final_model_dir": str(final_dir),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved model bundle → {final_dir}")
    print(f"Saved metrics → {metrics_path}")
    return payload
