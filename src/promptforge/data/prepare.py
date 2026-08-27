from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase


LABEL_COLUMNS = [
    "clarity",
    "specificity",
    "context",
    "goal_definition",
    "constraints",
    "completeness",
    "actionability",
]


def split_dataframe(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def dataframe_to_dataset_dict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )


def tokenize_and_label(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    label_columns: Sequence[str] = LABEL_COLUMNS,
) -> DatasetDict:
    label_columns = list(label_columns)

    def tokenize_function(batch: dict) -> dict:
        return tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "task_type"]
        if "task_type" in dataset["train"].column_names
        else ["prompt"],
    )

    # Drop optional helper columns if present.
    optional_drop = [c for c in ("quality_level",) if c in tokenized["train"].column_names]
    if optional_drop:
        tokenized = tokenized.remove_columns(optional_drop)

    def add_labels(example: dict) -> dict:
        example["labels"] = [float(example[col]) for col in label_columns]
        return example

    tokenized = tokenized.map(add_labels)

    drop_cols = [c for c in list(label_columns) + ["quality_score"] if c in tokenized["train"].column_names]
    tokenized = tokenized.remove_columns(drop_cols)
    return tokenized


def load_or_fail(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)
