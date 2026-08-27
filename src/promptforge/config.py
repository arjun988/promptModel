from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LABEL_NAMES = [
    "clarity",
    "specificity",
    "context",
    "goal_definition",
    "constraints",
    "completeness",
    "actionability",
]


@dataclass
class QualityScorerConfig:
    """Configuration for PromptForge Phase-1 quality scorer."""

    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    num_labels: int = 7
    label_names: list[str] = field(default_factory=lambda: list(DEFAULT_LABEL_NAMES))

    num_examples: int = 25_000
    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    early_stopping_patience: int = 2
    dropout: float = 0.1
    dimension_loss_weight: float = 0.8
    quality_loss_weight: float = 0.2

    prefer_gpu: bool = True
    use_fp16: bool = True
    use_bf16: bool = False

    output_dir: str = "outputs/promptforge-quality"
    dataset_path: str = "data/promptforge_dataset.csv"
    final_model_dir: str = "outputs/promptforge-quality-model"

    def __post_init__(self) -> None:
        if len(self.label_names) != self.num_labels:
            self.num_labels = len(self.label_names)

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train/val/test ratios must sum to 1.0, got {total:.4f}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityScorerConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QualityScorerConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
