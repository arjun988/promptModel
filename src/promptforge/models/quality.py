from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class PromptForgeQualityModel(nn.Module):
    """
    ModernBERT encoder + dual regression heads.

    Heads:
      - dimension_head: per-dimension scores in [0, 100]
      - quality_head: overall quality in [0, 100]
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_labels: int = 7,
        dropout: float = 0.1,
        dimension_loss_weight: float = 0.8,
        quality_loss_weight: float = 0.2,
        label_names: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout
        self.dimension_loss_weight = dimension_loss_weight
        self.quality_loss_weight = quality_loss_weight
        self.label_names = label_names or [
            "clarity",
            "specificity",
            "context",
            "goal_definition",
            "constraints",
            "completeness",
            "actionability",
        ]

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.dimension_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def mean_pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        dimension_logits = self.dimension_head(pooled)
        quality_logit = self.quality_head(pooled)

        dimension_predictions = torch.sigmoid(dimension_logits) * 100.0
        quality_prediction = torch.sigmoid(quality_logit) * 100.0

        loss = None
        if labels is not None:
            target_quality = labels.mean(dim=1, keepdim=True)
            dimension_loss = F.mse_loss(dimension_predictions, labels)
            quality_loss = F.mse_loss(quality_prediction, target_quality)
            loss = (
                self.dimension_loss_weight * dimension_loss
                + self.quality_loss_weight * quality_loss
            )

        return {
            "loss": loss,
            "logits": dimension_predictions,
            "quality": quality_prediction,
        }

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "model_type": "promptforge_quality",
            "base_model_name": self.model_name,
            "num_labels": self.num_labels,
            "dropout": self.dropout_prob,
            "dimension_loss_weight": self.dimension_loss_weight,
            "quality_loss_weight": self.quality_loss_weight,
            "label_names": self.label_names,
            "architectures": ["PromptForgeQualityModel"],
        }

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        with open(save_directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "PromptForgeQualityModel":
        load_directory = Path(load_directory)
        config_path = load_directory / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {load_directory}")

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        model = cls(
            model_name=cfg.get("base_model_name", "answerdotai/ModernBERT-base"),
            num_labels=int(cfg.get("num_labels", 7)),
            dropout=float(cfg.get("dropout", 0.1)),
            dimension_loss_weight=float(cfg.get("dimension_loss_weight", 0.8)),
            quality_loss_weight=float(cfg.get("quality_loss_weight", 0.2)),
            label_names=cfg.get("label_names"),
        )

        weights_path = load_directory / "pytorch_model.bin"
        safetensors_path = load_directory / "model.safetensors"
        if weights_path.exists():
            state = torch.load(weights_path, map_location=map_location or "cpu")
        elif safetensors_path.exists():
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path))
        else:
            raise FileNotFoundError(
                f"No pytorch_model.bin or model.safetensors in {load_directory}"
            )

        model.load_state_dict(state)
        return model


def save_quality_bundle(
    model: PromptForgeQualityModel,
    tokenizer: AutoTokenizer,
    save_directory: str | Path,
) -> Path:
    """Save model weights + tokenizer + config for Hub / local inference."""
    save_directory = Path(save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    return save_directory
