from __future__ import annotations

from pathlib import Path
from typing import Any

from promptforge.scorer import PromptQualityScorer


class PromptForge:
    """
    Public facade.

    Phase 1: analyze / score via PromptForge-Quality.
    Phase 2: optimize will plug in PromptForge-Optimizer.
    """

    def __init__(
        self,
        quality_model_path: str | Path | None = None,
        prefer_gpu: bool = True,
        max_length: int = 512,
    ) -> None:
        if quality_model_path is None:
            raise ValueError(
                "quality_model_path is required for Phase 1. "
                "Pass a local directory or Hugging Face repo id after training/upload."
            )
        self.scorer = PromptQualityScorer(
            model_path=quality_model_path,
            prefer_gpu=prefer_gpu,
            max_length=max_length,
        )

    def analyze(self, prompt: str) -> dict[str, Any]:
        return self.scorer.analyze(prompt)

    def score(self, prompt: str) -> dict[str, Any]:
        return self.scorer.score(prompt)

    def optimize(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Prompt optimizer is Phase 2. Train/load PromptForge-Optimizer first."
        )
