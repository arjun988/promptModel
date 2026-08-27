from __future__ import annotations

from pathlib import Path
from typing import Any

from promptforge.optimizer import PromptOptimizer
from promptforge.scorer import PromptQualityScorer


class PromptForge:
    """
    Public facade.

    Phase 1: analyze / score via PromptForge-Quality.
    Phase 2: optimize via PromptForge-Optimizer (LoRA).
    """

    def __init__(
        self,
        quality_model_path: str | Path | None = None,
        optimizer_model_path: str | Path | None = None,
        prefer_gpu: bool = True,
        max_length: int = 512,
        max_new_tokens: int = 512,
    ) -> None:
        self.prefer_gpu = prefer_gpu
        self.scorer: PromptQualityScorer | None = None
        self.optimizer: PromptOptimizer | None = None

        if quality_model_path is not None:
            self.scorer = PromptQualityScorer(
                model_path=quality_model_path,
                prefer_gpu=prefer_gpu,
                max_length=max_length,
            )

        if optimizer_model_path is not None:
            self.optimizer = PromptOptimizer(
                model_path=optimizer_model_path,
                prefer_gpu=prefer_gpu,
                max_new_tokens=max_new_tokens,
            )

        if self.scorer is None and self.optimizer is None:
            raise ValueError(
                "Provide quality_model_path and/or optimizer_model_path."
            )

    def analyze(self, prompt: str) -> dict[str, Any]:
        if self.scorer is None:
            raise RuntimeError("quality_model_path was not provided.")
        return self.scorer.analyze(prompt)

    def score(self, prompt: str) -> dict[str, Any]:
        if self.scorer is None:
            raise RuntimeError("quality_model_path was not provided.")
        return self.scorer.score(prompt)

    def optimize(
        self,
        prompt: str,
        analysis: dict[str, Any] | None = None,
        task_type: str = "general",
        use_scorer_analysis: bool = True,
    ) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError(
                "optimizer_model_path was not provided. "
                "Train/load PromptForge-Optimizer first."
            )

        if analysis is None and use_scorer_analysis and self.scorer is not None:
            analysis = self.scorer.analyze(prompt)

        result = self.optimizer.optimize(
            prompt,
            analysis=analysis,
            task_type=task_type,
        )
        if analysis is not None:
            result["analysis"] = analysis
        return result

    def analyze_and_optimize(
        self,
        prompt: str,
        task_type: str = "general",
    ) -> dict[str, Any]:
        analysis = self.analyze(prompt) if self.scorer is not None else None
        optimized = self.optimize(
            prompt,
            analysis=analysis,
            task_type=task_type,
            use_scorer_analysis=False,
        )
        return {
            "original_prompt": prompt,
            "analysis": analysis,
            "optimized_prompt": optimized["optimized_prompt"],
            "task_type": task_type,
        }
