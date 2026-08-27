from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from promptforge.analyzer import PromptAnalyzer
from promptforge.comparison import build_comparison_payload
from promptforge.optimizer import PromptOptimizer
from promptforge.scorer import PromptQualityScorer


class PromptForge:
    """
    Combined PromptForge pipeline (Phase 3).

    Prompt → Quality Scorer → Optimizer → (optional) re-score → comparison
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
        self.analyzer: PromptAnalyzer | None = None

        if quality_model_path is not None:
            self.scorer = PromptQualityScorer(
                model_path=quality_model_path,
                prefer_gpu=prefer_gpu,
                max_length=max_length,
            )
            self.analyzer = PromptAnalyzer(self.scorer)

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
        if self.analyzer is None:
            raise RuntimeError("quality_model_path was not provided.")
        return self.analyzer.analyze(prompt)

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
            analysis = self.analyze(prompt)

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
        rescore_optimized: bool = True,
    ) -> dict[str, Any]:
        return self.run(
            prompt,
            task_type=task_type,
            rescore_optimized=rescore_optimized,
        )

    def run(
        self,
        prompt: str,
        task_type: str = "general",
        rescore_optimized: bool = True,
    ) -> dict[str, Any]:
        """
        Full Phase-3 pipeline with before/after comparison.

        Returns structured JSON suitable for API / Space / CLI.
        """
        if self.scorer is None or self.optimizer is None:
            raise RuntimeError(
                "Combined pipeline requires both quality_model_path and optimizer_model_path."
            )

        before = self.analyze(prompt)
        optimized = self.optimize(
            prompt,
            analysis=before,
            task_type=task_type,
            use_scorer_analysis=False,
        )
        optimized_prompt = optimized["optimized_prompt"]

        after: dict[str, Any] | None = None
        if rescore_optimized:
            after = self.analyze(optimized_prompt)

        return build_comparison_payload(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            before_analysis=before,
            after_analysis=after,
            task_type=task_type,
        )

    def export_result(self, result: dict[str, Any], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return path
