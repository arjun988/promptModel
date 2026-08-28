from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from promptforge.analyzer import PromptAnalyzer
from promptforge.comparison import build_comparison_payload
from promptforge.evaluation import instruction_preservation
from promptforge.local_paths import prefer_gpu_from_env, resolve_model_path
from promptforge.optimizer import PromptOptimizer
from promptforge.scorer import PromptQualityScorer

_DEFAULT_MAX_NEW_TOKENS = 256


class PromptForge:
    """
    Combined PromptForge pipeline (Phases 1–4).

    Model paths resolve from:
      args → env → ~/.promptforge/config.yaml → ~/.promptforge/models → ./outputs
    """

    def __init__(
        self,
        quality_model_path: str | Path | None = None,
        optimizer_model_path: str | Path | None = None,
        prefer_gpu: bool | None = None,
        max_length: int = 512,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        require_quality: bool = False,
        require_optimizer: bool = False,
    ) -> None:
        if prefer_gpu is None:
            prefer_gpu = prefer_gpu_from_env(True)

        self.prefer_gpu = prefer_gpu
        self.scorer: PromptQualityScorer | None = None
        self.optimizer: PromptOptimizer | None = None
        self.analyzer: PromptAnalyzer | None = None

        quality_resolved = resolve_model_path(
            quality_model_path, kind="quality", allow_missing=True
        )
        optimizer_resolved = resolve_model_path(
            optimizer_model_path, kind="optimizer", allow_missing=True
        )

        if require_quality and not quality_resolved:
            raise ValueError(
                "quality model not found. Pass quality_model_path, set "
                "PROMPTFORGE_QUALITY_MODEL, or run: promptforge download"
            )
        if require_optimizer and not optimizer_resolved:
            raise ValueError(
                "optimizer model not found. Pass optimizer_model_path, set "
                "PROMPTFORGE_OPTIMIZER_MODEL, or run: promptforge download"
            )

        if quality_resolved is not None:
            self.scorer = PromptQualityScorer(
                model_path=quality_resolved,
                prefer_gpu=prefer_gpu,
                max_length=max_length,
            )
            self.analyzer = PromptAnalyzer(self.scorer)

        if optimizer_resolved is not None:
            self.optimizer = PromptOptimizer(
                model_path=optimizer_resolved,
                prefer_gpu=prefer_gpu,
                max_new_tokens=max_new_tokens,
            )

        if self.scorer is None and self.optimizer is None:
            raise ValueError(
                "No models found. Install models with:\n"
                "  promptforge download\n"
                "or pass quality_model_path / optimizer_model_path."
            )

        self.quality_model_path = quality_resolved
        self.optimizer_model_path = optimizer_resolved

    @classmethod
    def from_config(
        cls,
        prefer_gpu: bool | None = None,
        **kwargs: Any,
    ) -> "PromptForge":
        """Load using local config / env / default model dirs only."""
        return cls(prefer_gpu=prefer_gpu, **kwargs)

    def analyze(self, prompt: str) -> dict[str, Any]:
        if self.analyzer is None:
            raise RuntimeError("quality model not loaded.")
        return self.analyzer.analyze(prompt)

    def score(self, prompt: str) -> dict[str, Any]:
        if self.scorer is None:
            raise RuntimeError("quality model not loaded.")
        return self.scorer.score(prompt)

    def optimize(
        self,
        prompt: str,
        analysis: dict[str, Any] | None = None,
        task_type: str = "general",
        use_scorer_analysis: bool = True,
        validate: bool = True,
        fallback_on_invalid: bool = True,
    ) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("optimizer model not loaded.")

        if analysis is None and use_scorer_analysis and self.scorer is not None:
            analysis = self.analyze(prompt)

        result = self.optimizer.optimize(
            prompt,
            analysis=analysis,
            task_type=task_type,
            validate=validate,
            fallback_on_invalid=fallback_on_invalid,
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
        if self.scorer is None or self.optimizer is None:
            raise RuntimeError(
                "Combined pipeline requires both quality and optimizer models."
            )

        before = self.analyze(prompt)
        optimized = self.optimize(
            prompt,
            analysis=before,
            task_type=task_type,
            use_scorer_analysis=False,
        )
        optimized_prompt = optimized["optimized_prompt"]
        validation = optimized.get("validation") or {}

        after: dict[str, Any] | None = None
        if rescore_optimized:
            after = self.analyze(optimized_prompt)

        payload = build_comparison_payload(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            before_analysis=before,
            after_analysis=after,
            task_type=task_type,
        )

        intent = validation.get(
            "instruction_preservation",
            instruction_preservation(prompt, optimized_prompt),
        )
        used_fallback = optimized.get("used_fallback", False)
        score_delta = payload["delta"]["quality_score"]
        trustworthy = bool(
            not used_fallback
            and validation.get("valid", True)
            and intent >= 0.08
            and not validation.get("repetitive", False)
            and (score_delta <= 0 or intent >= 0.05)
        )

        payload["validation"] = validation
        payload["model_validation"] = optimized.get("model_validation")
        payload["raw_output"] = optimized.get("raw_output")
        payload["model_output"] = optimized.get("model_output")
        payload["used_fallback"] = used_fallback
        payload["trustworthy_improvement"] = trustworthy
        return payload

    def export_result(self, result: dict[str, Any], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return path
