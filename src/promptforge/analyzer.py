from __future__ import annotations

from typing import Any

from promptforge.scorer import PromptQualityScorer


class PromptAnalyzer:
    """
    Thin analysis helper (PRD `analyzer.py`).

    Wraps the Phase-1 scorer and normalizes the structured contract used by
    the optimizer and Phase-3 pipeline.
    """

    def __init__(self, scorer: PromptQualityScorer) -> None:
        self.scorer = scorer

    def analyze(self, prompt: str) -> dict[str, Any]:
        result = self.scorer.analyze(prompt)
        return {
            "prompt": prompt,
            "quality_score": result["quality_score"],
            "dimensions": result["dimensions"],
            "issues": result.get("issues", []),
            "missing_information": result.get("missing_information", []),
        }
