from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from promptforge.models.quality import PromptForgeQualityModel
from promptforge.utils.device import describe_device, get_device


class PromptQualityScorer:
    """Inference wrapper for PromptForge-Quality (Phase 1)."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device | None = None,
        max_length: int = 512,
        prefer_gpu: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.device = (
            torch.device(device)
            if device is not None
            else get_device(prefer_gpu=prefer_gpu)
        )

        self.model = PromptForgeQualityModel.from_pretrained(
            self.model_path,
            map_location="cpu",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.label_names = list(self.model.label_names)

        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded PromptForge-Quality on {describe_device(self.device)}")

    @torch.inference_mode()
    def score(self, prompt: str) -> dict[str, Any]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Prefer CUDA autocast when on GPU.
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        dim_scores = outputs["logits"].squeeze(0).detach().float().cpu().numpy()
        quality = outputs["quality"].squeeze().detach().float().cpu().item()

        dimensions = {
            name: round(float(score), 2)
            for name, score in zip(self.label_names, dim_scores)
        }
        # Prefer dedicated quality head; fall back to mean of dims.
        quality_score = round(float(quality), 2)

        issues = self._infer_issues(dimensions)
        missing = self._infer_missing(dimensions)

        return {
            "quality_score": quality_score,
            "dimensions": dimensions,
            "issues": issues,
            "missing_information": missing,
        }

    def analyze(self, prompt: str) -> dict[str, Any]:
        """Alias used by the public API / CLI."""
        result = self.score(prompt)
        result["prompt"] = prompt
        return result

    @staticmethod
    def _infer_issues(dimensions: dict[str, float]) -> list[str]:
        issues: list[str] = []
        if dimensions.get("clarity", 100) < 45:
            issues.append("unclear_request")
        if dimensions.get("specificity", 100) < 40:
            issues.append("too_vague")
        if dimensions.get("context", 100) < 40:
            issues.append("missing_context")
        if dimensions.get("goal_definition", 100) < 45:
            issues.append("ambiguous_objective")
        if dimensions.get("constraints", 100) < 40:
            issues.append("insufficient_constraints")
        if dimensions.get("completeness", 100) < 40:
            issues.append("incomplete_prompt")
        if dimensions.get("actionability", 100) < 40:
            issues.append("low_actionability")
        return issues

    @staticmethod
    def _infer_missing(dimensions: dict[str, float]) -> list[str]:
        missing: list[str] = []
        if dimensions.get("context", 100) < 45:
            missing.append("context")
        if dimensions.get("goal_definition", 100) < 45:
            missing.append("goal")
        if dimensions.get("constraints", 100) < 45:
            missing.append("constraints")
        if dimensions.get("specificity", 100) < 45:
            missing.append("specific_requirements")
        if dimensions.get("completeness", 100) < 45:
            missing.append("output_format")
        return missing

    def score_many(self, prompts: list[str]) -> list[dict[str, Any]]:
        return [self.score(p) for p in prompts]

    def to_json(self, prompt: str, indent: int = 2) -> str:
        return json.dumps(self.analyze(prompt), indent=indent)
