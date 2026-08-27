from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from promptforge.data.optimizer_generate import SYSTEM_PROMPT, format_optimizer_input
from promptforge.training.train_optimizer import load_optimizer_model
from promptforge.utils.device import describe_device, get_device


class PromptOptimizer:
    """Inference wrapper for PromptForge-Optimizer (Phase 2 LoRA)."""

    def __init__(
        self,
        model_path: str | Path,
        prefer_gpu: bool = True,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = get_device(prefer_gpu=prefer_gpu)
        self.model, self.tokenizer, self.meta = load_optimizer_model(
            self.model_path,
            device=self.device,
            prefer_gpu=prefer_gpu,
        )
        self.max_new_tokens = int(
            max_new_tokens or self.meta.get("max_new_tokens", 512)
        )
        print(f"Loaded PromptForge-Optimizer on {describe_device(self.device)}")

    def _build_prompt(
        self,
        prompt: str,
        analysis: dict[str, Any] | None = None,
        task_type: str = "general",
    ) -> str:
        analysis = analysis or {
            "quality_score": "unknown",
            "dimensions": {},
            "issues": [],
            "missing_information": [],
        }
        user = format_optimizer_input(prompt, analysis, task_type=task_type)
        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{user}\n"
            f"<|assistant|>\n"
        )

    @torch.inference_mode()
    def optimize(
        self,
        prompt: str,
        analysis: dict[str, Any] | None = None,
        task_type: str = "general",
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        text = self._build_prompt(prompt, analysis=analysis, task_type=task_type)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        else:
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        optimized = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized,
            "task_type": task_type,
            "analysis": analysis,
        }

    def to_json(self, prompt: str, analysis: dict[str, Any] | None = None, **kwargs: Any) -> str:
        return json.dumps(self.optimize(prompt, analysis=analysis, **kwargs), indent=2)
