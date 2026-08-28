from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from promptforge.optimizer_chat import apply_chat_prompt, generation_stop_ids
from promptforge.optimizer_validation import (
    clean_optimized_output,
    fallback_optimize,
    validate_optimization,
)
from promptforge.training.train_optimizer import load_optimizer_model
from promptforge.utils.device import describe_device, get_device

_DEFAULT_MAX_NEW_TOKENS = 256
_STOP_STRINGS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


class PromptOptimizer:
    """Inference wrapper for PromptForge-Optimizer (Phase 2 LoRA)."""

    def __init__(
        self,
        model_path: str | Path,
        prefer_gpu: bool = True,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = get_device(prefer_gpu=prefer_gpu)
        self.model, self.tokenizer, self.meta = load_optimizer_model(
            self.model_path,
            device=self.device,
            prefer_gpu=prefer_gpu,
        )
        self.max_new_tokens = int(
            max_new_tokens or self.meta.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS)
        )
        self._stop_ids = generation_stop_ids(self.tokenizer)
        print(f"Loaded PromptForge-Optimizer on {describe_device(self.device)}")

    @torch.inference_mode()
    def optimize(
        self,
        prompt: str,
        analysis: dict[str, Any] | None = None,
        task_type: str = "general",
        max_new_tokens: int | None = None,
        validate: bool = True,
        fallback_on_invalid: bool = True,
    ) -> dict[str, Any]:
        analysis = analysis or {
            "quality_score": "unknown",
            "dimensions": {},
            "issues": [],
            "missing_information": [],
        }

        prompt_text = apply_chat_prompt(
            self.tokenizer, prompt, analysis, task_type=task_type
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._stop_ids,
            tokenizer=self.tokenizer,
            stop_strings=list(_STOP_STRINGS),
        )

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        else:
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated = output_ids[0][prompt_len:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        optimized = clean_optimized_output(raw)

        model_validation: dict[str, Any] | None = None
        validation: dict[str, Any] | None = None
        used_fallback = False
        if validate:
            model_validation = validate_optimization(prompt, optimized)
            if not model_validation["valid"] and fallback_on_invalid:
                optimized = fallback_optimize(prompt, task_type=task_type)
                validation = validate_optimization(prompt, optimized)
                used_fallback = True
            else:
                validation = model_validation
                optimized = validation["optimized_prompt"]

        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized,
            "raw_output": raw,
            "model_output": clean_optimized_output(raw) if used_fallback else optimized,
            "task_type": task_type,
            "analysis": analysis,
            "model_validation": model_validation,
            "validation": validation,
            "used_fallback": used_fallback,
        }

    def to_json(self, prompt: str, analysis: dict[str, Any] | None = None, **kwargs: Any) -> str:
        return json.dumps(self.optimize(prompt, analysis=analysis, **kwargs), indent=2)
