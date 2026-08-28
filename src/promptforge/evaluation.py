from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:
    from promptforge.pipeline import PromptForge


DEFAULT_EVAL_PROMPTS = [
    "Make an app.",
    "Build me a website.",
    "Write something about AI.",
    "Make a Python API for beginners.",
    "Create a workout plan.",
    "Analyze this dataset.",
    "Help me with my startup idea.",
    "Build a chat bot.",
]


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2}


def _token_preserved(token: str, other_tokens: set[str], other_text: str) -> bool:
  if token in other_tokens:
    return True
  return token in other_text


def instruction_preservation(original: str, optimized: str) -> float:
    """Intent overlap — recall-heavy for short prompts that get rewritten."""
    a = _token_set(original)
    if not a:
        return 1.0
    b = _token_set(optimized)
    opt_lower = optimized.lower()

    recall = sum(1 for t in a if _token_preserved(t, b, opt_lower)) / len(a)
    jaccard = float(len(a & b) / len(a | b)) if (a | b) else 0.0

    # Vague short prompts ("Make an app.") are heavily expanded — recall matters more
    if len(a) <= 4:
        return max(jaccard, recall)
    return jaccard


def information_preservation(original: str, optimized: str) -> float:
    """Fraction of original tokens retained in optimized prompt."""
    a = _token_set(original)
    if not a:
        return 1.0
    b = _token_set(optimized)
    return float(len(a & b) / len(a))


def run_pipeline_evaluation(
    pf: "PromptForge",
    prompts: list[str] | None = None,
    task_type: str = "general",
    rescore_optimized: bool = True,
) -> dict[str, Any]:
    """
    Phase-3 evaluation:
    - score original
    - optimize (with validation + fallback)
    - re-score optimized with quality model
    - compute improvement / preservation / validity metrics
    """
    if pf.scorer is None or pf.optimizer is None:
        raise RuntimeError(
            "Phase-3 evaluation requires both quality_model_path and optimizer_model_path."
        )

    prompts = prompts or DEFAULT_EVAL_PROMPTS
    rows: list[dict[str, Any]] = []

    for prompt in prompts:
        comparison = pf.run(
            prompt,
            task_type=task_type,
            rescore_optimized=rescore_optimized,
        )
        before = comparison["before"]["quality_score"]
        after = comparison["after"]["quality_score"]
        validation = comparison.get("validation") or {}
        intent = validation.get(
            "instruction_preservation",
            instruction_preservation(prompt, comparison["optimized_prompt"]),
        )
        rows.append(
            {
                "prompt": prompt,
                "optimized_prompt": comparison["optimized_prompt"],
                "before_score": before,
                "after_score": after,
                "score_delta": comparison["delta"]["quality_score"],
                "improved": after > before,
                "valid": validation.get("valid", True),
                "repetitive": validation.get("repetitive", False),
                "used_fallback": comparison.get("used_fallback", False),
                "trustworthy_improvement": comparison.get("trustworthy_improvement", False),
                "instruction_preservation": round(intent, 4),
                "information_preservation": round(
                    information_preservation(prompt, comparison["optimized_prompt"]), 4
                ),
                "validation_issues": validation.get("issues", []),
                "changes": comparison["changes"],
            }
        )

    df = pd.DataFrame(rows)
    valid_mask = df["valid"] & ~df["repetitive"]
    trustworthy = df["trustworthy_improvement"]
    summary = {
        "n": int(len(df)),
        "mean_before_score": float(df["before_score"].mean()),
        "mean_after_score": float(df["after_score"].mean()),
        "mean_score_delta": float(df["score_delta"].mean()),
        "pct_improved": float(df["improved"].mean() * 100.0),
        "pct_valid": float(df["valid"].mean() * 100.0),
        "pct_repetitive": float(df["repetitive"].mean() * 100.0),
        "pct_used_fallback": float(df["used_fallback"].mean() * 100.0),
        "pct_trustworthy_improvement": float(trustworthy.mean() * 100.0),
        "mean_instruction_preservation": float(df["instruction_preservation"].mean()),
        "mean_information_preservation": float(df["information_preservation"].mean()),
        "mean_score_delta_valid_only": float(df.loc[valid_mask, "score_delta"].mean())
        if valid_mask.any()
        else 0.0,
    }
    return {"summary": summary, "rows": rows}


def heuristic_downstream_proxy(
    prompt: str,
    quality_score: float,
) -> float:
    """
    Cheap stand-in for downstream LLM success until a real judge is wired.

    Combines quality score with simple specificity signals.
    """
    length_bonus = min(len(prompt.split()) / 80.0, 1.0) * 10.0
    structure_bonus = 5.0 if ("requirement" in prompt.lower() or "return" in prompt.lower()) else 0.0
    return float(min(100.0, quality_score * 0.85 + length_bonus + structure_bonus))


def run_downstream_proxy_eval(
    pf: "PromptForge",
    prompts: list[str] | None = None,
    task_type: str = "general",
) -> dict[str, Any]:
    """Compare proxy downstream success: original vs optimized (valid outputs only)."""
    from promptforge.optimizer_validation import validate_optimization

    prompts = prompts or DEFAULT_EVAL_PROMPTS
    rows = []
    for prompt in prompts:
        result = pf.run(prompt, task_type=task_type, rescore_optimized=True)
        validation = result.get("validation") or validate_optimization(
            prompt, result["optimized_prompt"]
        )
        before_q = result["before"]["quality_score"]
        after_q = result["after"]["quality_score"]
        before_down = heuristic_downstream_proxy(prompt, before_q)
        after_down = heuristic_downstream_proxy(result["optimized_prompt"], after_q)
        rows.append(
            {
                "prompt": prompt,
                "valid": validation.get("valid", True),
                "trustworthy_improvement": result.get("trustworthy_improvement", False),
                "before_downstream": round(before_down, 2),
                "after_downstream": round(after_down, 2),
                "downstream_delta": round(after_down - before_down, 2),
                "before_quality": before_q,
                "after_quality": after_q,
            }
        )

    df = pd.DataFrame(rows)
    valid_df = df[df["valid"]]
    mean_before = float(valid_df["before_downstream"].mean()) if len(valid_df) else 0.0
    mean_after = float(valid_df["after_downstream"].mean()) if len(valid_df) else 0.0
    lift = ((mean_after - mean_before) / mean_before * 100.0) if mean_before else 0.0
    return {
        "summary": {
            "mean_before_downstream": mean_before,
            "mean_after_downstream": mean_after,
            "relative_lift_pct": float(lift),
            "n": int(len(df)),
            "n_valid": int(len(valid_df)),
        },
        "rows": rows,
    }


def save_eval_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


JudgeFn = Callable[[str, str], dict[str, Any]]


def llm_as_judge_stub(original: str, optimized: str) -> dict[str, Any]:
    """
    Placeholder for a real LLM-as-judge.

    Replace with an API call (OpenAI/Anthropic/local) that returns preference.
    """
    from promptforge.optimizer_validation import validate_optimization

    validation = validate_optimization(original, optimized)
    preferred = "optimized" if validation["valid"] else "original"
    return {
        "preferred": preferred,
        "confidence": 0.6 if validation["valid"] else 0.3,
        "rationale": "stub — prefers valid, intent-preserving optimizations",
        "validation": validation,
    }
