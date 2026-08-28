from __future__ import annotations

import re
from typing import Any

from promptforge.evaluation import instruction_preservation

# Patterns that indicate broken / runaway generation
_STOP_PATTERNS = [
    r"<\|im_start\|>",
    r"<\|assistant\|>",
    r"<\|user\|>",
    r"<\|system\|>",
    r"<\|endoftext\|>",
    r"^assistant\s*:",
]

_MAX_OPTIMIZED_WORDS = 220
_MIN_OPTIMIZED_WORDS = 8


def clean_optimized_output(text: str) -> str:
    """Strip chat markers and truncate at the first runaway segment."""
    if not text:
        return ""

    cleaned = text.strip()

    for pattern in _STOP_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[: match.start()].strip()

    # Drop repeated trailing "Return ..." loops
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.lower()
        if key in seen and line.lower().startswith("return"):
            break
        seen.add(key)
        deduped.append(line)

    cleaned = "\n".join(deduped).strip()
    words = cleaned.split()
    if len(words) > _MAX_OPTIMIZED_WORDS:
        cleaned = " ".join(words[:_MAX_OPTIMIZED_WORDS]).strip()

    return cleaned


def detect_repetition(text: str, min_repeats: int = 3) -> bool:
    """True if the same line repeats too many times."""
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    if len(lines) < min_repeats:
        return False
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
        if counts[line] >= min_repeats:
            return True
    return False


def validate_optimization(
    original: str,
    optimized: str,
    *,
    min_instruction_preservation: float = 0.08,
    min_words: int = _MIN_OPTIMIZED_WORDS,
    max_words: int = _MAX_OPTIMIZED_WORDS,
) -> dict[str, Any]:
    optimized_clean = clean_optimized_output(optimized)
    word_count = len(optimized_clean.split())
    intent = instruction_preservation(original, optimized_clean)
    repetitive = detect_repetition(optimized) or detect_repetition(optimized_clean)

    issues: list[str] = []
    if not optimized_clean:
        issues.append("empty_output")
    if word_count < min_words:
        issues.append("too_short")
    if word_count > max_words:
        issues.append("too_long")
    if intent < min_instruction_preservation:
        issues.append("low_intent_preservation")
    if repetitive:
        issues.append("repetitive")

    valid = len(issues) == 0
    return {
        "valid": valid,
        "issues": issues,
        "instruction_preservation": round(intent, 4),
        "word_count": word_count,
        "repetitive": repetitive,
        "optimized_prompt": optimized_clean,
    }


def fallback_optimize(original: str, task_type: str = "general") -> str:
    """Intent-preserving rewrite when the model output is invalid."""
    from promptforge.data.optimizer_generate import build_fallback_prompt

    return build_fallback_prompt(original, task_type=task_type)
