from __future__ import annotations

from typing import Any


def diff_dimensions(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(before) | set(after))
    return {
        key: round(float(after.get(key, 0.0) - before.get(key, 0.0)), 2)
        for key in keys
    }


def infer_changes(
    original: str,
    optimized: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> list[str]:
    """Heuristic change summary for before/after UI."""
    changes: list[str] = []
    original_l = original.lower()
    optimized_l = optimized.lower()

    markers = [
        ("audience", ["for ", "targeting", "audience", "developers", "users"]),
        ("objective", ["build", "create", "write", "implement", "produce"]),
        ("constraints", ["requirements", "constraints", "must", "use "]),
        ("output format", ["return", "output", "json", "markdown", "numbered"]),
        ("technical requirements", ["python", "react", "fastapi", "typescript", "postgresql"]),
        ("structure", ["1.", "2.", "-", "requirements:"]),
        ("context", ["this is for", "university", "production", "startup", "prototype"]),
    ]

    for label, needles in markers:
        if any(n in optimized_l for n in needles) and not any(n in original_l for n in needles):
            changes.append(f"Added {label}")

    if len(optimized.split()) > len(original.split()) * 1.8:
        changes.append("Expanded specificity")

    if before and after:
        b_dims = before.get("dimensions", {})
        a_dims = after.get("dimensions", {})
        for key, delta in diff_dimensions(b_dims, a_dims).items():
            if delta >= 15:
                changes.append(f"Improved {key} (+{delta})")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in changes:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique or ["Rewrote prompt for clarity and completeness"]


def build_comparison_payload(
    original_prompt: str,
    optimized_prompt: str,
    before_analysis: dict[str, Any],
    after_analysis: dict[str, Any] | None = None,
    task_type: str = "general",
) -> dict[str, Any]:
    after_analysis = after_analysis or {}
    before_score = float(before_analysis.get("quality_score", 0.0))
    after_score = float(after_analysis.get("quality_score", before_score))
    before_dims = before_analysis.get("dimensions", {})
    after_dims = after_analysis.get("dimensions", {})

    return {
        "original_prompt": original_prompt,
        "optimized_prompt": optimized_prompt,
        "task_type": task_type,
        "before": {
            "quality_score": before_score,
            "dimensions": before_dims,
            "issues": before_analysis.get("issues", []),
            "missing_information": before_analysis.get("missing_information", []),
        },
        "after": {
            "quality_score": after_score,
            "dimensions": after_dims,
            "issues": after_analysis.get("issues", []),
            "missing_information": after_analysis.get("missing_information", []),
        },
        "delta": {
            "quality_score": round(after_score - before_score, 2),
            "dimensions": diff_dimensions(before_dims, after_dims),
        },
        "changes": infer_changes(
            original_prompt,
            optimized_prompt,
            before_analysis,
            after_analysis,
        ),
    }
