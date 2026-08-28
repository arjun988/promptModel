from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import pandas as pd

from promptforge.data.generate import (
    AUDIENCES,
    CONSTRAINTS,
    CONTEXTS,
    OUTPUT_FORMATS,
    TASKS,
    clamp,
)
from promptforge.optimizer_chat import (
    SYSTEM_PROMPT,
    build_optimizer_messages,
    format_optimizer_input,
    tokenize_sft_messages,
)

# Re-export for backwards compatibility
__all__ = [
    "SYSTEM_PROMPT",
    "build_optimizer_messages",
    "format_optimizer_input",
    "generate_optimizer_dataset",
    "generate_optimizer_example",
    "row_to_analysis",
]

# Weak prompt → (domain, canonical task) for intent preservation
CANONICAL_WEAK: dict[str, tuple[str, str]] = {
    "Make an app.": ("coding", "Build a mobile application"),
    "Build something.": ("coding", "Build a software project"),
    "Create a website.": ("coding", "Build a responsive website"),
    "Build me a website.": ("coding", "Build a responsive website"),
    "Build me a project.": ("coding", "Build a software project"),
    "Make a Python API.": ("coding", "Build a Python REST API"),
    "Make a Python API for beginners.": ("coding", "Build a Python REST API"),
    "Write something about AI.": ("writing", "Write an article about AI"),
    "Write something good.": ("writing", "Write a high-quality article"),
    "Analyze this.": ("data", "Analyze a dataset"),
    "Create a workout plan.": ("general", "Create a workout plan"),
    "Help me with this.": ("general", "Help me solve this problem"),
    "Make this better.": ("general", "Improve this draft"),
}


def _scores_for_level(level: int, r: random.Random) -> dict[str, int]:
    clarity = clamp(30 + level * 17 + r.randint(-5, 5))
    specificity = clamp(15 + level * 20 + r.randint(-5, 5))
    context = clamp(10 + level * 20 + r.randint(-5, 5))
    goal_definition = clamp(25 + level * 17 + r.randint(-5, 5))
    constraints = clamp(5 + level * 22 + r.randint(-5, 5))
    completeness = clamp(15 + level * 20 + r.randint(-5, 5))
    actionability = clamp(20 + level * 18 + r.randint(-5, 5))
    dims = {
        "clarity": clarity,
        "specificity": specificity,
        "context": context,
        "goal_definition": goal_definition,
        "constraints": constraints,
        "completeness": completeness,
        "actionability": actionability,
    }
    dims["quality_score"] = clamp(float(np.mean(list(dims.values()))))
    return dims


def _missing_from_scores(scores: dict[str, int]) -> list[str]:
    missing: list[str] = []
    if scores["context"] < 45:
        missing.append("context")
    if scores["goal_definition"] < 45:
        missing.append("goal")
    if scores["constraints"] < 45:
        missing.append("constraints")
    if scores["specificity"] < 45:
        missing.append("specific_requirements")
    if scores["completeness"] < 45:
        missing.append("output_format")
    if scores["actionability"] < 45:
        missing.append("actionable_steps")
    return missing


def _issues_from_scores(scores: dict[str, int]) -> list[str]:
    issues: list[str] = []
    if scores["specificity"] < 40:
        issues.append("too_vague")
    if scores["context"] < 40:
        issues.append("missing_context")
    if scores["goal_definition"] < 45:
        issues.append("ambiguous_objective")
    if scores["constraints"] < 40:
        issues.append("insufficient_constraints")
    if scores["completeness"] < 40:
        issues.append("incomplete_prompt")
    return issues


def _weak_prompt(task: str, level: int, r: random.Random) -> str:
    if level <= 0:
        # Try to match a canonical weak prompt for this task
        for weak, (_, canon) in CANONICAL_WEAK.items():
            if canon.lower() in task.lower() or task.lower() in canon.lower():
                return weak
        return f"{task.split()[0]} something related to {task.lower()}."
    if level == 1:
        return f"{task}."
    if level == 2:
        return f"{task} for {r.choice(AUDIENCES)}."
    if level == 3:
        return (
            f"{task} for {r.choice(AUDIENCES)}. "
            f"{r.choice(CONTEXTS)} "
            f"{r.choice(OUTPUT_FORMATS)}"
        )
    return (
        f"{task} for {r.choice(AUDIENCES)}. "
        f"{r.choice(CONTEXTS)} Use reasonable defaults."
    )


def _build_optimized(
    domain: str,
    task: str,
    r: random.Random,
) -> str:
    """Intent-preserving optimized prompt — same task/topic, more detail."""
    audience = r.choice(AUDIENCES)
    context = r.choice(CONTEXTS)
    c1 = r.choice(CONSTRAINTS)
    c2 = r.choice(CONSTRAINTS)
    output = r.choice(OUTPUT_FORMATS)

    if domain == "coding":
        return (
            f"{task} for {audience}.\n"
            f"{context}\n\n"
            f"Requirements:\n"
            f"- {c1}\n"
            f"- {c2}\n"
            f"- Include error handling and clear structure\n\n"
            f"Return:\n"
            f"1. Project structure\n"
            f"2. Complete implementation\n"
            f"3. Setup instructions\n\n"
            f"{output}"
        )
    if domain == "writing":
        return (
            f"{task} for {audience}.\n"
            f"{context}\n\n"
            f"Requirements:\n"
            f"- Clear structure with headings\n"
            f"- Concrete examples\n"
            f"- {output}\n\n"
            f"Tone: professional and concise."
        )
    if domain == "research":
        return (
            f"{task}\n"
            f"Audience: {audience}.\n"
            f"{context}\n\n"
            f"Cover:\n"
            f"- Problem statement\n"
            f"- Key concepts\n"
            f"- Trade-offs and recommendation\n\n"
            f"{output}"
        )
    if domain == "data":
        return (
            f"{task} for {audience}.\n"
            f"{context}\n\n"
            f"Requirements:\n"
            f"- State assumptions\n"
            f"- {c1}\n"
            f"- {output}\n\n"
            f"Include interpretation of results."
        )
    return (
        f"{task} for {audience}.\n"
        f"{context}\n\n"
        f"Requirements:\n"
        f"- {c1}\n"
        f"- {output}\n\n"
        f"Be specific and actionable."
    )


def build_fallback_prompt(prompt: str, task_type: str = "general") -> str:
    """Intent-preserving fallback when model output fails validation."""
    r = random.Random(hash(prompt.strip().lower()) & 0xFFFFFFFF)
    normalized = prompt.strip()

    for weak, (domain, task) in CANONICAL_WEAK.items():
        if weak.lower() == normalized.lower():
            return _build_optimized(domain, task, r)

    if task_type in TASKS:
        return _build_optimized(task_type, normalized.rstrip("."), r)

    topic = normalized.rstrip(".")
    return _build_optimized("general", topic, r)


def generate_optimizer_example(rng: random.Random | None = None) -> dict[str, Any]:
    r = rng or random

    # 30% canonical real-world weak prompts
    if r.random() < 0.30 and CANONICAL_WEAK:
        weak = r.choice(list(CANONICAL_WEAK.keys()))
        domain, task = CANONICAL_WEAK[weak]
        level = r.choices([0, 1, 2], weights=[0.5, 0.35, 0.15], k=1)[0]
    else:
        domain = r.choice(list(TASKS.keys()))
        task = r.choice(TASKS[domain])
        level = r.choices([0, 1, 2, 3], weights=[0.30, 0.30, 0.25, 0.15], k=1)[0]
        weak = _weak_prompt(task, level, r)

    scores = _scores_for_level(level, r)
    missing = _missing_from_scores(scores)
    issues = _issues_from_scores(scores)
    optimized = _build_optimized(domain, task, r)

    analysis = {
        "quality_score": scores["quality_score"],
        "dimensions": {k: v for k, v in scores.items() if k != "quality_score"},
        "issues": issues,
        "missing_information": missing,
    }

    messages = build_optimizer_messages(
        weak, analysis, optimized_prompt=optimized, task_type=domain
    )

    return {
        "prompt": weak,
        "task_type": domain,
        "canonical_task": task,
        "quality_level": level,
        "quality_score": scores["quality_score"],
        "clarity": scores["clarity"],
        "specificity": scores["specificity"],
        "context": scores["context"],
        "goal_definition": scores["goal_definition"],
        "constraints": scores["constraints"],
        "completeness": scores["completeness"],
        "actionability": scores["actionability"],
        "issues": json.dumps(issues),
        "missing_information": json.dumps(missing),
        "analysis_json": json.dumps(analysis),
        "optimized_prompt": optimized,
        "messages_json": json.dumps(messages),
    }


def generate_optimizer_dataset(
    num_examples: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = [generate_optimizer_example(rng) for _ in range(num_examples)]
    return pd.DataFrame(rows)


def row_to_analysis(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    if row.get("analysis_json"):
        return json.loads(row["analysis_json"])
    return {
        "quality_score": row.get("quality_score"),
        "dimensions": {
            k: row[k]
            for k in (
                "clarity",
                "specificity",
                "context",
                "goal_definition",
                "constraints",
                "completeness",
                "actionability",
            )
            if k in row
        },
        "issues": json.loads(row["issues"]) if isinstance(row.get("issues"), str) else row.get("issues", []),
        "missing_information": (
            json.loads(row["missing_information"])
            if isinstance(row.get("missing_information"), str)
            else row.get("missing_information", [])
        ),
    }
