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
    LEVEL0_PROMPTS,
    OUTPUT_FORMATS,
    TASKS,
    clamp,
)


SYSTEM_PROMPT = (
    "You are PromptForge Optimizer. Rewrite the user's prompt into a clear, "
    "specific, complete, and actionable LLM prompt. Preserve the original intent. "
    "Add missing audience, context, constraints, and output format when needed. "
    "Return ONLY the optimized prompt text."
)


OPTIMIZED_TEMPLATES: dict[str, list[str]] = {
    "coding": [
        (
            "Build a production-ready {task_detail} for {audience}.\n"
            "{context}\n\n"
            "Requirements:\n"
            "- {c1}\n"
            "- {c2}\n"
            "- Include error handling and basic tests\n"
            "- Prefer clear project structure\n\n"
            "Return:\n"
            "1. Project structure\n"
            "2. Complete implementation\n"
            "3. Setup / run instructions\n"
            "4. Example usage"
        ),
        (
            "Create a well-specified {task_detail} targeting {audience}.\n"
            "{context}\n\n"
            "Constraints:\n"
            "- {c1}\n"
            "- {c2}\n"
            "- {output}\n\n"
            "Explain key design decisions briefly, then provide complete working code."
        ),
    ],
    "writing": [
        (
            "Write a high-quality {task_detail} for {audience}.\n"
            "{context}\n\n"
            "Requirements:\n"
            "- Clear structure with headings\n"
            "- Concrete examples\n"
            "- Practical takeaways\n"
            "- {output}\n\n"
            "Tone: professional, concise, and useful."
        ),
    ],
    "research": [
        (
            "Produce a structured research brief: {task_detail}.\n"
            "Audience: {audience}.\n"
            "{context}\n\n"
            "Cover:\n"
            "- Problem / question\n"
            "- Key concepts\n"
            "- Comparison criteria\n"
            "- Trade-offs\n"
            "- Recommendation\n\n"
            "{output}"
        ),
    ],
    "data": [
        (
            "Perform a practical data task: {task_detail} for {audience}.\n"
            "{context}\n\n"
            "Requirements:\n"
            "- State assumptions\n"
            "- Show steps clearly\n"
            "- {c1}\n"
            "- {output}\n\n"
            "Include interpretation of results, not only code."
        ),
    ],
    "creative": [
        (
            "Create {task_detail} for {audience}.\n"
            "{context}\n\n"
            "Creative constraints:\n"
            "- Original and specific\n"
            "- {c1}\n"
            "- {output}\n\n"
            "Deliver a polished draft ready for iteration."
        ),
    ],
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


def _weak_prompt(domain: str, task: str, level: int, r: random.Random) -> str:
    if level <= 0:
        return r.choice(LEVEL0_PROMPTS)
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
    # level 4 already strong — still produce a slightly weaker variant for rewrite practice
    return (
        f"{task} for {r.choice(AUDIENCES)}. "
        f"{r.choice(CONTEXTS)} "
        f"Use reasonable defaults."
    )


def _task_detail(task: str) -> str:
    return task[0].lower() + task[1:] if task else task


def _build_optimized(domain: str, task: str, r: random.Random) -> str:
    templates = OPTIMIZED_TEMPLATES.get(domain, OPTIMIZED_TEMPLATES["coding"])
    template = r.choice(templates)
    return template.format(
        task_detail=_task_detail(task),
        audience=r.choice(AUDIENCES),
        context=r.choice(CONTEXTS),
        c1=r.choice(CONSTRAINTS),
        c2=r.choice(CONSTRAINTS),
        output=r.choice(OUTPUT_FORMATS),
    )


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


def format_optimizer_input(
    prompt: str,
    analysis: dict[str, Any],
    task_type: str = "general",
) -> str:
    """User message content fed to the optimizer LM."""
    dims = analysis.get("dimensions", analysis)
    lines = [
        f"Task type: {task_type}",
        "Original prompt:",
        prompt.strip(),
        "",
        "Quality analysis:",
        f"- quality_score: {analysis.get('quality_score', dims.get('quality_score', 'n/a'))}",
    ]
    for key in (
        "clarity",
        "specificity",
        "context",
        "goal_definition",
        "constraints",
        "completeness",
        "actionability",
    ):
        if key in dims:
            lines.append(f"- {key}: {dims[key]}")

    missing = analysis.get("missing_information") or analysis.get("missing") or []
    issues = analysis.get("issues") or []
    if issues:
        lines.append(f"- issues: {', '.join(issues)}")
    if missing:
        lines.append(f"- missing_information: {', '.join(missing)}")

    lines.extend(
        [
            "",
            "Rewrite this into a high-quality optimized prompt.",
            "Return only the optimized prompt.",
        ]
    )
    return "\n".join(lines)


def build_training_text(
    prompt: str,
    analysis: dict[str, Any],
    optimized_prompt: str,
    task_type: str = "general",
) -> str:
    """Plain-text supervised example (instruction → response)."""
    user = format_optimizer_input(prompt, analysis, task_type=task_type)
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n{optimized_prompt.strip()}"
    )


def generate_optimizer_example(rng: random.Random | None = None) -> dict[str, Any]:
    r = rng or random
    domain = r.choice(list(TASKS.keys()))
    task = r.choice(TASKS[domain])
    # Bias toward weak/medium prompts that need optimization
    level = r.choices([0, 1, 2, 3, 4], weights=[0.25, 0.25, 0.25, 0.15, 0.10], k=1)[0]

    weak = _weak_prompt(domain, task, level, r)
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

    return {
        "prompt": weak,
        "task_type": domain,
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
        "training_text": build_training_text(
            weak, analysis, optimized, task_type=domain
        ),
    }


def generate_optimizer_dataset(
    num_examples: int = 10_000,
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
