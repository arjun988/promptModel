from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd


TASKS: dict[str, list[str]] = {
    "coding": [
        "Build a REST API",
        "Create a React application",
        "Write a Python script",
        "Build a CLI tool",
        "Create a database schema",
    ],
    "writing": [
        "Write a blog post",
        "Write an email",
        "Write a technical article",
        "Create a product description",
        "Write a LinkedIn post",
    ],
    "research": [
        "Research a technology",
        "Compare two databases",
        "Analyze a software architecture",
        "Explain a machine learning technique",
        "Evaluate a programming language",
    ],
    "data": [
        "Analyze a dataset",
        "Create a data visualization",
        "Build a machine learning model",
        "Clean a dataset",
        "Generate a statistical report",
    ],
    "creative": [
        "Create a story",
        "Write a game concept",
        "Design a character",
        "Create a marketing campaign",
        "Generate a product idea",
    ],
}

AUDIENCES = [
    "beginner developers",
    "experienced developers",
    "software engineers",
    "students",
    "technical managers",
    "startup founders",
    "data scientists",
    "general users",
]

CONSTRAINTS = [
    "Use Python 3.12.",
    "Use TypeScript and React.",
    "Keep the solution under 200 lines.",
    "Return the answer as Markdown.",
    "Include complete working code.",
    "Do not use external libraries.",
    "Include error handling.",
    "Make the solution production-ready.",
    "Use PostgreSQL.",
    "Make it mobile responsive.",
]

OUTPUT_FORMATS = [
    "Return the answer as a numbered list.",
    "Return complete source code.",
    "Return JSON.",
    "Return a step-by-step explanation.",
    "Return a Markdown document.",
    "Include examples.",
]

CONTEXTS = [
    "This is for a university project.",
    "This is for a production SaaS application.",
    "This is for an internal developer tool.",
    "This will be used by beginners.",
    "This is a prototype for a startup.",
    "This will run locally on a laptop.",
]

LEVEL0_PROMPTS = [
    "Make an app.",
    "Build something.",
    "Write something good.",
    "Help me with this.",
    "Create a website.",
    "Make this better.",
    "Analyze this.",
    "Build me a project.",
]


def clamp(value: float) -> int:
    return max(0, min(100, int(round(value))))


def generate_example(rng: random.Random | None = None) -> dict[str, Any]:
    """Generate one synthetic prompt + multi-dimension quality labels."""
    r = rng or random

    domain = r.choice(list(TASKS.keys()))
    task = r.choice(TASKS[domain])
    level = r.randint(0, 4)

    clarity = clamp(30 + level * 17 + r.randint(-5, 5))
    specificity = clamp(15 + level * 20 + r.randint(-5, 5))
    context = clamp(10 + level * 20 + r.randint(-5, 5))
    goal_definition = clamp(25 + level * 17 + r.randint(-5, 5))
    constraints = clamp(5 + level * 22 + r.randint(-5, 5))
    completeness = clamp(15 + level * 20 + r.randint(-5, 5))
    actionability = clamp(20 + level * 18 + r.randint(-5, 5))

    if level == 0:
        prompt = r.choice(LEVEL0_PROMPTS)
    elif level == 1:
        prompt = f"{task}."
    elif level == 2:
        prompt = f"{task} for {r.choice(AUDIENCES)}."
    elif level == 3:
        prompt = (
            f"{task} for {r.choice(AUDIENCES)}. "
            f"{r.choice(CONTEXTS)} "
            f"{r.choice(OUTPUT_FORMATS)}"
        )
    else:
        prompt = (
            f"{task} for {r.choice(AUDIENCES)}. "
            f"{r.choice(CONTEXTS)} "
            f"Requirements: {r.choice(CONSTRAINTS)} "
            f"{r.choice(CONSTRAINTS)} "
            f"{r.choice(OUTPUT_FORMATS)} "
            f"Explain important design decisions and include examples."
        )

    overall = float(
        np.mean(
            [
                clarity,
                specificity,
                context,
                goal_definition,
                constraints,
                completeness,
                actionability,
            ]
        )
    )

    return {
        "prompt": prompt,
        "clarity": clarity,
        "specificity": specificity,
        "context": context,
        "goal_definition": goal_definition,
        "constraints": constraints,
        "completeness": completeness,
        "actionability": actionability,
        "quality_score": clamp(overall),
        "task_type": domain,
        "quality_level": level,
    }


def generate_dataset(
    num_examples: int = 25_000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    examples = [generate_example(rng) for _ in range(num_examples)]
    return pd.DataFrame(examples)


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    quality_bins = (
        pd.cut(
            df["quality_score"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["0-20", "21-40", "41-60", "61-80", "81-100"],
        )
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {
        "size": len(df),
        "quality_score_describe": df["quality_score"].describe().to_dict(),
        "task_type_counts": df["task_type"].value_counts().to_dict(),
        "quality_bins": {str(k): int(v) for k, v in quality_bins.items()},
    }
