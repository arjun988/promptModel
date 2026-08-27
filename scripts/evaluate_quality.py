#!/usr/bin/env python
"""Evaluate a saved PromptForge-Quality model on a CSV or built-in test prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_PROMPTS = [
    "Make an app.",
    "Build me a website.",
    "Write something about AI.",
    "Make a Python API for beginners.",
    (
        "Build a production-ready REST API using FastAPI and PostgreSQL. "
        "Implement JWT authentication, request validation, structured "
        "error handling and OpenAPI documentation. The API will be used "
        "by a React frontend. Return the complete project structure, "
        "implementation and example requests."
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "outputs" / "promptforge-quality-model"),
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from promptforge.scorer import PromptQualityScorer

    scorer = PromptQualityScorer(
        model_path=args.model,
        prefer_gpu=not args.cpu,
    )

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    results = []
    for prompt in prompts:
        result = scorer.analyze(prompt)
        results.append(result)
        print("=" * 80)
        print("PROMPT:", prompt.strip()[:200])
        print("SCORE:", result["quality_score"])
        print(json.dumps(result["dimensions"], indent=2))

    out = Path(args.model) / "smoke_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
