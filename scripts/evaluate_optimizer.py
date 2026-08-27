#!/usr/bin/env python
"""Smoke-test PromptForge-Optimizer generations."""

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
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "outputs" / "promptforge-optimizer-model"),
    )
    parser.add_argument("--quality-model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from promptforge import PromptForge

    pf = PromptForge(
        quality_model_path=args.quality_model,
        optimizer_model_path=args.model,
        prefer_gpu=not args.cpu,
    )

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    results = []
    for prompt in prompts:
        analysis = pf.analyze(prompt) if args.quality_model else None
        result = pf.optimize(prompt, analysis=analysis, use_scorer_analysis=False)
        results.append(result)
        print("=" * 80)
        print("ORIGINAL:", prompt)
        print("\nOPTIMIZED:\n", result["optimized_prompt"])

    out = Path(args.model) / "smoke_optimize.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
