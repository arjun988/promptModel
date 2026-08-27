"""Minimal local inference example (Phase 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from promptforge import PromptForge


def main() -> None:
    model_path = ROOT / "outputs" / "promptforge-quality-model"
    if not model_path.exists():
        print(
            f"Model not found at {model_path}. "
            "Train first: python scripts/train_quality.py --require-gpu"
        )
        sys.exit(1)

    pf = PromptForge(quality_model_path=model_path, prefer_gpu=True)
    result = pf.analyze("Build me a website for a startup.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
