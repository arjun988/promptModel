#!/usr/bin/env python
"""Export a minimal Hugging Face Space folder for PromptForge Phase 3."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP_PY = '''\
"""Hugging Face Space entrypoint for PromptForge."""
import os

from app_impl import build_demo

demo = build_demo(
    quality_model_path=os.environ.get(
        "PROMPTFORGE_QUALITY_MODEL",
        "ArjunShukla/PromptForge-Quality",
    ),
    optimizer_model_path=os.environ.get(
        "PROMPTFORGE_OPTIMIZER_MODEL",
        "ArjunShukla/PromptForge-Optimizer",
    ),
    prefer_gpu=True,
)

if __name__ == "__main__":
    demo.launch()
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(ROOT / "outputs" / "hf_space"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ROOT / "demo" / "app.py", out / "app_impl.py")
    (out / "app.py").write_text(APP_PY, encoding="utf-8")
    shutil.copy2(ROOT / "demo" / "README.md", out / "README.md")

    req = "\n".join(
        [
            "torch",
            "transformers>=4.48.0",
            "peft>=0.11.0",
            "accelerate",
            "gradio>=4.0.0",
            "pandas",
            "numpy",
            "scipy",
            "scikit-learn",
            "pyyaml",
            "huggingface_hub",
            # Install your published package, or copy src/ into the Space.
            # "promptforge @ git+https://github.com/arjun988/promptModel.git",
        ]
    )
    (out / "requirements.txt").write_text(req + "\n", encoding="utf-8")
    print(f"HF Space scaffold → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
