#!/usr/bin/env python
"""Push a local PromptForge-Quality bundle to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


README_CARD = """---
language:
  - en
license: mit
library_name: transformers
base_model: answerdotai/ModernBERT-base
tags:
  - promptforge
  - prompt-engineering
  - prompt-quality
  - modernbert
  - regression
  - text-classification
  - llm
pipeline_tag: text-classification
---

# PromptForge-Quality

Multi-dimension prompt quality scorer (ModernBERT). See the full model card in this repo's `README.md` after upload, or the checkpoint folder `outputs/promptforge-quality-model/README.md`.

Scores: clarity, specificity, context, goal_definition, constraints, completeness, actionability.

```python
from promptforge import PromptForge
pf = PromptForge(quality_model_path="ArjunShukla/PromptForge-Quality")
print(pf.analyze("Build me a website"))
```
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(ROOT / "outputs" / "promptforge-quality-model"),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="e.g. your-username/PromptForge-Quality",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    from huggingface_hub import HfApi, login

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    readme = model_dir / "README.md"
    if not readme.exists():
        readme.write_text(README_CARD, encoding="utf-8")

    if args.token:
        login(token=args.token)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        private=args.private,
        exist_ok=True,
        repo_type="model",
    )
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded → https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
