"""Phase-3 end-to-end example."""

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
    quality = ROOT / "outputs" / "promptforge-quality-model"
    optimizer = ROOT / "outputs" / "promptforge-optimizer-model"
    if not quality.exists() or not optimizer.exists():
        print("Train Phase 1 + Phase 2 models first.")
        sys.exit(1)

    pf = PromptForge(
        quality_model_path=quality,
        optimizer_model_path=optimizer,
        prefer_gpu=True,
    )
    result = pf.run("Build me a website for a startup.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
