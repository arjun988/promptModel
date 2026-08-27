#!/usr/bin/env python
"""Phase-3 combined pipeline evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-model",
        default=str(ROOT / "outputs" / "promptforge-quality-model"),
    )
    parser.add_argument(
        "--optimizer-model",
        default=str(ROOT / "outputs" / "promptforge-optimizer-model"),
    )
    parser.add_argument("--task-type", default="general")
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "phase3_eval" / "pipeline_report.json"),
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    from promptforge import PromptForge
    from promptforge.evaluation import (
        run_downstream_proxy_eval,
        run_pipeline_evaluation,
        save_eval_report,
    )

    pf = PromptForge(
        quality_model_path=args.quality_model,
        optimizer_model_path=args.optimizer_model,
        prefer_gpu=not args.cpu,
    )
    report = run_pipeline_evaluation(pf, task_type=args.task_type)
    downstream = run_downstream_proxy_eval(pf, task_type=args.task_type)
    payload = {"pipeline": report, "downstream_proxy": downstream}
    save_eval_report(payload, args.output)
    print(json.dumps(payload["pipeline"]["summary"], indent=2))
    print(json.dumps(payload["downstream_proxy"]["summary"], indent=2))
    print("Wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
