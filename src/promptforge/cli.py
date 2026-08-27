from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_analyze(args: argparse.Namespace) -> int:
    from promptforge.scorer import PromptQualityScorer

    scorer = PromptQualityScorer(
        model_path=args.model,
        prefer_gpu=not args.cpu,
        max_length=args.max_length,
    )
    result = scorer.analyze(args.prompt)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"Prompt Quality: {result['quality_score']}/100\n")
    print("Dimensions:")
    for key, value in result["dimensions"].items():
        print(f"  {key:20s}: {value}")
    if result["issues"]:
        print("\nProblems:")
        for issue in result["issues"]:
            print(f"  ✗ {issue}")
    if result["missing_information"]:
        print("\nMissing information:")
        for item in result["missing_information"]:
            print(f"  - {item}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from promptforge.config import QualityScorerConfig
    from promptforge.training.train_quality import train_quality_scorer

    if args.config:
        config = QualityScorerConfig.from_yaml(args.config)
    else:
        config = QualityScorerConfig()

    if args.num_examples is not None:
        config.num_examples = args.num_examples
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.final_model_dir:
        config.final_model_dir = args.final_model_dir
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.cpu:
        config.prefer_gpu = False
        config.use_fp16 = False
        config.use_bf16 = False

    train_quality_scorer(config, regenerate_dataset=args.regenerate)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptforge",
        description="PromptForge — score and (soon) optimize LLM prompts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Score a prompt with PromptForge-Quality")
    analyze.add_argument("prompt", type=str, help="Prompt text to analyze")
    analyze.add_argument(
        "--model",
        type=str,
        default="outputs/promptforge-quality-model",
        help="Local model dir or HF repo id",
    )
    analyze.add_argument("--max-length", type=int, default=512)
    analyze.add_argument("--json", action="store_true", help="Print raw JSON")
    analyze.add_argument("--cpu", action="store_true", help="Force CPU")
    analyze.set_defaults(func=_cmd_analyze)

    train = sub.add_parser("train-quality", help="Train Phase-1 quality scorer (GPU-first)")
    train.add_argument("--config", type=str, default="configs/quality_scorer.yaml")
    train.add_argument("--num-examples", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--output-dir", type=str, default=None)
    train.add_argument("--final-model-dir", type=str, default=None)
    train.add_argument("--dataset-path", type=str, default=None)
    train.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate synthetic dataset even if CSV exists",
    )
    train.add_argument("--cpu", action="store_true", help="Allow CPU training (slow)")
    train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
