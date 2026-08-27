#!/usr/bin/env python
"""Train PromptForge-Optimizer (Phase 2 LoRA). GPU-first."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PromptForge optimizer (LoRA)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "optimizer.yaml"),
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--final-model-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    if args.require_gpu:
        os.environ["PROMPTFORGE_REQUIRE_GPU"] = "1"

    from promptforge.config import OptimizerConfig
    from promptforge.training.train_optimizer import train_optimizer

    config = OptimizerConfig.from_yaml(args.config)
    if args.num_examples is not None:
        config.num_examples = args.num_examples
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.base_model:
        config.base_model_name = args.base_model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.final_model_dir:
        config.final_model_dir = args.final_model_dir
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.load_in_4bit:
        config.load_in_4bit = True
    if args.cpu:
        config.prefer_gpu = False
        config.use_fp16 = False
        config.use_bf16 = False

    train_optimizer(config, regenerate_dataset=args.regenerate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
