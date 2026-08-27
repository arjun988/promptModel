from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_prompt(prompt: str | None, file: str | None) -> str:
    if file:
        return Path(file).read_text(encoding="utf-8").strip()
    if prompt == "-":
        return sys.stdin.read().strip()
    if prompt:
        return prompt
    raise SystemExit("Provide a prompt argument, --file, or '-' for stdin.")


def _cmd_version(_: argparse.Namespace) -> int:
    from promptforge import __version__

    print(__version__)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    from promptforge.local_paths import (
        ensure_dirs,
        load_user_config,
        save_user_config,
    )

    paths = ensure_dirs()
    cfg = load_user_config()
    cfg.setdefault("prefer_gpu", not args.cpu)
    if args.quality_model:
        cfg["quality_model_path"] = args.quality_model
    if args.optimizer_model:
        cfg["optimizer_model_path"] = args.optimizer_model
    saved = save_user_config(cfg)
    print("PromptForge home:", paths["home"])
    print("Models dir:      ", paths["models"])
    print("Config:          ", saved)
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    from promptforge.hub import doctor_report

    report = doctor_report()
    print("PromptForge doctor")
    print("─" * 40)
    for key, value in report.items():
        print(f"{key:24s}: {value}")

    ok = True
    if not report["quality_model"]:
        print("\nMissing quality model → run: promptforge download --quality")
        ok = False
    if not report["optimizer_model"]:
        print("Missing optimizer model → run: promptforge download --optimizer")
        ok = False
    if report["prefer_gpu"] and not report["cuda_available"]:
        print("\nWARNING: prefer_gpu=true but CUDA is unavailable (will use CPU).")
    return 0 if ok else 1


def _cmd_download(args: argparse.Namespace) -> int:
    from promptforge.hub import (
        DEFAULT_OPTIMIZER_REPO,
        DEFAULT_QUALITY_REPO,
        download_defaults,
        download_optimizer_model,
        download_quality_model,
    )

    quality_repo = args.quality_repo or DEFAULT_QUALITY_REPO
    optimizer_repo = args.optimizer_repo or DEFAULT_OPTIMIZER_REPO

    if args.quality and not args.optimizer:
        path = download_quality_model(repo_id=quality_repo, token=args.token)
        print("Downloaded quality →", path)
        return 0
    if args.optimizer and not args.quality:
        path = download_optimizer_model(repo_id=optimizer_repo, token=args.token)
        print("Downloaded optimizer →", path)
        return 0

    paths = download_defaults(
        quality_repo=quality_repo,
        optimizer_repo=optimizer_repo,
        token=args.token,
        update_config=not args.no_config,
    )
    print("Downloaded quality   →", paths["quality"])
    print("Downloaded optimizer →", paths["optimizer"])
    if not args.no_config:
        print("Updated ~/.promptforge/config.yaml")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from promptforge import PromptForge

    prompt = _read_prompt(args.prompt, args.file)
    pf = PromptForge(
        quality_model_path=args.model,
        prefer_gpu=not args.cpu,
        require_quality=True,
    )
    result = pf.analyze(prompt)
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
    print("\nRun:\n  promptforge optimize \"...\"")
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    from promptforge import PromptForge

    prompt = _read_prompt(args.prompt, args.file)
    pf = PromptForge(
        quality_model_path=args.quality_model,
        optimizer_model_path=args.model,
        prefer_gpu=not args.cpu,
        require_optimizer=True,
    )
    analysis = None
    if pf.scorer is not None and not args.no_analysis:
        analysis = pf.analyze(prompt)

    result = pf.optimize(
        prompt,
        analysis=analysis,
        task_type=args.task_type,
        use_scorer_analysis=False,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("Optimized Prompt")
    print("─" * 32)
    print(result["optimized_prompt"])
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from promptforge import PromptForge

    prompt = _read_prompt(args.prompt, args.file)
    pf = PromptForge(
        quality_model_path=args.quality_model,
        optimizer_model_path=args.optimizer_model,
        prefer_gpu=not args.cpu,
        require_quality=True,
        require_optimizer=True,
    )
    result = pf.run(
        prompt,
        task_type=args.task_type,
        rescore_optimized=not args.no_rescore,
    )
    if args.output:
        pf.export_result(result, args.output)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    before = result["before"]["quality_score"]
    after = result["after"].get("quality_score", "n/a")
    print(f"Prompt Quality: {before} → {after}  (Δ {result['delta'].get('quality_score', 'n/a')})")
    if result["before"].get("issues"):
        print("\nProblems:")
        for issue in result["before"]["issues"]:
            print(f"  ✗ {issue}")
    print("\nChanges:")
    for change in result.get("changes", []):
        print(f"  + {change}")
    print("\nOptimized Prompt")
    print("─" * 32)
    print(result["optimized_prompt"])
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
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
        require_quality=True,
        require_optimizer=True,
    )
    report = run_pipeline_evaluation(pf, task_type=args.task_type)
    downstream = run_downstream_proxy_eval(pf, task_type=args.task_type)
    payload = {"pipeline": report, "downstream_proxy": downstream}

    out = args.output or "outputs/phase3_eval/pipeline_report.json"
    save_eval_report(payload, out)
    print(json.dumps(payload["pipeline"]["summary"], indent=2))
    print("\nDownstream proxy:")
    print(json.dumps(payload["downstream_proxy"]["summary"], indent=2))
    print(f"\nWrote {out}")
    return 0


def _cmd_space(args: argparse.Namespace) -> int:
    import importlib.util
    from pathlib import Path

    from promptforge.local_paths import resolve_model_path

    app_path = Path(__file__).resolve().parents[2] / "demo" / "app.py"
    if not app_path.exists():
        app_path = Path.cwd() / "demo" / "app.py"
    spec = importlib.util.spec_from_file_location("promptforge_demo_app", app_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load Gradio app at {app_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    quality = resolve_model_path(args.quality_model, kind="quality")
    optimizer = resolve_model_path(args.optimizer_model, kind="optimizer")
    if not quality or not optimizer:
        raise SystemExit("Models missing. Run: promptforge download")

    demo = mod.build_demo(
        quality_model_path=quality,
        optimizer_model_path=optimizer,
        prefer_gpu=not args.cpu,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def _cmd_train_quality(args: argparse.Namespace) -> int:
    from promptforge.config import QualityScorerConfig
    from promptforge.training.train_quality import train_quality_scorer

    config = (
        QualityScorerConfig.from_yaml(args.config)
        if args.config
        else QualityScorerConfig()
    )
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


def _cmd_train_optimizer(args: argparse.Namespace) -> int:
    from promptforge.config import OptimizerConfig
    from promptforge.training.train_optimizer import train_optimizer

    config = (
        OptimizerConfig.from_yaml(args.config)
        if args.config
        else OptimizerConfig()
    )
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
    if args.base_model:
        config.base_model_name = args.base_model
    if args.cpu:
        config.prefer_gpu = False
        config.use_fp16 = False
        config.use_bf16 = False
    if args.load_in_4bit:
        config.load_in_4bit = True

    train_optimizer(config, regenerate_dataset=args.regenerate)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptforge",
        description="PromptForge — local prompt quality scoring & optimization",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ver = sub.add_parser("version", help="Print package version")
    ver.set_defaults(func=_cmd_version)

    init = sub.add_parser("init", help="Create ~/.promptforge config + models dirs")
    init.add_argument("--quality-model", type=str, default=None)
    init.add_argument("--optimizer-model", type=str, default=None)
    init.add_argument("--cpu", action="store_true")
    init.set_defaults(func=_cmd_init)

    doctor = sub.add_parser("doctor", help="Check local install, GPU, and model paths")
    doctor.set_defaults(func=_cmd_doctor)

    dl = sub.add_parser("download", help="Download models from Hugging Face Hub")
    dl.add_argument("--quality", action="store_true", help="Download quality model only")
    dl.add_argument("--optimizer", action="store_true", help="Download optimizer only")
    dl.add_argument("--quality-repo", type=str, default=None)
    dl.add_argument("--optimizer-repo", type=str, default=None)
    dl.add_argument("--token", type=str, default=None)
    dl.add_argument("--no-config", action="store_true")
    dl.set_defaults(func=_cmd_download)

    analyze = sub.add_parser("analyze", help="Score a prompt")
    analyze.add_argument("prompt", nargs="?", default=None)
    analyze.add_argument("--file", type=str, default=None)
    analyze.add_argument("--model", type=str, default=None)
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--cpu", action="store_true")
    analyze.set_defaults(func=_cmd_analyze)

    optimize = sub.add_parser("optimize", help="Optimize a prompt")
    optimize.add_argument("prompt", nargs="?", default=None)
    optimize.add_argument("--file", type=str, default=None)
    optimize.add_argument("--model", type=str, default=None)
    optimize.add_argument("--quality-model", type=str, default=None)
    optimize.add_argument("--task-type", type=str, default="general")
    optimize.add_argument("--no-analysis", action="store_true")
    optimize.add_argument("--json", action="store_true")
    optimize.add_argument("--cpu", action="store_true")
    optimize.set_defaults(func=_cmd_optimize)

    run = sub.add_parser("run", help="Analyze + optimize + compare")
    run.add_argument("prompt", nargs="?", default=None)
    run.add_argument("--file", type=str, default=None)
    run.add_argument("--quality-model", type=str, default=None)
    run.add_argument("--optimizer-model", type=str, default=None)
    run.add_argument("--task-type", type=str, default="general")
    run.add_argument("--no-rescore", action="store_true")
    run.add_argument("--output", type=str, default=None)
    run.add_argument("--json", action="store_true")
    run.add_argument("--cpu", action="store_true")
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("eval", help="Run evaluation reports")
    ev.add_argument("--quality-model", type=str, default=None)
    ev.add_argument("--optimizer-model", type=str, default=None)
    ev.add_argument("--task-type", type=str, default="general")
    ev.add_argument("--output", type=str, default="outputs/phase3_eval/pipeline_report.json")
    ev.add_argument("--cpu", action="store_true")
    ev.set_defaults(func=_cmd_eval)

    space = sub.add_parser("space", help="Launch Gradio demo")
    space.add_argument("--quality-model", type=str, default=None)
    space.add_argument("--optimizer-model", type=str, default=None)
    space.add_argument("--host", type=str, default="0.0.0.0")
    space.add_argument("--port", type=int, default=7860)
    space.add_argument("--share", action="store_true")
    space.add_argument("--cpu", action="store_true")
    space.set_defaults(func=_cmd_space)

    train_q = sub.add_parser("train-quality", help="Train Phase-1 quality scorer")
    train_q.add_argument("--config", type=str, default="configs/quality_scorer.yaml")
    train_q.add_argument("--num-examples", type=int, default=None)
    train_q.add_argument("--epochs", type=int, default=None)
    train_q.add_argument("--batch-size", type=int, default=None)
    train_q.add_argument("--output-dir", type=str, default=None)
    train_q.add_argument("--final-model-dir", type=str, default=None)
    train_q.add_argument("--dataset-path", type=str, default=None)
    train_q.add_argument("--regenerate", action="store_true")
    train_q.add_argument("--cpu", action="store_true")
    train_q.set_defaults(func=_cmd_train_quality)

    train_o = sub.add_parser("train-optimizer", help="Train Phase-2 LoRA optimizer")
    train_o.add_argument("--config", type=str, default="configs/optimizer.yaml")
    train_o.add_argument("--num-examples", type=int, default=None)
    train_o.add_argument("--epochs", type=int, default=None)
    train_o.add_argument("--batch-size", type=int, default=None)
    train_o.add_argument("--base-model", type=str, default=None)
    train_o.add_argument("--output-dir", type=str, default=None)
    train_o.add_argument("--final-model-dir", type=str, default=None)
    train_o.add_argument("--dataset-path", type=str, default=None)
    train_o.add_argument("--regenerate", action="store_true")
    train_o.add_argument("--load-in-4bit", action="store_true")
    train_o.add_argument("--cpu", action="store_true")
    train_o.set_defaults(func=_cmd_train_optimizer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
