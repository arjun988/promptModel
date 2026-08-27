"""
PromptForge Gradio demo — Hugging Face Space entrypoint.

Launch locally:
  python demo/app.py

Or:
  promptforge space --quality-model ... --optimizer-model ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_demo(
    quality_model_path: str,
    optimizer_model_path: str,
    prefer_gpu: bool = True,
):
    import gradio as gr

    from promptforge import PromptForge

    pf = PromptForge(
        quality_model_path=quality_model_path,
        optimizer_model_path=optimizer_model_path,
        prefer_gpu=prefer_gpu,
    )

    def run_pipeline(prompt: str, task_type: str, rescore: bool):
        if not prompt or not prompt.strip():
            return "Please enter a prompt.", "", "", ""

        result = pf.run(
            prompt.strip(),
            task_type=task_type,
            rescore_optimized=rescore,
        )

        before = result["before"]
        after = result["after"]
        delta = result["delta"]

        scoreboard = (
            f"### Before / After\n"
            f"| | Original | Optimized | Δ |\n"
            f"|---|---:|---:|---:|\n"
            f"| **Quality** | {before['quality_score']} | {after.get('quality_score', '—')} | "
            f"{delta.get('quality_score', '—')} |\n"
        )

        dim_rows = []
        dims = before.get("dimensions", {})
        after_dims = after.get("dimensions", {})
        for key in dims:
            b = dims.get(key, 0)
            a = after_dims.get(key, b)
            d = delta.get("dimensions", {}).get(key, a - b)
            dim_rows.append(f"| {key} | {b} | {a} | {d} |")
        if dim_rows:
            scoreboard += (
                "\n### Dimensions\n"
                "| Dimension | Original | Optimized | Δ |\n"
                "|---|---:|---:|---:|\n"
                + "\n".join(dim_rows)
            )

        changes = result.get("changes", [])
        changes_md = "### What changed\n" + "\n".join(f"- {c}" for c in changes)

        issues = before.get("issues", [])
        missing = before.get("missing_information", [])
        diagnosis = "### Diagnosis (original)\n"
        if issues:
            diagnosis += "**Issues**\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
        if missing:
            diagnosis += "**Missing**\n" + "\n".join(f"- {m}" for m in missing)

        return (
            scoreboard + "\n\n" + changes_md + "\n\n" + diagnosis,
            result["optimized_prompt"],
            json.dumps(result, indent=2),
            f"{before['quality_score']} → {after.get('quality_score', 'n/a')}",
        )

    examples = [
        ["Make an app.", "coding", True],
        ["Build me a website.", "coding", True],
        ["Write something about AI.", "writing", True],
        ["Make a Python API for beginners.", "coding", True],
        ["Create a workout plan.", "general", True],
    ]

    with gr.Blocks(title="PromptForge") as demo:
        gr.Markdown(
            """
# PromptForge
Evaluate prompt quality and automatically optimize prompts.

**Pipeline:** Prompt → Quality Scorer → Optimizer → Re-score
"""
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Original prompt",
                    lines=6,
                    placeholder="Make me a website for a startup.",
                )
                task_type = gr.Dropdown(
                    choices=[
                        "general",
                        "coding",
                        "writing",
                        "research",
                        "data",
                        "creative",
                    ],
                    value="general",
                    label="Task type",
                )
                rescore = gr.Checkbox(value=True, label="Re-score optimized prompt")
                btn = gr.Button("Analyze & Optimize", variant="primary")
            with gr.Column():
                score_line = gr.Textbox(label="Quality score", interactive=False)
                report = gr.Markdown(label="Report")
                optimized = gr.Textbox(label="Optimized prompt", lines=12)
                raw = gr.Code(label="Structured JSON", language="json")

        btn.click(
            fn=run_pipeline,
            inputs=[prompt, task_type, rescore],
            outputs=[report, optimized, raw, score_line],
        )
        gr.Examples(examples=examples, inputs=[prompt, task_type, rescore])

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch PromptForge Gradio Space")
    parser.add_argument(
        "--quality-model",
        default=os.environ.get(
            "PROMPTFORGE_QUALITY_MODEL",
            str(ROOT / "outputs" / "promptforge-quality-model"),
        ),
    )
    parser.add_argument(
        "--optimizer-model",
        default=os.environ.get(
            "PROMPTFORGE_OPTIMIZER_MODEL",
            str(ROOT / "outputs" / "promptforge-optimizer-model"),
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    demo = build_demo(
        quality_model_path=args.quality_model,
        optimizer_model_path=args.optimizer_model,
        prefer_gpu=not args.cpu,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
