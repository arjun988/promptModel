# PromptForge

Open-source prompt quality scoring, optimization, and local developer tooling (Phases 1–4).

> Quantitatively evaluate prompt quality and improve prompts — with measurable downstream gains.

## Phase 4 — Local package + CLI

```bash
pip install -e ".[demo,dev]"
promptforge init
promptforge doctor

# After models are trained / published:
promptforge download --quality-repo YOUR_USER/PromptForge-Quality \
  --optimizer-repo YOUR_USER/PromptForge-Optimizer

promptforge analyze "Build me a website"
promptforge optimize "Build me a website"
promptforge run "Build me a website"
```

Python:

```python
from promptforge import PromptForge

pf = PromptForge.from_config()  # uses ~/.promptforge or env
print(pf.analyze("Build me a website"))
```

Details: [docs/LOCAL.md](docs/LOCAL.md) · Colab/local check: `PromptModelphase4.ipynb`

## Phase 3 — Combined pipeline + eval + Space

```bash
pip install -e ".[demo]"

# End-to-end
promptforge run "Build me a website" \
  --quality-model outputs/promptforge-quality-model \
  --optimizer-model outputs/promptforge-optimizer-model --json

# Evaluation report
python scripts/evaluate_pipeline.py

# Gradio demo
python demo/app.py
# or
promptforge space --share
```

Colab: open `PromptModelphase3.ipynb` (GPU). Requires Phase 1 + Phase 2 model folders.

## Phase 2 — Prompt Optimizer (LoRA)

```bash
pip install -e .
python scripts/train_optimizer.py --require-gpu --regenerate
promptforge optimize "Build me a website" --model outputs/promptforge-optimizer-model
```

Colab (self-contained, Phase-1 style): open `PromptModelphase2.ipynb` with a **GPU** runtime.

Package-driven Colab: `notebooks/PromptForge_Phase2_Optimizer.ipynb`

## Phase 1 status

**PromptForge-Quality** — ModernBERT encoder + dual regression heads.

Trained in Colab on ~25k synthetic laddered prompts. Held-out results from the Phase 1 experiment:

| Metric | Value |
|--------|-------|
| Val MAE | ~2.72 |
| Val Pearson | ~0.993 |
| Overall Pearson | ~0.999 |

Vague prompts score low; detailed prompts score high — as expected.

## Architecture

```text
Prompt
  ↓
ModernBERT encoder
  ↓
mean pooling
  ↓
┌─────────────────┬──────────────────┐
│ dimension head  │  quality head    │
│ (7 scores)      │  (overall)       │
└─────────────────┴──────────────────┘
```

Dimensions: `clarity`, `specificity`, `context`, `goal_definition`, `constraints`, `completeness`, `actionability`.

## Repo layout

```text
promptforge/
├── configs/quality_scorer.yaml
├── src/promptforge/
│   ├── models/quality.py      # model + save/load
│   ├── data/                  # synthetic dataset + tokenization
│   ├── training/              # trainer, metrics, train loop
│   ├── scorer.py              # inference API
│   ├── pipeline.py            # PromptForge facade
│   └── cli.py
├── scripts/
│   ├── train_quality.py
│   ├── evaluate_quality.py
│   └── export_to_hub.py
├── notebooks/
│   └── PromptForge_Phase1_Quality.ipynb   # Colab entrypoint
├── examples/
└── PRD.md
```

## Quickstart (local)

```bash
# GPU recommended (CUDA). CPU works but is slow for training.
pip install -e .

# Train Phase 1 (GPU-first, fp16 on CUDA)
python scripts/train_quality.py --require-gpu --regenerate

# Or via CLI
promptforge train-quality --config configs/quality_scorer.yaml --regenerate

# Score a prompt
promptforge analyze "Build me a website" --model outputs/promptforge-quality-model --json
```

Python API:

```python
from promptforge import PromptForge

pf = PromptForge(quality_model_path="outputs/promptforge-quality-model")
print(pf.analyze("Make an app."))
```

## Colab

1. Open `notebooks/PromptForge_Phase1_Quality.ipynb` in Google Colab.
2. Runtime → **GPU** (T4 / L4 / A100).
3. Run all cells (install → train → smoke eval → optional Hub upload).

The notebook installs this repo editable and calls the same training code as local scripts.

## Hugging Face upload

```bash
huggingface-cli login
python scripts/export_to_hub.py --repo-id YOUR_USER/PromptForge-Quality
```

## Force GPU

```bash
# Fail loudly if CUDA is missing
export PROMPTFORGE_REQUIRE_GPU=1
python scripts/train_quality.py --require-gpu
```

## Phase 2 (next)

- PromptForge-Optimizer (LoRA on a small instruction LM)
- Combined pipeline + Gradio Space
- Downstream LLM success benchmark

See [PRD.md](PRD.md) for the full product plan.

## License

MIT
