# PromptForge

**Open-source AI for prompt quality scoring and optimization.**

PromptForge evaluates prompts across multiple quality dimensions, then rewrites them into clearer, more actionable prompts — with a local Python package, CLI, and Gradio demo.

```text
"Make me a website"
        │
        ▼
┌───────────────────┐
│ Quality Scorer    │  → score, dimensions, issues, missing info
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Prompt Optimizer  │  → optimized prompt (LoRA)
└─────────┬─────────┘
          ▼
   before / after comparison
```

---

## Status

| Phase | Deliverable | Status |
|------:|-------------|--------|
| **1** | Quality scorer (ModernBERT) | Done |
| **2** | Prompt optimizer (LoRA) | Done |
| **3** | Combined pipeline + eval + Gradio Space | Done |
| **4** | Local Python package + CLI | Done |
| **5** | VS Code / Cursor integration | Planned |

Full product plan: [docs/PRD.md](docs/PRD.md)

---

## Quickstart

```bash
git clone https://github.com/YOUR_USER/promptModel.git
cd promptModel

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[demo,dev]"
promptforge init
promptforge doctor
```

Point config at trained models (Colab exports or Hub):

```bash
promptforge init \
  --quality-model outputs/promptforge-quality-model \
  --optimizer-model outputs/promptforge-optimizer-model

promptforge analyze "Build me a website"
promptforge optimize "Build me a website"
promptforge run "Build me a website"
```

Python API:

```python
from promptforge import PromptForge

pf = PromptForge.from_config()
print(pf.analyze("Make an app."))
print(pf.run("Build me a website for a startup."))
```

Local setup guide: [docs/LOCAL.md](docs/LOCAL.md)

---

## Phases

### Phase 1 — Quality Scorer

ModernBERT encoder with multi-dimension regression heads.

Scores: `clarity`, `specificity`, `context`, `goal_definition`, `constraints`, `completeness`, `actionability`.

| Colab (self-contained) | Package-driven |
|------------------------|----------------|
| [notebooks/colab/01_quality_scorer.ipynb](notebooks/colab/01_quality_scorer.ipynb) | [notebooks/package/01_quality_scorer.ipynb](notebooks/package/01_quality_scorer.ipynb) |

```bash
python scripts/train_quality.py --require-gpu --regenerate
```

### Phase 2 — Prompt Optimizer

LoRA fine-tune on a small instruction model (`Qwen2.5-0.5B-Instruct` by default).

| Colab (self-contained) | Package-driven |
|------------------------|----------------|
| [notebooks/colab/02_prompt_optimizer.ipynb](notebooks/colab/02_prompt_optimizer.ipynb) | [notebooks/package/02_prompt_optimizer.ipynb](notebooks/package/02_prompt_optimizer.ipynb) |

```bash
python scripts/train_optimizer.py --require-gpu --regenerate
```

### Phase 3 — Combined Pipeline

Score → optimize → re-score → before/after report + Gradio Space.

| Colab | Package-driven |
|-------|----------------|
| [notebooks/colab/03_combined_pipeline.ipynb](notebooks/colab/03_combined_pipeline.ipynb) | [notebooks/package/03_combined_pipeline.ipynb](notebooks/package/03_combined_pipeline.ipynb) |

```bash
promptforge run "Build me a website" --json
python scripts/evaluate_pipeline.py
python demo/app.py
```

### Phase 4 — Local Package + CLI

Installable product with `~/.promptforge` config, `doctor`, and `download`.

| Notebook |
|----------|
| [notebooks/colab/04_local_package.ipynb](notebooks/colab/04_local_package.ipynb) |

```bash
promptforge init
promptforge doctor
promptforge download --quality-repo YOUR_USER/PromptForge-Quality \
  --optimizer-repo YOUR_USER/PromptForge-Optimizer
```

### Phase 5 — Editor Integrations (later)

VS Code / Cursor extension and related developer-tool surfaces. Not started yet.

---

## Repository layout

```text
promptModel/
├── src/promptforge/          # Python package (API + CLI)
├── configs/                  # Training + local defaults
├── scripts/                  # Train / eval / export helpers
├── examples/                 # Minimal usage scripts
├── demo/                     # Gradio app (HF Space)
├── notebooks/
│   ├── colab/                # Self-contained Colab experiments
│   └── package/              # Thin notebooks that call the package
├── docs/
│   ├── PRD.md                # Product requirements
│   └── LOCAL.md              # Local install guide
├── models/                   # Placeholder for downloaded weights
├── data/                     # Generated datasets (gitignored)
├── outputs/                  # Checkpoints / reports (gitignored)
├── tests/
├── pyproject.toml
└── README.md
```

Notebook guide: [notebooks/README.md](notebooks/README.md)

---

## CLI

| Command | Description |
|---------|-------------|
| `promptforge init` | Create `~/.promptforge` |
| `promptforge doctor` | Check GPU, config, models |
| `promptforge download` | Fetch models from Hugging Face |
| `promptforge analyze` | Score a prompt |
| `promptforge optimize` | Rewrite a prompt |
| `promptforge run` | Full analyze → optimize → compare |
| `promptforge eval` | Evaluation reports |
| `promptforge space` | Launch Gradio demo |
| `promptforge train-quality` | Train Phase 1 |
| `promptforge train-optimizer` | Train Phase 2 |

---

## Training notes

- **Train in Colab (GPU)** — use notebooks under `notebooks/colab/`
- **Develop the package locally** — `src/promptforge/`, CLI, tests
- Prefer CUDA + fp16; set `PROMPTFORGE_REQUIRE_GPU=1` to fail without a GPU

Hugging Face upload:

```bash
huggingface-cli login
python scripts/export_to_hub.py --repo-id YOUR_USER/PromptForge-Quality
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

[MIT](LICENSE)
