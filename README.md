# PromptForge

**Local-first prompt quality scoring and optimization.**

Published on PyPI as **[`tuneprompt`](https://pypi.org/project/tuneprompt/)**.

PromptForge scores LLM prompts across seven quality dimensions, then rewrites weak prompts into clear, intent-preserving instructions — runnable on your machine via Python API, CLI, or Gradio.

```text
"Make an app about social media like facebook"
                    │
                    ▼
         ┌─────────────────────┐
         │  Quality Scorer     │  ModernBERT · ~150M
         │  41.5 → issues…     │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Prompt Optimizer   │  Qwen2.5-1.5B + LoRA
         └──────────┬──────────┘
                    ▼
   Build a social media app similar to Facebook…
   profiles · feed · likes · constraints · output format
                    │
                    ▼
              41.5 → 94.0
```

[PyPI](https://pypi.org/project/tuneprompt/) · [Docs](docs/LOCAL.md) · [Product plan](docs/PRD.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)

---

## Why PromptForge

Most prompt tools either **judge** quality or **rewrite** text. PromptForge does both in one local pipeline:

| Capability | What you get |
|------------|--------------|
| **Multi-dimension scoring** | Clarity, specificity, context, goals, constraints, completeness, actionability |
| **Intent-preserving rewrite** | Optimizes the *same* topic — not a generic template |
| **Validation & fallback** | Rejects repetitive / off-topic generations |
| **Runs locally** | ~1.65B total params; trains on 8 GB GPUs |
| **Dev-ready surface** | `pip` package, CLI, Gradio demo, Colab notebooks |

No API key required for inference once models are on disk.

---

## Example

**Input**

```text
Make an app about social media like facebook and stuff
```

**Output (optimizer)**

```text
Build a social media app similar to Facebook for product managers.
This is for a portfolio demo.

Core features:
- User profiles and friend connections
- News feed with posts, likes, and comments
- Basic notifications

Requirements:
- Use Python and Flask.
- Keep the first version simple and usable
- Include error handling and clear project structure

Include short examples.
```

**Quality:** `41.5 → 94.0` (Δ +52.5) · topic preserved · no fallback

---

## Models

| Component | Base | Size | Training |
|-----------|------|------|----------|
| **Scorer** | [`ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) | ~150M | Full fine-tune |
| **Optimizer** | [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | LoRA (base frozen) |

Weights are **not** stored in git. Train locally or download from Hugging Face:

```bash
pip install tuneprompt

python -m promptforge download \
  --quality-repo ArjunShukla/PromptForge-Quality \
  --optimizer-repo ArjunShukla/PromptForge-Optimizer
```

- Quality: https://huggingface.co/ArjunShukla/PromptForge-Quality  
- Optimizer: https://huggingface.co/ArjunShukla/PromptForge-Optimizer

---

## Results

Quality scorer (held-out):

| Split | MAE | Pearson |
|-------|----:|--------:|
| Validation | **2.73** | **0.993** |
| Test (overall) | **0.96** | **0.999** |

---

## Quickstart

### Install

```bash
pip install tuneprompt
```

Package: [`tuneprompt` on PyPI](https://pypi.org/project/tuneprompt/1.0.0/)  
Import module: `promptforge` · CLI: `tuneprompt` or `promptforge`

### Download models & run

```bash
python -m promptforge download \
  --quality-repo ArjunShukla/PromptForge-Quality \
  --optimizer-repo ArjunShukla/PromptForge-Optimizer

python -m promptforge init
python -m promptforge doctor
python -m promptforge run "Build me a website for a startup"
python -m promptforge analyze "Make an app." --json
```

Same via CLI entrypoints:

```bash
tuneprompt run "Build me a website for a startup"
# or
promptforge run "Build me a website for a startup"
```

> On some Windows setups, Application Control blocks `.venv\Scripts\*.exe`. Prefer `python -m promptforge …`.

### Python API

```python
from promptforge import PromptForge

# After download + init, or pass Hub / local paths:
pf = PromptForge(
    quality_model_path="ArjunShukla/PromptForge-Quality",
    optimizer_model_path="ArjunShukla/PromptForge-Optimizer",
)

print(pf.analyze("Make an app."))
result = pf.run("Make an app about social media like facebook and stuff")
print(result["optimized_prompt"])
print(result["delta"]["quality_score"])
```

### Install from source (optional)

```bash
git clone https://github.com/arjun988/promptModel.git
cd promptModel

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -U pip
pip install -e ".[demo,dev]"
```

**GPU tip:** default `pip install torch` is often CPU-only. For NVIDIA (incl. RTX 50-series):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Full local guide: [docs/LOCAL.md](docs/LOCAL.md)

---

## Train your own

Anyone can improve the models with their own data:

```bash
# Phase 1 — quality scorer
python scripts/train_quality.py --require-gpu --regenerate

# Phase 2 — optimizer (recommended on 8GB GPUs)
python scripts/train_optimizer.py --require-gpu --fast --regenerate
```

| Flag / config | Purpose |
|---------------|---------|
| `--fast` | Loads `configs/optimizer_fast_8gb.yaml` |
| `--regenerate` | Rebuild curated optimizer dataset |
| `load_in_4bit: true` | Use if 1.5B LoRA OOMs |

Publish checkpoints:

```bash
hf auth login
hf upload ArjunShukla/PromptForge-Quality outputs/promptforge-quality-model --repo-type model
hf upload ArjunShukla/PromptForge-Optimizer outputs/promptforge-optimizer-model --repo-type model
```

---

## CLI

| Command | Description |
|---------|-------------|
| `init` | Create `~/.promptforge` and register model paths |
| `doctor` | GPU / config / model health check |
| `download` | Pull models from Hugging Face |
| `analyze` | Score a prompt |
| `optimize` | Rewrite a prompt |
| `run` | Score → optimize → compare |
| `eval` | Pipeline evaluation reports |
| `space` | Launch Gradio demo |
| `train-quality` / `train-optimizer` | Training entrypoints |

---

## Project layout

```text
promptModel/
├── src/promptforge/     # Package: scorer, optimizer, pipeline, CLI
├── configs/             # Training + local defaults
├── scripts/             # Train / eval / Hub export
├── demo/                # Gradio app
├── notebooks/
│   ├── colab/           # Self-contained experiments
│   └── package/         # Thin package drivers
├── docs/                # PRD + local setup
├── tests/
└── pyproject.toml
```

Notebooks: [notebooks/README.md](notebooks/README.md)

---

## Roadmap

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 1 | Multi-dimension quality scorer | Done |
| 2 | Intent-preserving prompt optimizer (LoRA) | Done |
| 3 | Combined pipeline + eval + Gradio | Done |
| 4 | Local package + CLI (`tuneprompt`) | Done |
| 5 | VS Code / Cursor extension | Planned |

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, style, and PR expectations.

```bash
pip install -e ".[dev]"
pytest -q
```

---

## License

[MIT](LICENSE) © PromptForge contributors
