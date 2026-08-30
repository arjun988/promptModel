# PromptForge

**Local-first prompt quality scoring and optimization.**

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

[Docs](docs/LOCAL.md) · [Product plan](docs/PRD.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)

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
python -m promptforge download \
  --quality-repo ArjunShukla/PromptForge-Quality \
  --optimizer-repo ArjunShukla/PromptForge-Optimizer
```

- Quality: https://huggingface.co/ArjunShukla/PromptForge-Quality  
- Optimizer: https://huggingface.co/ArjunShukla/PromptForge-Optimizer

---

## Results

Benchmarks from local training on **RTX 5060 Laptop (8 GB)** · Windows · `torch+cu128`.

### Quality scorer

| Split | MAE | Pearson |
|-------|----:|--------:|
| Validation | **2.73** | **0.993** |
| Test (overall) | **0.96** | **0.999** |

25k synthetic examples · 3 epochs · ~33 min

### Prompt optimizer

| Metric | Value |
|--------|------:|
| Validation loss | **0.121** |
| Train loss | 0.466 |
| Data | **800** curated weak→strong pairs |
| Epochs | 6 |
| Wall time | ~87 min |

Config: `configs/optimizer_fast_8gb.yaml` (seq 512, grad checkpointing, batch 1 × accum 8)

---

## Quickstart

### Install from PyPI

```bash
pip install tuneprompt
```

> **Note:** The PyPI package is **`tuneprompt`** (`promptforge` is already taken).  
> Import and CLI stay the same: `from promptforge import PromptForge` · `promptforge` / `tuneprompt`.

### Install from source

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

### Configure & run

```bash
# Point at your trained checkpoints
python -m promptforge init \
  --quality-model outputs/promptforge-quality-model \
  --optimizer-model outputs/promptforge-optimizer-model

python -m promptforge doctor
python -m promptforge run "Build me a website for a startup"
python -m promptforge analyze "Make an app." --json
```

> On some Windows setups, Application Control blocks `.venv\Scripts\promptforge.exe`. Prefer `python -m promptforge …`.

### Python API

```python
from promptforge import PromptForge

pf = PromptForge.from_config()

print(pf.analyze("Make an app."))
result = pf.run("Make an app about social media like facebook and stuff")
print(result["optimized_prompt"])
print(result["delta"]["quality_score"])
```

---

## Train your own

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
huggingface-cli login
python scripts/export_to_hub.py --repo-id ArjunShukla/PromptForge-Quality
python scripts/export_to_hub.py --repo-id ArjunShukla/PromptForge-Optimizer --optimizer
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
| 4 | Local package + CLI | Done |
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
