# Local install guide (Phase 4)

## Install

```bash
# from repo root
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -U pip
pip install -e ".[demo,dev]"
```

## First-time setup

```bash
promptforge init
promptforge doctor

# After you publish models to the Hub:
promptforge download --quality-repo YOUR_USER/PromptForge-Quality \
  --optimizer-repo YOUR_USER/PromptForge-Optimizer

# Or point at Colab exports:
promptforge init \
  --quality-model outputs/promptforge-quality-model \
  --optimizer-model outputs/promptforge-optimizer-model
```

## CLI

```bash
promptforge analyze "Build me a website"
promptforge optimize "Build me a website"
promptforge run "Build me a website"
promptforge run --file prompt.txt --json
echo "Make an app." | promptforge analyze -
promptforge eval
promptforge space
```

## Python API

```python
from promptforge import PromptForge

pf = PromptForge.from_config()

print(pf.analyze("Build me a website"))
print(pf.optimize("Build me a website"))
print(pf.run("Build me a website"))
```

## Notebooks

- Self-contained Colab: [`notebooks/colab/`](../notebooks/colab/)
- Package-driven: [`notebooks/package/`](../notebooks/package/)
- Index: [`notebooks/README.md`](../notebooks/README.md)

Phase 4 smoke notebook: [`notebooks/colab/04_local_package.ipynb`](../notebooks/colab/04_local_package.ipynb)

## Environment variables

| Variable | Meaning |
|----------|---------|
| `PROMPTFORGE_QUALITY_MODEL` | Quality model path or Hub id |
| `PROMPTFORGE_OPTIMIZER_MODEL` | Optimizer model path or Hub id |
| `PROMPTFORGE_HOME` | Override `~/.promptforge` |
| `PROMPTFORGE_CONFIG` | Override config yaml path |
| `PROMPTFORGE_PREFER_GPU` | `1` / `0` |
| `PROMPTFORGE_REQUIRE_GPU` | Fail training if no CUDA |

## Tests

```bash
pytest -q
```
