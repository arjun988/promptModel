# Contributing to PromptForge

Thanks for contributing.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[demo,dev]"
# PyPI name will be: pip install tuneprompt
promptforge doctor
# or: tuneprompt doctor
pytest -q
```

## Project layout

- `src/promptforge/` — library + CLI (source of truth)
- `notebooks/colab/` — self-contained Colab training notebooks
- `notebooks/package/` — thin notebooks that import the package
- `scripts/` — train / eval / export entrypoints
- `docs/PRD.md` — product requirements

## Workflow

1. Train / experiment in Colab notebooks (GPU).
2. Keep reusable logic in `src/promptforge/` (not only in notebooks).
3. Expose features via CLI (`promptforge ...`) and Python API.
4. Add/adjust tests under `tests/` for non-GPU unit logic.

## Phases

| Phase | Focus |
|------:|-------|
| 1 | Quality scorer |
| 2 | LoRA optimizer |
| 3 | Combined pipeline + Space |
| 4 | Local package + CLI |
| 5 | VS Code / Cursor (planned) |

Prefer small, focused PRs aligned to one phase or bugfix.

## Style

- Python 3.10+
- GPU-first for training paths
- Structured JSON outputs for analyze/optimize/run
- Do not commit model weights, datasets, or secrets

## License

By contributing, you agree your work is licensed under the MIT License.
