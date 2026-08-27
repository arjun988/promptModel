---
title: PromptForge
emoji: ⚒️
colorFrom: slate
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - prompt-engineering
  - llm
  - prompt-optimization
---

# PromptForge Space

Combined pipeline:

1. **PromptForge-Quality** — multi-dimension prompt scoring
2. **PromptForge-Optimizer** — LoRA prompt rewriting

Set Space secrets / variables:

- `PROMPTFORGE_QUALITY_MODEL` — HF repo or local path for the scorer
- `PROMPTFORGE_OPTIMIZER_MODEL` — HF repo or local path for the optimizer

Or edit defaults in `app.py`.
