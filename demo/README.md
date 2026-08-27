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

Phase 3 demo — combined pipeline:

1. **PromptForge-Quality** — multi-dimension prompt scoring  
2. **PromptForge-Optimizer** — LoRA prompt rewriting  

## Local

```bash
pip install -e ".[demo]"
python demo/app.py
# or
promptforge space
```

## Space secrets / variables

- `PROMPTFORGE_QUALITY_MODEL` — HF repo or path for the scorer  
- `PROMPTFORGE_OPTIMIZER_MODEL` — HF repo or path for the optimizer  

See the main [README](../README.md) and [docs/PRD.md](../docs/PRD.md).
