# Notebooks

Two tracks — same phases, different style.

## `colab/` — self-contained experiments

Upload to Google Colab and run with a **GPU** runtime.  
Logic is inlined in cells (good for training experiments).

| Phase | File |
|------:|------|
| 1 Quality scorer | [colab/01_quality_scorer.ipynb](colab/01_quality_scorer.ipynb) |
| 2 Prompt optimizer | [colab/02_prompt_optimizer.ipynb](colab/02_prompt_optimizer.ipynb) |
| 3 Combined pipeline | [colab/03_combined_pipeline.ipynb](colab/03_combined_pipeline.ipynb) |
| 4 Local package check | [colab/04_local_package.ipynb](colab/04_local_package.ipynb) |

## `package/` — package-driven drivers

Install the repo (`pip install -e .`) and call `src/promptforge` — same code as local scripts/CLI.

| Phase | File |
|------:|------|
| 1 | [package/01_quality_scorer.ipynb](package/01_quality_scorer.ipynb) |
| 2 | [package/02_prompt_optimizer.ipynb](package/02_prompt_optimizer.ipynb) |
| 3 | [package/03_combined_pipeline.ipynb](package/03_combined_pipeline.ipynb) |
| 4 | [package/04_local_package.ipynb](package/04_local_package.ipynb) |

## Recommended order

1. Run **colab/01** (or package/01) → save quality model  
2. Run **colab/02** → save optimizer adapter  
3. Run **colab/03** → end-to-end + Gradio  
4. Run **04** / local CLI for product smoke tests  

Phase 5 (editor integrations) has no notebook yet.
