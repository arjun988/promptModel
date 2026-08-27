# PRD — PromptForge

**Product:** PromptForge  
**Type:** Open-source AI models + local developer tooling  
**Status:** Draft / MVP planning  
**Last updated:** 2026-08-27

---

## 1. Overview

### 1.1 One-line description

An open-source AI system that evaluates, scores, analyzes, and optimizes prompts for LLMs.

### 1.2 Problem

Most prompts are vague, incomplete, or ambiguous. Developers and users get inconsistent LLM outputs because prompt quality is judged subjectively and improved by hand.

Existing tooling often treats prompt quality as a binary “good / bad” label. That is too simplistic to drive reliable improvement.

### 1.3 Solution

PromptForge is built as **two related models**, not one giant model:

1. **Prompt Quality Scorer** — evaluates a prompt across multiple quality dimensions.
2. **Prompt Optimizer** — improves and restructures the prompt using that analysis.

Training happens in **Google Colab**; the finished models run locally via a Python package, CLI, and (later) integrations.

### 1.4 Core idea

**Input:**

```text
Make me a website for a startup.
```

**Quality analysis output:**

```json
{
  "quality_score": 42,
  "clarity": 51,
  "specificity": 28,
  "context": 35,
  "constraints": 20,
  "ambiguity": 81,
  "missing_information": [
    "target audience",
    "website type",
    "technology",
    "visual style",
    "required features"
  ]
}
```

**Optimized prompt output:**

```text
Build a modern SaaS landing page for an AI developer
tool targeting software engineers.

Requirements:
- Next.js + TypeScript
- Tailwind CSS
- Responsive design
- Hero section
- Feature section
- Pricing section
- GitHub CTA
- Dark modern visual style

Return the complete implementation...
```

### 1.5 Differentiator

Not “we fine-tuned an LLM.”

> PromptForge quantitatively evaluates prompt quality and automatically improves prompts, with measurable downstream performance gains.

Example Hugging Face positioning:

```text
PromptForge
──────────────────────────────

              Original    Optimized

Quality          31          92
Clarity          42          94
Specificity      18          89

Downstream Task
Success Rate     61%         78%
```

---

## 2. Goals & non-goals

### 2.1 Goals (MVP)

- Score prompts 0–100 with multi-dimensional quality signals.
- Detect weaknesses and missing information.
- Optimize prompts into structured, actionable versions.
- Return **structured JSON**, not free-form prose.
- Train in Colab; ship a local Python API + CLI.
- Publish models on Hugging Face Hub.
- Prove that optimized prompts improve downstream LLM results.

### 2.2 Non-goals (MVP)

- Separate models per domain (coding, writing, etc.) at launch.
- Full SaaS product / hosted API as the first deliverable.
- Training a foundation model from scratch.
- Building all ecosystem integrations (VS Code, Chrome, Raycast, etc.) in v1.
- Manually labeling hundreds of thousands of examples.

---

## 3. Users & use cases

### 3.1 Primary users

- Developers building LLM apps who need better prompts.
- Prompt engineers and researchers benchmarking prompt quality.
- Open-source contributors experimenting with local models.

### 3.2 Core use cases

| ID | Use case | Outcome |
|----|----------|---------|
| UC1 | Score a prompt | Multi-dimension scores + overall quality |
| UC2 | Diagnose a weak prompt | Issues + missing information |
| UC3 | Optimize a prompt | Improved prompt + change summary |
| UC4 | Compare before/after | Score delta + structural improvements |
| UC5 | Integrate in code | Structured JSON via Python API |
| UC6 | Use from terminal | Analyze/optimize via CLI |

---

## 4. Product features

### 4.1 Feature 1 — Prompt scoring

Input prompt → overall score and dimension scores (0–100).

Example:

```text
Overall:       74
Clarity:       88
Specificity:   62
Context:       70
Constraints:   51
Completeness:  68
Ambiguity:     31
```

### 4.2 Feature 2 — Weakness detection

Identify problems such as:

- Missing context
- Ambiguous objective
- No target audience
- No output format
- Conflicting requirements
- Insufficient constraints
- Vague terminology

### 4.3 Feature 3 — Missing information

Example:

```text
Prompt:
"Create a workout plan."

Missing:
- goal
- experience level
- available equipment
- schedule
- duration
```

### 4.4 Feature 4 — Prompt optimization

Input:

```text
Make a Python API.
```

Output:

```text
Create a production-ready REST API using Python and FastAPI.

Requirements:
- Python 3.12+
- FastAPI
- PostgreSQL
- SQLAlchemy
- JWT authentication
- Input validation
- Error handling
- OpenAPI documentation

Return:
1. Project structure
2. Installation instructions
3. Complete source code
4. Environment configuration
5. Example API requests
```

### 4.5 Feature 5 — Prompt rewriting modes (post-MVP / eventually)

Modes / domains:

```text
General
Coding
Research
Writing
Image generation
Data analysis
Reasoning
Agent
RAG
System prompt
```

Optional target model conditioning:

```text
Optimize for: Coding
Model: Claude / GPT / Gemini / Llama / Generic
```

**MVP constraint:** Do **not** train separate models per domain. Use one model with task/domain labels.

### 4.6 Feature 6 — Before / after comparison

Show original vs optimized scores and what changed:

```text
Original                         Optimized

Score: 41                       Score: 91

"Make a website"                "Build a responsive SaaS
                                 landing page for..."
```

Change summary:

```text
+ Added objective
+ Added target audience
+ Added constraints
+ Added output format
+ Added technical requirements
```

### 4.7 Feature 7 — Structured output (critical)

The system must return machine-readable structured output, for example:

```json
{
  "score": 87,
  "dimensions": {
    "clarity": 91,
    "specificity": 84,
    "context": 79,
    "constraints": 88,
    "completeness": 85,
    "ambiguity": 12
  },
  "issues": [],
  "missing_information": [],
  "optimized_prompt": "..."
}
```

This is required for API/CLI/integration usability.

---

## 5. What the model should learn

Do **not** train only “good prompt / bad prompt.”

Train **prompt quality dimensions**:

| Dimension | Meaning |
|-----------|---------|
| Clarity | Is the request understandable? |
| Specificity | Are details concrete enough? |
| Context | Is background / domain context present? |
| Goal definition | Is the objective explicit? |
| Constraints | Are limits, stack, style, scope defined? |
| Structure | Is the prompt organized well? |
| Completeness | Are necessary fields present? |
| Ambiguity | How open to conflicting interpretations? |
| Actionability | Can a model execute it cleanly? |
| Output specification | Is the expected return format defined? |

Flow:

```text
Prompt
   │
   ▼
┌──────────────────────┐
│   PromptForge Model  │
└──────────────────────┘
   │
   ├── Quality Score
   ├── Clarity
   ├── Specificity
   ├── Context
   ├── Completeness
   ├── Ambiguity
   ├── Missing Information
   └── Improvement Suggestions
```

---

## 6. Architecture

### 6.1 Two-stage design

#### Stage 1 — Quality model

A relatively small encoder/classifier (~100M–400M parameters):

```text
Prompt
  ↓
Encoder
  ↓
Multiple prediction heads
  ↓
Quality dimensions
```

Responsible for scoring, issues, and missing information.

#### Stage 2 — Optimizer

A small instruction-tuned causal LM (LoRA/PEFT fine-tune):

```text
Prompt
+
Quality analysis
+
Optimization instructions
        ↓
      LLM
        ↓
Optimized prompt
```

Rationale: scoring and generation are different tasks; forcing one model to do both poorly is worse than a clean two-stage pipeline.

### 6.2 Combined pipeline

```text
                  Prompt
                    │
                    ▼
             Quality Model
                    │
        ┌───────────┴──────────┐
        │                      │
      Score                 Problems
        │                      │
        └──────────┬───────────┘
                   ▼
             Optimizer LLM
                   │
                   ▼
           Optimized Prompt
```

### 6.3 Recommended model stack

| Component | Recommendation |
|-----------|----------------|
| Quality scorer base | **ModernBERT** (alternatives: DeBERTa, RoBERTa) |
| Optimizer base | Small open instruction model + **LoRA/PEFT** |
| Training compute | Google Colab |
| Packaging / inference | Local Python + Transformers |

Hugging Face PEFT freezes the pretrained model and trains a small adapter, reducing memory and compute — suitable for Colab/local experimentation.

---

## 7. Tech stack

### 7.1 Training

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Accelerate
- scikit-learn
- Evaluate

### 7.2 Experiment tracking

- Initial: CSV / JSON + TensorBoard
- Later: Weights & Biases

### 7.3 Hosting & demo

- Model hosting: Hugging Face Hub
- Demo: Gradio (Hugging Face Space)

### 7.4 Local inference

- Initial: PyTorch + Transformers
- Later: ONNX, llama.cpp, Ollama

---

## 8. Dataset design

Dataset quality is the most important part of the project.

### 8.1 Schema

Each example should include:

```text
prompt
quality_score
clarity
specificity
context
constraints
completeness
ambiguity
task_type
issues
missing_information
optimized_prompt
```

Example:

```json
{
  "prompt": "Build me an app",
  "quality_score": 18,
  "clarity": 30,
  "specificity": 5,
  "context": 0,
  "constraints": 0,
  "completeness": 8,
  "ambiguity": 95,
  "task_type": "coding",
  "issues": [
    "missing_platform",
    "missing_requirements",
    "missing_technology"
  ],
  "missing_information": [
    "platform",
    "target_users",
    "features",
    "technology"
  ],
  "optimized_prompt": "Build a..."
}
```

### 8.2 Generation strategy

Do **not** manually write hundreds of thousands of prompts.

Combine:

1. Existing prompt datasets
2. Synthetic generation
3. Human-created gold examples
4. Hard negatives

Generate quality ladders so the model learns progression:

```text
Level 1 — "Make an app."
Level 2 — "Make a fitness app."
Level 3 — "Make a fitness tracking app with authentication."
Level 4 — "Build a React Native fitness tracking app..."
```

### 8.3 Pairwise / ranking data

Create preference pairs and rankings:

```text
Prompt A: Build me an ecommerce website.
Prompt B: Build a responsive ecommerce website using Next.js,
          TypeScript and PostgreSQL for selling men's clothing.
          Include authentication, product search, cart,
          checkout and an admin dashboard.

Signal: B > A
Then:   C > B > A
```

This supports ranking objectives in addition to regression/classification.

### 8.4 Dataset sizes

| Stage | Size |
|-------|------|
| Experiment | ~5,000 |
| MVP | ~25,000 |
| Serious model | 100,000+ |
| Advanced | 500,000+ |

**Quality of examples matters more than raw count.**

---

## 9. Training strategy

### 9.1 Model 1 — Quality scorer

```text
Prompt
 ↓
Encoder
 ↓
Multi-label / regression heads
```

Predict:

- quality
- clarity
- specificity
- context
- constraints
- completeness
- ambiguity
- (issues / missing information as applicable)

Loss combination:

```text
classification loss
+ regression loss
+ ranking loss
```

### 9.2 Model 2 — Optimizer

Train:

```text
INPUT
Prompt: "Make a website"
Analysis:
  clarity = 30
  specificity = 5
  missing = ["audience", "features", "technology"]

OUTPUT
Build a modern...
```

Use **LoRA initially** rather than full fine-tuning to keep Colab experiments manageable.

### 9.3 Colab notebook structure

```text
PromptForge.ipynb

01_environment
02_configuration
03_dataset_download
04_dataset_cleaning
05_dataset_generation
06_dataset_validation
07_tokenization
08_quality_model
09_quality_training
10_quality_evaluation
11_optimizer_dataset
12_optimizer_training
13_optimizer_evaluation
14_combined_pipeline
15_inference
16_export
17_huggingface_upload
```

### 9.4 Colab phases

**Phase 1 — Quality scorer**

```text
Dataset → ModernBERT → Fine-tune → Evaluate
```

Targets (set after baseline, not before):

- MAE < target
- F1 > target
- Correlation > target

**Phase 2 — Optimizer**

```text
Prompt + quality analysis → LoRA fine-tuning → optimized prompt
```

**Phase 3 — Combined pipeline**

Wire scorer → optimizer → end-to-end inference and export.

---

## 10. Evaluation

Do not only evaluate whether outputs “look good.”

### 10.1 Quality model metrics

- MAE
- RMSE
- Pearson correlation
- Spearman correlation
- F1
- Accuracy

### 10.2 Optimizer metrics

- Human preference
- LLM-as-judge
- Instruction preservation
- Information preservation
- Prompt improvement (score delta via quality model)

### 10.3 Primary success benchmark (most important)

Run original and optimized prompts through target LLMs and measure downstream task quality:

```text
Original prompt  → target LLM → result quality
Optimized prompt → target LLM → result quality
```

Report:

> PromptForge improves downstream task performance by X%.

This is more compelling than model accuracy alone.

---

## 11. Local product surface

### 11.1 Package layout

```text
promptforge/
│
├── models/
│
├── src/
│   └── promptforge/
│       ├── scorer.py
│       ├── optimizer.py
│       ├── analyzer.py
│       └── pipeline.py
│
├── examples/
│
├── tests/
│
├── demo/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 11.2 Python API

```python
from promptforge import PromptForge

pf = PromptForge()

result = pf.analyze("Build me a website")
# {
#   "score": 21,
#   "issues": ["too_vague", "missing_context", "missing_requirements"]
# }

optimized = pf.optimize("Build me a website")
```

### 11.3 CLI

```bash
promptforge analyze "Build me a website"
```

```text
Prompt Quality: 21/100

Problems:
  ✗ Missing objective
  ✗ Missing audience
  ✗ Missing requirements
  ✗ Missing technology
```

```bash
promptforge optimize "Build me a website"
```

```text
Optimized Prompt
────────────────────────────────
Build a modern responsive SaaS...
```

### 11.4 Workflow principle

Train in Colab; develop the package locally.

Colab = GPU/training environment.  
Local repo = product, API, CLI, tests, demos.

PEFT adapters can be saved locally and pushed to the Hub, matching this workflow.

---

## 12. Releases & ecosystem

### 12.1 Hugging Face model releases

| Model | Role |
|-------|------|
| **PromptForge-Quality** | Prompt → structured quality analysis |
| **PromptForge-Optimizer** | Prompt + analysis → optimized prompt |
| **PromptForge-Base** (later) | Combined pipeline packaging |

### 12.2 Near-term surfaces

```text
                  PromptForge
                      │
        ┌─────────────┼─────────────┐
        │             │             │
      Model          CLI          API
        │             │             │
        │          Local AI      SaaS (later)
        │
   Hugging Face
        │
        └────── Hugging Face Space
```

### 12.3 Later integrations

- VS Code Extension
- Chrome Extension
- Raycast
- Cursor integration
- Claude Code integration
- OpenAI API wrapper
- LangChain integration
- LlamaIndex integration

---

## 13. Phased roadmap

```text
PHASE 1
Dataset
   ↓
Quality scorer
   ↓
Hugging Face

PHASE 2
Prompt optimizer
   ↓
LoRA
   ↓
Hugging Face

PHASE 3
Combined pipeline
   ↓
Evaluation
   ↓
Hugging Face Space

PHASE 4
Local Python package
   ↓
CLI

PHASE 5
VS Code / Cursor integration
   ↓
Open-source developer tool
```

---

## 14. Success criteria

### MVP success

- [ ] Quality scorer returns structured multi-dimension scores for held-out prompts.
- [ ] Weakness + missing-information detection is useful on common vague prompts.
- [ ] Optimizer produces clearly improved prompts (score delta + human/LLM preference).
- [ ] Downstream benchmark shows measurable improvement vs original prompts.
- [ ] Models published on Hugging Face.
- [ ] Local `analyze` / `optimize` API works offline after download.
- [ ] CLI demo works for the README examples.

### Product success (later)

- [ ] Open-source adoption via HF downloads / Stars / Space usage.
- [ ] Integrations used by developers in real workflows.
- [ ] Ecosystem packaging (Space, CLI, package) feels cohesive.

---

## 15. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Synthetic labels are noisy | Seed with human gold set; validate ladders; hard negatives |
| Scorer overfits to style | Diversify domains/task types; ranking pairs |
| Optimizer invents facts / over-constrains | Preserve intent metrics; optional “minimal rewrite” mode later |
| Colab resource limits | LoRA/PEFT; smaller bases; phased training |
| “Looks better” ≠ better downstream | Make downstream LLM eval the north-star metric |
| One model forced to do both tasks | Keep two-stage architecture |

---

## 16. Open decisions

- Exact ModernBERT checkpoint size / variant for MVP.
- Exact optimizer base model (size vs Colab VRAM).
- Final issue taxonomy enum.
- Ambiguity polarity convention (higher = worse; keep consistent in schema + UI).
- Whether MVP returns full optimized prompt in one call or scorer/optimizer separately.
- License choice for models + package (recommend OSI-friendly; decide before Hub upload).

---

## 17. References

- [LoRA methods · Hugging Face PEFT](https://huggingface.co/docs/peft/main/task_guides/lora_based_methods)
- [LoRA conceptual guide · Hugging Face PEFT](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [Fine-tuning · Hugging Face Transformers](https://huggingface.co/docs/transformers/en/training)
- [LoRA package reference · Hugging Face PEFT](https://huggingface.co/docs/peft/main/package_reference/lora)
- [PEFT model config tutorial · Hugging Face PEFT](https://huggingface.co/docs/peft/main/tutorial/peft_model_config)

---

## Appendix A — Example end-to-end contract

**Request**

```python
pf.analyze_and_optimize("Make me a website for a startup.")
```

**Response (illustrative)**

```json
{
  "original_prompt": "Make me a website for a startup.",
  "score": 42,
  "dimensions": {
    "clarity": 51,
    "specificity": 28,
    "context": 35,
    "constraints": 20,
    "completeness": 30,
    "ambiguity": 81
  },
  "issues": [
    "too_vague",
    "missing_audience",
    "missing_requirements",
    "missing_output_format"
  ],
  "missing_information": [
    "target audience",
    "website type",
    "technology",
    "visual style",
    "required features"
  ],
  "optimized_prompt": "Build a modern SaaS landing page for an AI developer tool targeting software engineers.\n\nRequirements:\n- Next.js + TypeScript\n- Tailwind CSS\n- Responsive design\n- Hero, Feature, Pricing sections\n- GitHub CTA\n- Dark modern visual style\n\nReturn the complete implementation...",
  "changes": [
    "Added objective",
    "Added target audience",
    "Added constraints",
    "Added output format",
    "Added technical requirements"
  ]
}
```
