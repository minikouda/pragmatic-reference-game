# Cost-Aware Clarification in Reference Games

CS 288 Spring 2026 Final Project — Shizhe Zhang

## Overview

This project investigates **when a language-model agent should ask a clarifying question vs. commit to an answer** in a reference game.  The core idea is to model clarification as a cost-utility optimization at inference time (no fine-tuning required), using referential entropy H(T | u) to trigger clarification when ambiguity is high.

**Decision rule:**

```
E[U | commit] = max_i P(t_i | u)       # confidence if we guess argmax
E[U | ask]    = 1 − c                  # expected reward after asking (perfect resolution assumed)

action = "ask"    if E[U | ask] > E[U | commit]
       = "commit" otherwise
```

Equivalently: ask whenever the listener's confidence falls below the threshold `1 − c`.

**Primary metric:** Cost-Penalized Accuracy (CPA)

```
CPA = Accuracy − c × ClarificationRate
```

## Repository Structure

```
.
├── src/refgame/               # Main package (pip install -e .)
│   ├── data/
│   │   ├── schema.py          # Dataclasses: Object, Scene, Utterance,
│   │   │                      #   ListenerOutput, ClarificationDecision, EvalRecord
│   │   ├── generator.py       # SceneGenerator — controllable referential entropy
│   │   ├── renderer.py        # PNG renderer — generates scene images
│   │   └── dataset.py         # JSONL I/O, stratified train/val/test split
│   ├── speakers/
│   │   ├── base.py            # BaseSpeaker ABC
│   │   ├── literal.py         # LiteralSpeaker — minimal distinguishing feature subset
│   │   ├── rsa.py             # RSASpeaker (S1) — pragmatic RSA speaker
│   │   └── vllm.py            # VLLMSpeaker — naive + CoT via vision-language model
│   ├── listeners/
│   │   ├── base.py            # BaseListener ABC
│   │   ├── literal.py         # LiteralListener (L0) — uniform over compatible objects
│   │   ├── rsa.py             # RSAListener (L1) — inverts S1 via exact enumeration
│   │   ├── cost_aware.py      # CostAwareListener — EU clarification policy wrapper
│   │   └── vllm.py            # VLLMListener — posterior via VLM forced-choice scoring
│   ├── metrics/
│   │   └── core.py            # entropy, Brier score, ECE, CPA
│   ├── eval/
│   │   ├── harness.py         # run_grid(): full speaker × listener × cost sweep
│   │   └── reporter.py        # Aggregation, LaTeX table export, JSONL/JSON save
│   └── utils/
│       └── llm_client.py      # Unified LLM client (OpenRouter / Anthropic / OpenAI)
│                              #   supports text + image (base64 inline) inputs
├── scripts/
│   ├── generate_data.py       # CLI: generate scenes with images
│   └── run_eval.py            # CLI: run evaluation grid, save results
├── report/                    # LaTeX source (note.tex / note.pdf)
├── data/                      # Generated datasets (gitignored)
├── results/                   # Evaluation outputs (gitignored)
└── pyproject.toml
```

## Quickstart

### 1. Install

```bash
pip install -e ".[dev]"
pip install pillow          # required for image rendering
```

### 2. Generate data

See the **Data Generation** section below for full details.

```bash
# Quick start: 500 scenes with PNG images, balanced across ambiguity tiers
python scripts/generate_data.py --n_per_tier 167 --n_objects 6 --out data/scenes --split
```

### 3. Run evaluation

```bash
export OPENROUTER_API_KEY=sk-or-...

python scripts/run_eval.py \
    --jsonl data/scenes_test.jsonl \
    --vllm_model anthropic/claude-haiku-4-5 \
    --symbolic_baselines \
    --workers 8 \
    --out results/
```

Results are saved as:
- `results/eval_records.jsonl` — one record per (scene × speaker × listener × cost)
- `results/eval_summary.json` — aggregated metrics by condition
- `results/eval_by_tier.json` — broken down by ambiguity tier

---

## Data Generation

All data is **synthetic**: scenes are programmatically generated and rendered to PNG images.
No manual annotation is required.

### Scene format

Each scene contains:
- **N objects** drawn on a 330×328 white canvas, each with attributes:
  `color` ∈ {black, blue, green, red, yellow},
  `shape` ∈ {circle, square, triangle},
  `size` ∈ {small (8px), medium (12px), large (16px)},
  `x_loc / y_loc` ∈ [10, 90] (canvas coordinates, y=0 at bottom)
- A **target object** designated by `target_idx`
- An **ambiguity annotation** recording `min_desc_length` and `ambiguity_tier`

### Ambiguity control

Scenes are stratified by `min_desc_length` — the minimum number of symbolic features
needed to uniquely identify the target among distractors:

| Tier | `min_desc_length` | How generated |
|------|-------------------|---------------|
| `low` | 1 | Rejection sampling — one feature suffices to distinguish target |
| `medium` | 2 | Rejection sampling — two features needed |
| `high` | ≥ 3 | Constructive: distractors are built to share 3 features each with the target, blocking all C(4,2)=6 feature-pairs |

### Rendering

`renderer.py` draws each scene to a PNG matching the style of the original
`reference_game_dataset/` images:

```
Canvas : 330 × 328 px, white background, 1px black border
Margin : 5 px inside the border
Coords : pixel_x = 5 + x_loc/100 × 320
         pixel_y = 323 − y_loc/100 × 318   (y-axis inverted: 0 = bottom)
Size   : half_width/radius = raw_size × 3 px
```

### Generate commands

```bash
# 500 scenes (flat, unconstrained ambiguity) with images
python scripts/generate_data.py --n 500 --n_objects 6 --out data/scenes --split

# 500 scenes balanced across tiers (167 low + 167 medium + 167 high)
python scripts/generate_data.py --n_per_tier 167 --n_objects 6 --out data/scenes --split

# Generate from Python directly
python3 - <<'EOF'
from src.refgame.data.generator import GeneratorConfig, SceneGenerator
from src.refgame.data.dataset import save_jsonl, split_dataset

gen    = SceneGenerator(GeneratorConfig(n_objects=6, seed=42))
scenes = gen.generate_with_images(n=500, out_dir="data/generated_scenes", prefix="scene")
train, val, test = split_dataset(scenes)
save_jsonl(train, "data/scenes_train.jsonl")
save_jsonl(val,   "data/scenes_val.jsonl")
save_jsonl(test,  "data/scenes_test.jsonl")
EOF
```

Each JSONL line stores the full scene dict including `image_path` (relative path to the PNG).
The PNG files must remain at that path relative to the working directory when running evaluation.

---

## Models

### Speakers

| Name | Description |
|------|-------------|
| `literal` | Rule-based. Emits the shortest feature subset that uniquely identifies the target. No API. |
| `rsa(α,c)` | RSA S1 speaker. Ranks utterances by `α·log L0(t\|u) − cost·len`, takes argmax. |
| `vllm-naive(model)` | VLM sees the annotated scene image (target highlighted with red box), produces a brief expression. |
| `vllm-pragmatic(model)` | VLM with chain-of-thought prompt: reason about distinguishing features, then produce minimal expression. |

### Listeners

| Name | Description |
|------|-------------|
| `literal` | L0. Uniform posterior over objects whose feature set contains every content token in the utterance. |
| `rsa(α,c)` | L1. Inverts S1 via exact enumeration: P(t\|u) ∝ S1(u\|t). |
| `vllm-listener(model)` | VLM sees the image with all objects labeled 0–N−1, outputs a JSON confidence dict; softmax gives the posterior. |
| `cost_aware(base, c)` | Wraps any listener with the EU clarification decision. Applied automatically by the harness. |

---

## Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| CPA | `Accuracy − c × ClarificationRate` | Primary metric; higher is better |
| Accuracy | `# correct commits / N` | Ignores ask/commit decision |
| ClarificationRate | `# asks / N` | Should decrease as c increases |
| Brier Score | `(1/N) Σ (p_i − y_i)²` | Per-instance, lower is better |
| ECE | binned \|avg_conf − avg_acc\| | Calibration; 0 = perfect |
| H(T\|u) | `−Σ p_i log p_i` | Referential entropy; 0 = certain |

---

## Extending the project

**New speaker:** subclass `BaseSpeaker` in `src/refgame/speakers/`, implement `name` and `speak(scene, target_idx) -> Utterance`.

**New listener:** subclass `BaseListener` in `src/refgame/listeners/`, implement `name` and `listen(scene, utterance) -> ListenerOutput`.

Both are automatically picked up by `run_grid()` in the harness.

---

## Compile the report

```bash
cd report && pdflatex note.tex
```

## Dependencies

- Python ≥ 3.11
- `openai` ≥ 1.30 (also used for OpenRouter)
- `anthropic` ≥ 0.25
- `pillow` ≥ 10.0 (image rendering and annotation)
- `numpy`, `pandas`, `tqdm`
