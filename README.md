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
│   │   └── dataset.py         # JSONL I/O, stratified train/val/test split
│   ├── speakers/
│   │   ├── base.py            # BaseSpeaker ABC
│   │   ├── literal.py         # LiteralSpeaker — minimal distinguishing feature subset
│   │   ├── rsa.py             # RSASpeaker (S1) — pragmatic RSA speaker
│   │   └── llm.py             # LLMSpeaker — naive + chain-of-thought via LLM API
│   ├── listeners/
│   │   ├── base.py            # BaseListener ABC
│   │   ├── literal.py         # LiteralListener (L0) — uniform over compatible objects
│   │   ├── rsa.py             # RSAListener (L1) — inverts S1 via exact enumeration
│   │   └── cost_aware.py      # CostAwareListener — EU clarification policy wrapper
│   ├── metrics/
│   │   └── core.py            # entropy, Brier score, ECE, CPA
│   ├── eval/
│   │   ├── harness.py         # run_grid(): full speaker × listener × cost sweep
│   │   └── reporter.py        # Aggregation, LaTeX table export, JSONL/JSON save
│   └── utils/
│       └── llm_client.py      # Unified LLM client (OpenRouter / Anthropic / OpenAI)
├── scripts/
│   ├── generate_data.py       # CLI: generate and save scene datasets
│   └── run_eval.py            # CLI: run evaluation grid, save results
├── report/                    # LaTeX source (note.tex / note.pdf)
├── data/                      # Generated datasets (gitignored)
├── results/                   # Evaluation outputs (gitignored)
└── pyproject.toml
```

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Generate a dataset

```bash
# 500 scenes balanced across ambiguity tiers (low / medium / high)
python scripts/generate_data.py --n_per_tier 167 --n_objects 4 --out data/scenes.jsonl --split

# Output: data/scenes_train.jsonl, data/scenes_val.jsonl, data/scenes_test.jsonl
```

### 3. Run evaluation (rule-based only, no API key needed)

```bash
python scripts/run_eval.py \
    --scenes data/scenes_test.jsonl \
    --costs 0.1 0.25 0.5 \
    --out results/
```

### 4. Run with an LLM speaker (OpenRouter)

```bash
export OPENROUTER_API_KEY=sk-or-...

python scripts/run_eval.py \
    --scenes data/scenes_test.jsonl \
    --llm_model anthropic/claude-haiku-4-5 \
    --workers 8 \
    --out results/
```

Results are saved as:
- `results/eval_records.jsonl` — one record per (scene × speaker × listener × cost)
- `results/eval_summary.json` — aggregated metrics by condition
- `results/eval_by_tier.json` — broken down by ambiguity tier

## Models

### Speakers

| Name | Description |
|------|-------------|
| `literal` | Rule-based. Emits the shortest feature subset that uniquely identifies the target among distractors. Deterministic, no API. |
| `rsa(α,c)` | RSA S1 speaker. Ranks utterances by `α·log L0(t\|u) − cost·len`, then takes the argmax (or samples). |
| `llm-naive(model)` | LLM prompted to produce a brief referring expression. |
| `llm-pragmatic(model)` | LLM with chain-of-thought prompt: enumerate distinguishing features, then produce minimal expression. |

### Listeners

| Name | Description |
|------|-------------|
| `literal` | L0. Uniform posterior over all objects whose features contain every content token in the utterance. |
| `rsa(α,c)` | L1. Inverts S1 via exact enumeration: P(t\|u) ∝ S1(u\|t). |
| `cost_aware(base, c)` | Wraps any base listener with the EU clarification decision. Not a standalone listener — it is applied automatically by the evaluation harness for every (listener, cost_c) pair. |

### Scene ambiguity tiers

Scenes are stratified by `min_desc_length` — the minimum number of features needed to uniquely identify the target:

| Tier | `min_desc_length` | Meaning |
|------|-------------------|---------|
| `low` | 1 | One feature (e.g. color) rules out all distractors |
| `medium` | 2 | Two features needed |
| `high` | ≥ 3 | Three or four features needed |

## Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| CPA | `Accuracy − c × ClarificationRate` | Primary metric; higher is better |
| Accuracy | `# correct commits / N` | Ignores ask/commit decision |
| ClarificationRate | `# asks / N` | Should decrease as c increases |
| Brier Score | `(1/N) Σ (p_i − y_i)²` | Per-instance, lower is better |
| ECE | binned \|avg_conf − avg_acc\| | Calibration; 0 = perfect |
| H(T\|u) | `-Σ p_i log p_i` | Referential entropy; 0 = certain |

## Extending the project

**New speaker:** subclass `BaseSpeaker` in `src/refgame/speakers/`, implement `name` and `speak(scene, target_idx) -> Utterance`.

**New listener:** subclass `BaseListener` in `src/refgame/listeners/`, implement `name` and `listen(scene, utterance) -> ListenerOutput`.

Both are automatically picked up by `run_grid()` in the harness.

## Compile the report

```bash
cd report && pdflatex note.tex
```

## Dependencies

- Python ≥ 3.11
- `openai` ≥ 1.30 (also used for OpenRouter)
- `anthropic` ≥ 0.25
- `numpy`, `pandas`, `tqdm`
