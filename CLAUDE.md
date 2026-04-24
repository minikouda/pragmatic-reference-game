# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS 288 Spring 2026 final project by Shizhe Zhang. The project investigates **cost-aware clarification in reference games** using Rational Speech Act (RSA) theory and expected utility calculations to determine when an LLM-based agent should ask a clarifying question vs. commit to an answer.

Core idea: model clarification as a cost-utility optimization at inference time (training-free), using referential entropy `H(T | u)` to trigger clarification when ambiguity is high and the cost of a wrong answer outweighs the cost of asking.

## Research Context

- **Task:** Reference game where an agent must identify a target object from a scene given an utterance
- **Key decision:** Ask a clarifying question (cheap but has a cost `c`) vs. commit to a guess (risky under high ambiguity)
- **Framework:** RSA-based pragmatic inference + expected utility: `Reward = Accuracy - c × (Questions Asked)`
- **Metrics:** Cost-Penalized Accuracy, Clarification Frequency (vs. entropy/cost), ECE, Brier Score

## Repository Structure

```
src/refgame/
  data/
    schema.py          # Scene, Object, EvalRecord, Utterance, ListenerOutput dataclasses
    generator.py       # Synthetic scene generator (controllable ambiguity, overlap)
    dataset.py         # load_jsonl helper
    renderer.py        # PIL-based scene image renderer
  speakers/
    base.py            # BaseSpeaker ABC
    vllm.py            # VLLMSpeaker (naive + pragmatic); sends target props as text, raw image
    scene_aware.py     # SceneAwareSpeaker: full distractor list → LLM picks minimal features
    landmark.py        # LandmarkSpeaker (rule) + LandmarkVLLMSpeaker (LLM picks landmark)
    contrastive.py     # ContrastiveSpeaker (rule) + ContrastiveVLLMSpeaker (LLM picks foil)
    ordinal.py         # OrdinalSpeaker: superlatives/uniqueness, rule-based, no LLM
  listeners/
    base.py            # BaseListener ABC
    vllm.py            # VLLMListener: predicts (x,y) coords → distance softmax posterior
    cost_aware.py      # CostAwareListener: EU(commit) vs EU(ask) threshold
  metrics/
    core.py            # brier_score, referential_entropy
  eval/
    harness.py         # run_grid: speaker cache → listener cache → cost sweep
    reporter.py        # save_results, summarize → CPA, accuracy, clarification rate
  utils/
    llm_client.py      # LLMClient: openrouter/anthropic/openai, retry with 429 backoff

data/
  scenes_6_none.jsonl        # 6-object scenes, no forced overlap
  scenes_6_force.jsonl       # 6-object scenes, forced target overlap
  scenes_8_*/scenes_10_*     # 8- and 10-object variants

scripts/
  run_speaker_comparison.py  # Main experiment: all speakers × VLLMListener × cost sweep
  run_diagnostic.py          # Per-scene verbose trace for debugging accuracy failures
```

## Coordinate System

Scene objects store `x_loc`, `y_loc` in **bottom-left=(0,0)** space (y increases upward).
The VLLMListener prompt uses the same convention; no y-flip is applied when matching
predicted coords to object positions.

Location labels (`top`, `bottom-left`, etc.) in the data use the same convention:
small y = visually low on the rendered image, large y = visually high.

## Current Implementation Status

**Speakers implemented:**
- `vllm-naive` / `vllm-pragmatic`: target props in prompt + raw image (no annotation)
- `scene-aware` / `scene-ranked`: full distractor list, LLM picks minimal features
- `landmark-vllm`: LLM selects landmark and spatial relation from distractor list
- `contrastive-vllm`: LLM identifies foil and contrasting feature from distractor list
- `ordinal`: rule-based superlatives/uniqueness (free, no LLM)
- `landmark` / `contrastive`: pure rule-based baselines

**Listener:**
- `VLLMListener`: predicts (x,y) in bottom-left coords → inverse-distance softmax posterior
- `CostAwareListener`: wraps any listener; commits if `max(posterior) ≥ 1 - cost_c`, else asks

**Key design decisions:**
- Speaker receives target properties as text (not visual annotation) to avoid color hallucination
- Listener predicts coordinates, not object indices — avoids annotation/index indirection
- Speaker utterances cached per (scene, speaker); listener posteriors cached per (scene, speaker, listener)

**Smoke test results (scenes_6_none, n=20, gemini-2.0-flash-001):**
- Best accuracy: vllm-pragmatic / scene-aware / contrastive variants at 95%
- Best CPA at c=0.5: ordinal (+0.425, 35% ask rate)
- All speakers ask 100% at c≤0.25 (posterior too flat to trigger commit at low cost)

## Known Issues / Next Steps

- **Flat posterior at low cost**: inverse-distance softmax rarely exceeds `1 - c` for c≤0.25.
  Possible fixes: temperature scaling on distance softmax, sharper falloff function.
- **scene-ranked underperforms** (70% acc): ranking step may over-specify descriptions.
- `CommitAcc` is always NaN (no commits at low cost) — needs posterior sharpening to be meaningful.