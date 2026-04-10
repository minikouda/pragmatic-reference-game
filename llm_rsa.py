"""
CS 288 Final Project — LLM-RSA Baseline (Phase 3, Shizhe)
==========================================================
Uses OpenRouter to score P_{S0}(u | t_i) for each candidate object,
then computes the RSA posterior P_{L1}(t | u) and makes a
cost-aware clarification decision.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python llm_rsa.py

The scoring strategy uses a forced-choice prompt:
    "Given object description X, how naturally does the phrase Y refer to it?
     Answer with a single number 1–10."
This sidesteps log-prob access (unavailable on many OpenRouter models) while
still giving a calibrated ranking over candidate objects.
"""

import os
import math
import json
import time
from dataclasses import dataclass

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "meta-llama/llama-3.3-70b-instruct"   # change to any OpenRouter model
COST_C = 0.25                                   # clarification cost

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    objects: list[str]       # e.g. ["large red square", "small blue circle", ...]
    target_idx: int
    utterance: str


# ---------------------------------------------------------------------------
# Dataset: same scenes as demo.py, plus a few more
# ---------------------------------------------------------------------------

SCENES = [
    Scene(
        objects=["large red square", "small blue circle", "medium green triangle", "large yellow pentagon"],
        target_idx=0,
        utterance="the red square",
    ),
    Scene(
        objects=["large red square", "small red circle", "medium blue triangle", "large green star"],
        target_idx=0,
        utterance="the red one",
    ),
    Scene(
        objects=["large red square", "large blue circle", "large green triangle", "large yellow pentagon"],
        target_idx=2,
        utterance="the large object",
    ),
    Scene(
        objects=["small blue square", "small blue circle", "medium red triangle", "large green pentagon"],
        target_idx=1,
        utterance="the small blue thing",
    ),
    Scene(
        objects=["medium red square", "medium blue square", "medium green square", "small yellow circle"],
        target_idx=0,
        utterance="the red square",
    ),
]


# ---------------------------------------------------------------------------
# LLM scoring: P_{S0}(u | t_i)
# ---------------------------------------------------------------------------

SCORE_PROMPT = """\
You are evaluating how naturally a short phrase refers to an object.

Object: {obj}
Phrase: "{utterance}"

On a scale from 1 to 10, how naturally does this phrase refer to this object?
- 10 = the phrase uniquely and perfectly identifies this object
- 5  = the phrase partially matches (shares some attributes)
- 1  = the phrase does not match at all

Reply with ONLY a single integer between 1 and 10. No explanation."""


def score_object(obj: str, utterance: str, retries: int = 3) -> float:
    """Return a raw score in [1, 10] for how well utterance refers to obj."""
    prompt = SCORE_PROMPT.format(obj=obj, utterance=utterance)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            # parse first integer found
            for tok in text.split():
                tok = tok.strip(".,")
                if tok.isdigit():
                    return float(tok)
            # fallback: try converting entire text
            return float(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [warn] scoring failed for '{obj}': {e}")
                return 5.0  # neutral fallback


def rsa_posterior(raw_scores: list[float]) -> list[float]:
    """
    Literal listener posterior via softmax over raw scores:
      P_{L1}(t_i | u) ∝ exp(score_i)
    Uniform prior P(t_i) = 1/N assumed.
    """
    exp_scores = [math.exp(s) for s in raw_scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def brier_score(probs: list[float], target_idx: int) -> float:
    n = len(probs)
    return sum((p - (1.0 if i == target_idx else 0.0)) ** 2
               for i, p in enumerate(probs)) / n


def clarification_decision(probs: list[float], c: float):
    e_commit = max(probs)
    e_ask = 1.0 - c
    return ("ask" if e_ask > e_commit else "commit"), e_commit, e_ask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scene(scene: Scene, idx: int) -> dict:
    print(f"\n--- Scene {idx+1} ---")
    print(f"  Objects   : {scene.objects}")
    print(f"  Utterance : \"{scene.utterance}\"")
    print(f"  Target    : {scene.objects[scene.target_idx]}")

    # Score each object
    raw_scores = []
    for obj in scene.objects:
        score = score_object(obj, scene.utterance)
        raw_scores.append(score)
        print(f"  score({obj!r:30s}) = {score}")

    probs = rsa_posterior(raw_scores)
    pred_idx = probs.index(max(probs))
    correct = int(pred_idx == scene.target_idx)

    H = entropy(probs)
    BS = brier_score(probs, scene.target_idx)
    decision, e_commit, e_ask = clarification_decision(probs, COST_C)

    print(f"\n  Posterior P(t|u): {[f'{p:.3f}' for p in probs]}")
    print(f"  H(T|u)={H:.3f}  BS={BS:.3f}  E_commit={e_commit:.3f}  E_ask={e_ask:.3f}")
    print(f"  Decision: {decision.upper()}  |  Prediction: {scene.objects[pred_idx]}  |  {'CORRECT' if correct else 'WRONG'}")

    return dict(
        utterance=scene.utterance,
        target=scene.objects[scene.target_idx],
        prediction=scene.objects[pred_idx],
        correct=correct,
        H=H,
        BS=BS,
        decision=decision,
        e_commit=e_commit,
    )


def main():
    print("=" * 60)
    print(f"LLM-RSA Baseline  |  model: {MODEL}  |  c={COST_C}")
    print("=" * 60)

    results = []
    for i, scene in enumerate(SCENES):
        result = run_scene(scene, i)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    accuracy = sum(r["correct"] for r in results) / len(results)
    mean_H = sum(r["H"] for r in results) / len(results)
    mean_BS = sum(r["BS"] for r in results) / len(results)
    ask_rate = sum(1 for r in results if r["decision"] == "ask") / len(results)

    print(f"  Accuracy       : {accuracy:.2f}")
    print(f"  Mean H(T|u)    : {mean_H:.3f}")
    print(f"  Mean Brier     : {mean_BS:.3f}")
    print(f"  Ask rate (c={COST_C}): {ask_rate:.2f}")

    # Save results
    with open("results_llm_rsa.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to results_llm_rsa.json")


if __name__ == "__main__":
    main()
