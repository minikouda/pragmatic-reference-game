"""
CS 288 Final Project — Demo Experiment (Phase 1)
=================================================
Shows the core pipeline of the project WITHOUT any LLM API calls.
Uses simulated model probabilities to illustrate:

  1. What a reference game scene looks like
  2. Referential entropy H(T | u)
  3. Brier Score (multi-class)
  4. Expected Calibration Error (ECE)
  5. Cost-aware clarification decision E[U_ask] vs E[U_commit]

Run: python demo.py
"""

import math
import random
from dataclasses import dataclass

random.seed(42)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Object:
    color: str
    shape: str
    size: str

    def __str__(self):
        return f"{self.size} {self.color} {self.shape}"


@dataclass
class Scene:
    objects: list[Object]
    target_idx: int
    utterance: str

    @property
    def target(self):
        return self.objects[self.target_idx]


# ---------------------------------------------------------------------------
# Simulated model: P(t_i | u)
# ---------------------------------------------------------------------------

def simulate_model_probs(scene: Scene, noise: float = 0.05) -> list[float]:
    """
    Simulate LLM probabilities by assigning high weight to objects
    whose attributes match words in the utterance, with small noise.
    This stands in for actual LLM log-likelihood scoring.
    """
    utterance_words = scene.utterance.lower().split()
    scores = []
    for obj in scene.objects:
        obj_words = str(obj).lower().split()
        overlap = sum(1 for w in obj_words if w in utterance_words)
        scores.append(math.exp(overlap) + random.uniform(0, noise))
    total = sum(scores)
    return [s / total for s in scores]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def referential_entropy(probs: list[float]) -> float:
    """H(T | u) = -sum p_i * log(p_i)"""
    return -sum(p * math.log(p) for p in probs if p > 0)


def brier_score(probs: list[float], target_idx: int) -> float:
    """Multi-class Brier Score = (1/N) * sum (p_i - y_i)^2"""
    n = len(probs)
    return sum((p - (1.0 if i == target_idx else 0.0)) ** 2 for i, p in enumerate(probs)) / n


def expected_calibration_error(confidences: list[float], corrects: list[int], n_bins: int = 5) -> float:
    """ECE using top predicted confidence per instance."""
    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    ece = 0.0
    total = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        avg_acc = sum(x[1] for x in b) / len(b)
        ece += (len(b) / total) * abs(avg_conf - avg_acc)
    return ece


# ---------------------------------------------------------------------------
# Clarification decision
# ---------------------------------------------------------------------------

def clarification_decision(probs: list[float], cost_c: float) -> tuple[str, float, float]:
    """
    Compare expected utilities:
      E[U_commit] = max_i P(t_i | u)          (accuracy if we commit to argmax)
      E[U_ask]    = E[accuracy after answer] - c
                  ≈ 1.0 - c  (assume question resolves ambiguity perfectly)

    Returns: ("ask" or "commit", E_commit, E_ask)
    """
    e_commit = max(probs)
    e_ask = 1.0 - cost_c
    decision = "ask" if e_ask > e_commit else "commit"
    return decision, e_commit, e_ask


# ---------------------------------------------------------------------------
# Example scenes
# ---------------------------------------------------------------------------

SCENES = [
    # Low ambiguity: utterance uniquely identifies target
    Scene(
        objects=[Object("red", "square", "large"),
                 Object("blue", "circle", "small"),
                 Object("green", "triangle", "medium")],
        target_idx=0,
        utterance="the red square"
    ),
    # Medium ambiguity: two red objects
    Scene(
        objects=[Object("red", "square", "large"),
                 Object("red", "circle", "small"),
                 Object("blue", "triangle", "medium")],
        target_idx=0,
        utterance="the red one"
    ),
    # High ambiguity: utterance matches all objects equally
    Scene(
        objects=[Object("red", "square", "large"),
                 Object("blue", "circle", "large"),
                 Object("green", "triangle", "large")],
        target_idx=2,
        utterance="the large object"
    ),
]

AMBIGUITY_LABELS = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CS 288 — Reference Game Demo")
    print("Cost-Aware Clarification via RSA + Expected Utility")
    print("=" * 60)

    cost_c = 0.25  # cost of asking a clarifying question

    all_confidences, all_corrects, all_bs = [], [], []

    for i, (scene, label) in enumerate(zip(SCENES, AMBIGUITY_LABELS)):
        print(f"\n--- Scene {i+1}: {label} Ambiguity ---")
        print(f"  Objects: {', '.join(str(o) for o in scene.objects)}")
        print(f"  Utterance: \"{scene.utterance}\"")
        print(f"  Target: {scene.target}")

        probs = simulate_model_probs(scene)
        print(f"\n  Model P(t | u):")
        for obj, p in zip(scene.objects, probs):
            marker = " <-- target" if obj is scene.target else ""
            print(f"    {str(obj):28s}  {p:.3f}{marker}")

        H = referential_entropy(probs)
        BS = brier_score(probs, scene.target_idx)
        decision, e_commit, e_ask = clarification_decision(probs, cost_c)
        correct = int(probs.index(max(probs)) == scene.target_idx)

        print(f"\n  Referential Entropy H(T|u) = {H:.3f}  (0=certain, ln(N)={math.log(len(probs)):.2f}=uniform)")
        print(f"  Brier Score               = {BS:.3f}")
        print(f"  E[U_commit] = {e_commit:.3f}   E[U_ask] = {e_ask:.3f}  (c={cost_c})")
        print(f"  Decision: {decision.upper()}  |  argmax prediction {'CORRECT' if correct else 'WRONG'}")

        all_confidences.append(max(probs))
        all_corrects.append(correct)
        all_bs.append(BS)

    print("\n" + "=" * 60)
    print("Aggregate Metrics (3-instance toy set)")
    print("=" * 60)
    print(f"  Mean Brier Score : {sum(all_bs)/len(all_bs):.3f}")
    ece = expected_calibration_error(all_confidences, all_corrects)
    print(f"  ECE              : {ece:.3f}")
    print(f"  Accuracy         : {sum(all_corrects)/len(all_corrects):.2f}")

    print("\nInterpretation:")
    print("  - Low entropy  → model is confident → likely COMMIT")
    print("  - High entropy → model is uncertain → likely ASK (if c is low)")
    print("  - ECE near 0   → confidence is well-calibrated → utility calc is trustworthy")
    print("  - High Brier   → probability mass is spread wrong → utility calc is unreliable")
    print()


if __name__ == "__main__":
    main()
