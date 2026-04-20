"""
Evaluation metrics for the reference game.

All functions are pure (no I/O) and operate on lists/arrays of scalars
so they can be used both online (per-scene) and offline (over result tables).

Metrics
-------
referential_entropy    : H(T | u) = -Σ p_i log p_i
brier_score            : (1/N) Σ (p_i - y_i)²
ece                    : Expected Calibration Error (binned)
cost_penalized_accuracy: Accuracy − c × ClarificationRate  (primary metric)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal


# ── Per-instance metrics ──────────────────────────────────────────────────────

def referential_entropy(posterior: list[float]) -> float:
    """H(T | u): 0 = certain, log(N) = maximally uniform."""
    return -sum(p * math.log(p) for p in posterior if p > 0)


def brier_score(posterior: list[float], target_idx: int) -> float:
    """Multi-class Brier Score ∈ [0, 2]. Lower is better."""
    n = len(posterior)
    return sum(
        (p - (1.0 if i == target_idx else 0.0)) ** 2
        for i, p in enumerate(posterior)
    ) / n


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: list[float],
    corrects:    list[bool],
    n_bins:      int = 10,
) -> float:
    """
    ECE: measures whether confidence ≈ accuracy.

    Bins instances by confidence, computes |avg_conf − avg_acc| per bin,
    weighted by bin size.  ECE = 0 is perfectly calibrated.
    """
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for conf, correct in zip(confidences, corrects):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))

    total = len(confidences)
    ece   = 0.0
    for b_items in bins.values():
        avg_conf = sum(x[0] for x in b_items) / len(b_items)
        avg_acc  = sum(x[1] for x in b_items) / len(b_items)
        ece += (len(b_items) / total) * abs(avg_conf - avg_acc)
    return ece


def cost_penalized_accuracy(
    corrects:            list[bool],
    actions:             list[Literal["commit", "ask"]],
    cost_c:              float,
) -> float:
    """
    Primary metric: Accuracy − c × ClarificationRate.

    Reward = (# correct commits) / N − c × (# asks) / N
    where N = total scenes.  Higher is better.
    """
    n = len(corrects)
    if n == 0:
        return 0.0
    n_correct = sum(
        1 for correct, action in zip(corrects, actions)
        if correct and action == "commit"
    )
    n_ask = sum(1 for a in actions if a == "ask")
    return n_correct / n - cost_c * (n_ask / n)


def clarification_rate(actions: list[Literal["commit", "ask"]]) -> float:
    """Fraction of scenes where the listener chose to ask."""
    if not actions:
        return 0.0
    return sum(1 for a in actions if a == "ask") / len(actions)


def commit_accuracy(
    corrects: list[bool],
    actions:  list[Literal["commit", "ask"]],
) -> float:
    """Accuracy only on scenes where the listener committed (no ask)."""
    commits = [(c, a) for c, a in zip(corrects, actions) if a == "commit"]
    if not commits:
        return float("nan")
    return sum(c for c, _ in commits) / len(commits)


# ── Full aggregate summary ────────────────────────────────────────────────────

def aggregate_metrics(
    corrects:     list[bool],
    actions:      list[Literal["commit", "ask"]],
    posteriors:   list[list[float]],
    target_idxs:  list[int],
    cost_c:       float,
    n_ece_bins:   int = 10,
) -> dict[str, float]:
    """
    Compute all metrics at once.  Returns a flat dict for easy logging.
    """
    confidences = [max(p) for p in posteriors]
    entropies   = [referential_entropy(p) for p in posteriors]
    briers      = [brier_score(p, t) for p, t in zip(posteriors, target_idxs)]

    return {
        "cpa":               cost_penalized_accuracy(corrects, actions, cost_c),
        "accuracy":          sum(corrects) / len(corrects) if corrects else 0.0,
        "commit_accuracy":   commit_accuracy(corrects, actions),
        "clarification_rate": clarification_rate(actions),
        "mean_entropy":      sum(entropies) / len(entropies) if entropies else 0.0,
        "mean_brier":        sum(briers) / len(briers) if briers else 0.0,
        "ece":               expected_calibration_error(confidences, corrects, n_ece_bins),
        "n":                 len(corrects),
        "cost_c":            cost_c,
    }
