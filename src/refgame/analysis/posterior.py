"""
Posterior analysis utilities for VLLMListener experiments.

Functions
---------
posterior_sharpness(records)
    Per-record max(posterior) statistics; identifies flat-posterior failure mode.

failure_breakdown(records)
    Categorizes listener failures: coord_miss, parse_fail, correct.

speaker_comparison(records, cost_c)
    Accuracy / ask-rate / eu_commit table grouped by speaker.

tier_breakdown(records, cost_c)
    Performance split by ambiguity_tier (low / medium / high).

simulate_kernel_sharpness(n_objects, n_trials, sigma_values)
    Simulates how much sharper Gaussian vs inverse-distance posteriors are,
    given typical coordinate-prediction errors.  Returns a dict of stats.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── Record loading ─────────────────────────────────────────────────────────────

def load_records(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ── Posterior sharpness ────────────────────────────────────────────────────────

def posterior_sharpness(records: list[dict]) -> dict[str, Any]:
    """
    Analyse the distribution of eu_commit (= max posterior) across records.

    Returns a dict with:
      mean, median, stdev, pct_ge_50, pct_ge_75, pct_ge_90,
      histogram (bucket → count)
    """
    values = [r["eu_commit"] for r in records if "eu_commit" in r]
    if not values:
        return {}

    buckets = [0.0, 0.3, 0.5, 0.6, 0.75, 0.9, 1.001]
    hist = {}
    for lo, hi in zip(buckets, buckets[1:]):
        label = f"[{lo:.2f},{hi:.2f})"
        hist[label] = sum(1 for v in values if lo <= v < hi)

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "pct_ge_50": sum(1 for v in values if v >= 0.50) / len(values),
        "pct_ge_75": sum(1 for v in values if v >= 0.75) / len(values),
        "pct_ge_90": sum(1 for v in values if v >= 0.90) / len(values),
        "histogram": hist,
    }


# ── Failure breakdown ──────────────────────────────────────────────────────────

def failure_breakdown(records: list[dict]) -> dict[str, Any]:
    """
    Categorize listener outcomes.

    Returns counts for: correct, wrong_coord (snapped to wrong object),
    ask (clarification triggered), by (speaker_type, listener_type) pair.
    """
    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for r in records:
        key = (r.get("speaker_type", "?"), r.get("listener_type", "?"))
        groups[key]["total"] += 1
        if r.get("action") == "ask":
            groups[key]["ask"] += 1
        elif r.get("correct"):
            groups[key]["correct"] += 1
        else:
            groups[key]["wrong_coord"] += 1

    result = {}
    for (sp, li), counts in groups.items():
        n = counts["total"]
        result[f"{sp} | {li}"] = {
            "n": n,
            "correct_pct": counts["correct"] / n,
            "wrong_coord_pct": counts["wrong_coord"] / n,
            "ask_pct": counts["ask"] / n,
        }
    return result


# ── Speaker comparison ─────────────────────────────────────────────────────────

def speaker_comparison(records: list[dict], cost_c: float = 0.5) -> list[dict]:
    """
    Group records by speaker_type at a fixed cost_c.
    Returns list of dicts sorted descending by CPA.
    """
    filtered = [r for r in records if abs(r.get("cost_c", -1) - cost_c) < 1e-6]
    groups: dict[str, list] = defaultdict(list)
    for r in filtered:
        groups[r.get("speaker_type", "?")].append(r)

    rows = []
    for sp, rs in groups.items():
        commits = [r for r in rs if r.get("action") != "ask"]
        asks = [r for r in rs if r.get("action") == "ask"]
        acc = sum(r["correct"] for r in commits) / len(rs) if rs else 0
        ask_rate = len(asks) / len(rs) if rs else 0
        cpa = acc - cost_c * ask_rate
        mean_eu = statistics.mean(r["eu_commit"] for r in rs) if rs else 0
        rows.append({
            "speaker": sp,
            "n": len(rs),
            "accuracy": round(acc, 3),
            "ask_rate": round(ask_rate, 3),
            "cpa": round(cpa, 3),
            "mean_eu_commit": round(mean_eu, 3),
        })

    return sorted(rows, key=lambda x: -x["cpa"])


# ── Tier breakdown ─────────────────────────────────────────────────────────────

def tier_breakdown(records: list[dict], cost_c: float = 0.5) -> dict[str, dict]:
    """
    Split performance by ambiguity_tier at a fixed cost.
    """
    filtered = [r for r in records if abs(r.get("cost_c", -1) - cost_c) < 1e-6]
    tiers: dict[str, list] = defaultdict(list)
    for r in filtered:
        tiers[r.get("ambiguity_tier", "unknown")].append(r)

    result = {}
    for tier, rs in sorted(tiers.items()):
        commits = [r for r in rs if r.get("action") != "ask"]
        acc = sum(r["correct"] for r in commits) / len(rs) if rs else 0
        ask_rate = sum(1 for r in rs if r.get("action") == "ask") / len(rs) if rs else 0
        cpa = acc - cost_c * ask_rate
        result[tier] = {
            "n": len(rs),
            "accuracy": round(acc, 3),
            "ask_rate": round(ask_rate, 3),
            "cpa": round(cpa, 3),
        }
    return result


# ── Kernel sharpness simulation ────────────────────────────────────────────────

def simulate_kernel_sharpness(
    n_objects: int = 6,
    n_trials: int = 2000,
    sigma_values: list[float] | None = None,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Monte Carlo comparison of posterior kernels.

    Generates random scenes: target object at distance 2-10 from predicted point,
    distractors at 15-60.  Computes max(posterior) under each kernel.

    Returns dict kernel_name → {mean_max, pct_ge_50, pct_ge_75, pct_ge_90}.
    """
    import random
    rng = random.Random(seed)

    if sigma_values is None:
        sigma_values = [5.0, 10.0, 15.0, 20.0]

    def inv_dist(dists):
        scores = [1.0 / (d + 1e-3) for d in dists]
        total = sum(scores)
        return [s / total for s in scores]

    def gaussian(dists, sigma):
        scores = [math.exp(-d * d / (2 * sigma * sigma)) for d in dists]
        total = sum(scores)
        if total < 1e-12:
            return [1.0 / len(dists)] * len(dists)
        return [s / total for s in scores]

    samples: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_trials):
        target_d = rng.uniform(2, 10)
        other_ds = [rng.uniform(15, 60) for _ in range(n_objects - 1)]
        dists = [target_d] + other_ds

        samples["inv_dist"].append(max(inv_dist(dists)))
        for sigma in sigma_values:
            samples[f"gaussian_σ{sigma:.0f}"].append(max(gaussian(dists, sigma)))

    result = {}
    for kernel, maxes in samples.items():
        result[kernel] = {
            "mean_max": round(statistics.mean(maxes), 3),
            "pct_ge_50": round(sum(1 for x in maxes if x >= 0.50) / len(maxes), 3),
            "pct_ge_75": round(sum(1 for x in maxes if x >= 0.75) / len(maxes), 3),
            "pct_ge_90": round(sum(1 for x in maxes if x >= 0.90) / len(maxes), 3),
        }
    return result


# ── Speaker utterance analysis ─────────────────────────────────────────────────

def utterance_info_content(records: list[dict]) -> dict[str, dict]:
    """
    Estimate how informative speaker utterances are by measuring listener
    entropy conditioned on each speaker type.

    Returns dict speaker → {mean_entropy, mean_eu_commit, accuracy}.
    """
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        sp = r.get("speaker_type", "?")
        groups[sp].append(r)

    result = {}
    for sp, rs in groups.items():
        ents = [r.get("entropy", 0) for r in rs]
        eus = [r.get("eu_commit", 0) for r in rs]
        commits = [r for r in rs if r.get("action") != "ask"]
        acc = sum(r["correct"] for r in commits) / len(rs) if rs else 0
        result[sp] = {
            "n": len(rs),
            "mean_entropy": round(statistics.mean(ents), 4),
            "mean_eu_commit": round(statistics.mean(eus), 4),
            "accuracy": round(acc, 3),
        }
    return dict(sorted(result.items(), key=lambda x: -x[1]["accuracy"]))
