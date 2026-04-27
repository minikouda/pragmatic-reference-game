"""
Result reporting and aggregation.

Two output modes are supported:

1. Legacy per-dataset files (``save_results``):
   ``{prefix}_records.jsonl``, ``{prefix}_summary.json``, ``{prefix}_by_tier.json``.
   Kept for backward compatibility with existing analysis/plotting scripts.

2. Unified single-file output (``append_records`` + ``write_summary``):
   ``experiment_records.jsonl`` (all trials across all configurations) and
   ``experiment_summary.json`` (one rich aggregation grouped by speaker /
   listener / cost_c / scene_size / condition with by-tier and by-region
   breakdowns and the advanced confidence/error/utterance metrics).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..data.schema import EvalRecord
from ..metrics.core import aggregate_metrics


_DEFAULT_GROUP_KEYS = ["speaker_type", "listener_type", "cost_c"]

# Confidence bucket used by premature-commit / over-caution / pct_high_conf.
# This is independent of the cost_c sweep — see compute_summary docstring.
HIGH_CONF_THRESHOLD = 0.75


def summarize(
    records:    list[EvalRecord],
    group_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Aggregate EvalRecord list by `group_keys`.

    Returns a list of dicts, each containing the group key values plus
    all metrics from `aggregate_metrics`.
    """
    if group_keys is None:
        group_keys = _DEFAULT_GROUP_KEYS
    # Group records
    groups: dict[tuple, list[EvalRecord]] = {}
    for r in records:
        key = tuple(getattr(r, k) for k in group_keys)
        groups.setdefault(key, []).append(r)

    rows: list[dict] = []
    for key_vals, group in groups.items():
        corrects    = [r.correct for r in group]
        actions     = [r.action for r in group]   # type: ignore[arg-type]
        posteriors  = [[r.eu_commit] for r in group]   # proxy: single-value posteriors not stored
        target_idxs = [r.target_idx for r in group]
        cost_c      = group[0].cost_c

        # Reconstruct a minimal 1-element posterior for brier/entropy proxies
        # (full posteriors are not stored in EvalRecord to keep it lightweight)
        dummy_posteriors = [
            [r.eu_commit] + [(1 - r.eu_commit) / max(1, r.target_idx)]
            for r in group
        ]

        row = dict(zip(group_keys, key_vals))
        row.update({
            "cpa":                round(sum(
                1 for r in group if r.correct and r.action == "commit"
            ) / len(group) - cost_c * sum(
                1 for r in group if r.action == "ask"
            ) / len(group), 4),
            "accuracy":           round(sum(corrects) / len(corrects), 4),
            "clarification_rate": round(sum(1 for a in actions if a == "ask") / len(actions), 4),
            "mean_entropy":       round(sum(r.entropy for r in group) / len(group), 4),
            "mean_brier":         round(sum(r.brier_score for r in group) / len(group), 4),
            "n":                  len(group),
        })
        rows.append(row)

    return sorted(rows, key=lambda r: (-r.get("cpa", 0),))


def summarize_by_tier(
    records:    list[EvalRecord],
    group_keys: list[str] | None = None,
) -> list[dict]:
    """Summarize broken down by ambiguity_tier × group_keys."""
    keys = list(group_keys) if group_keys else _DEFAULT_GROUP_KEYS
    return summarize(records, group_keys=keys + ["ambiguity_tier"])


def to_latex_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render aggregated rows as a LaTeX booktabs table."""
    if not rows:
        return ""
    cols = columns or list(rows[0].keys())
    header = " & ".join(cols) + r" \\"
    lines  = [r"\begin{tabular}{" + "l" * len(cols) + "}",
               r"\toprule", header, r"\midrule"]
    for row in rows:
        cells = " & ".join(str(row.get(c, "")) for c in cols)
        lines.append(cells + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def save_results(
    records: list[EvalRecord],
    out_dir: str | Path,
    prefix:  str = "eval",
) -> None:
    """
    Save results in three formats:
      - {prefix}_records.jsonl   : full record list (one JSON per line)
      - {prefix}_summary.json    : aggregated metrics table
      - {prefix}_by_tier.json    : tier-broken-down summary
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw records
    with (out_dir / f"{prefix}_records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__) + "\n")

    # Summaries
    summary = summarize(records)
    with (out_dir / f"{prefix}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    by_tier = summarize_by_tier(records)
    with (out_dir / f"{prefix}_by_tier.json").open("w") as f:
        json.dump(by_tier, f, indent=2)

    print(f"Results saved to {out_dir}/")


# ── Unified single-file output ───────────────────────────────────────────────

DEFAULT_RECORDS_FILE = "experiment_records.jsonl"
DEFAULT_SUMMARY_FILE = "experiment_summary.json"

UNIFIED_GROUP_KEYS = [
    "speaker_type", "listener_type", "cost_c", "scene_size", "condition",
]

# Expected breakdown buckets — listed explicitly so the summary always emits
# the same shape (with n=0 for missing slices) and downstream code can rely
# on the keys.
_TIER_BUCKETS   = ("low", "medium", "high")
_REGION_BUCKETS = ("corner", "edge", "center")


def append_records(
    records:  Iterable[EvalRecord],
    out_path: str | Path,
) -> int:
    """
    Append records to the unified JSONL file.

    Streams one JSON object per line so per-dataset runs can flush as they
    complete (no need to hold every record in memory across the whole sweep).
    Returns the number of records written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("a") as f:
        for r in records:
            f.write(json.dumps(r.__dict__) + "\n")
            n += 1
    return n


def reset_records_file(out_path: str | Path) -> None:
    """Delete the unified records file if present (call before a fresh run)."""
    p = Path(out_path)
    if p.exists():
        p.unlink()


def compute_summary(
    records:    list[EvalRecord],
    group_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Rich aggregation grouped by ``group_keys``.

    For each group emits the standard accuracy / CPA / clarification rate
    plus the advanced metrics requested by the manual-analysis review:

      * eu_commit distribution stats: mean, median, std, pct_high_conf
        (fraction with eu_commit > 0.75). Used to surface bimodal vs.
        flat / miscalibrated listeners.
      * Premature-commit rate: eu_commit > 0.75 AND not correct.
        Independent of action — uses the confidence bucket directly so the
        metric is comparable across cost_c values.
      * Over-caution rate: eu_commit < 0.75 AND top-1 prediction == target.
        Same independence rationale.
      * Mean Manhattan error on incorrect trials, on the 3x3 spatial grid.
      * Utterance verbosity: word-count mean and median.
      * by_tier:   accuracy + ask_rate broken down by ambiguity_tier.
      * by_region: accuracy + ask_rate broken down by target_region
        (corner / edge / center).

    Returns a list of dicts sorted by descending CPA.
    """
    if group_keys is None:
        group_keys = UNIFIED_GROUP_KEYS

    groups: dict[tuple, list[EvalRecord]] = defaultdict(list)
    for r in records:
        key = tuple(getattr(r, k, None) for k in group_keys)
        groups[key].append(r)

    rows: list[dict] = []
    for key_vals, group in groups.items():
        row = dict(zip(group_keys, key_vals))
        row.update(_aggregate_group(group))
        rows.append(row)

    return sorted(rows, key=lambda r: -r.get("cpa", 0))


def write_summary(
    records:      list[EvalRecord],
    out_dir:      str | Path,
    summary_file: str = DEFAULT_SUMMARY_FILE,
) -> Path:
    """Aggregate ``records`` and write the unified summary JSON. Returns the path."""
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / summary_file
    summary  = compute_summary(records)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return out_path


# ── Aggregation helpers ──────────────────────────────────────────────────────

def _aggregate_group(records: list[EvalRecord]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}

    cost_c          = records[0].cost_c
    n_correct       = sum(1 for r in records if r.correct)
    n_ask           = sum(1 for r in records if r.action == "ask")
    n_correct_commit = sum(1 for r in records if r.correct and r.action == "commit")

    eu_commits      = [r.eu_commit for r in records]
    n_high_conf     = sum(1 for v in eu_commits if v > HIGH_CONF_THRESHOLD)

    # Premature commitment: high confidence but the top-1 was wrong.
    # We use predicted_idx (argmax of posterior) rather than action == "commit"
    # so the metric is comparable across cost_c values.
    n_premature = sum(
        1 for r in records
        if r.eu_commit > HIGH_CONF_THRESHOLD and not r.correct
    )
    # Over-caution: low confidence yet top-1 happened to be correct.
    n_over_caution = sum(
        1 for r in records
        if r.eu_commit < HIGH_CONF_THRESHOLD and r.predicted_idx == r.target_idx
    )

    wrong_with_dist = [
        r.manhattan_error for r in records
        if not r.correct and r.manhattan_error is not None
    ]
    word_counts = [
        r.utterance_word_count for r in records
        if r.utterance_word_count is not None
    ]

    return {
        "n":                   n,
        "cpa":                 _round(n_correct_commit / n - cost_c * n_ask / n),
        "accuracy":            _round(n_correct / n),
        "clarification_rate":  _round(n_ask / n),
        "mean_entropy":        _round(sum(r.entropy for r in records) / n),
        "mean_brier":          _round(sum(r.brier_score for r in records) / n),

        "eu_commit_mean":      _round(sum(eu_commits) / n),
        "eu_commit_median":    _round(_median(eu_commits)),
        "eu_commit_std":       _round(_std(eu_commits)),
        "pct_high_conf":       _round(n_high_conf / n),

        "premature_commit_rate": _round(n_premature / n),
        "over_caution_rate":     _round(n_over_caution / n),

        "mean_manhattan_error_on_wrong":
            _round(sum(wrong_with_dist) / len(wrong_with_dist))
            if wrong_with_dist else None,
        "n_wrong_with_dist":   len(wrong_with_dist),

        "utt_word_count_mean":   _round(sum(word_counts) / len(word_counts), 2)
                                  if word_counts else None,
        "utt_word_count_median": _round(_median(word_counts), 2)
                                  if word_counts else None,

        "by_tier":   _breakdown_by(records, "ambiguity_tier", _TIER_BUCKETS),
        "by_region": _breakdown_by(records, "target_region",  _REGION_BUCKETS),
    }


def _breakdown_by(
    records: list[EvalRecord],
    attr:    str,
    buckets: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Sub-aggregate accuracy + ask_rate per bucket, always emitting all keys."""
    out: dict[str, dict[str, Any]] = {}
    for b in buckets:
        sub = [r for r in records if getattr(r, attr, None) == b]
        if not sub:
            out[b] = {"n": 0, "accuracy": None, "ask_rate": None}
            continue
        n = len(sub)
        out[b] = {
            "n":        n,
            "accuracy": _round(sum(1 for r in sub if r.correct) / n),
            "ask_rate": _round(sum(1 for r in sub if r.action == "ask") / n),
        }
    return out


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def _std(vals: list[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / n)


def _round(x: float, ndigits: int = 4) -> float:
    return round(x, ndigits)
