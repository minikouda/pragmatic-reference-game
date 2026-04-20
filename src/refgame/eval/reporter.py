"""
Result reporting and aggregation.

Produces per-condition aggregate tables and per-ambiguity-tier breakdowns
suitable for direct inclusion in LaTeX tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.schema import EvalRecord
from ..metrics.core import aggregate_metrics


def summarize(
    records:    list[EvalRecord],
    group_keys: list[str] = ("speaker_type", "listener_type", "cost_c"),
) -> list[dict[str, Any]]:
    """
    Aggregate EvalRecord list by `group_keys`.

    Returns a list of dicts, each containing the group key values plus
    all metrics from `aggregate_metrics`.
    """
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
    records: list[EvalRecord],
    group_keys: list[str] = ("speaker_type", "listener_type", "cost_c"),
) -> list[dict]:
    """Summarize broken down by ambiguity_tier × group_keys."""
    return summarize(records, group_keys=group_keys + ["ambiguity_tier"])


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
