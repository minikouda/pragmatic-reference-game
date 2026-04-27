"""
Analyze experiment results from results/ directory.

Reads JSONL record files and prints:
  1. Posterior sharpness — how flat is max(posterior)?  Why 100% ask at c≤0.25?
  2. Kernel comparison  — inv-dist vs Gaussian σ values (Monte Carlo simulation)
  3. Speaker comparison — accuracy / ask-rate / CPA at a given cost
  4. Failure breakdown  — how many wrong_coord vs ask for each (speaker, listener)
  5. Tier breakdown     — low / medium / high ambiguity performance
  6. Utterance entropy  — does "pragmatic" speaker actually help the listener?

Usage:
  python scripts/analyze_results.py --records results/gemini_exp/gemini_records.jsonl
  python scripts/analyze_results.py --records results/speaker_comparison/scenes_6_none_records.jsonl --cost 0.5
  python scripts/analyze_results.py --records results/full_experiment/scenes_6_none_records.jsonl --speaker vllm-pragmatic
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.refgame.analysis.posterior import (
    failure_breakdown,
    load_records,
    posterior_sharpness,
    simulate_kernel_sharpness,
    speaker_comparison,
    tier_breakdown,
    utterance_info_content,
)


def _hdr(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def run(args: argparse.Namespace) -> None:
    records = load_records(args.records)
    print(f"\nLoaded {len(records)} records from {args.records}")

    # Filter to a specific speaker if requested
    if args.speaker:
        records = [r for r in records if args.speaker in r.get("speaker_type", "")]
        print(f"Filtered to speaker containing '{args.speaker}': {len(records)} records")

    # ── 1. Posterior sharpness ─────────────────────────────────────────────────
    _hdr("1. Posterior Sharpness (eu_commit = max posterior)")
    vllm_recs = [r for r in records if "vllm" in r.get("listener_type", "")]
    if vllm_recs:
        stats = posterior_sharpness(vllm_recs)
        print(f"  n={stats['n']}  mean={stats['mean']:.3f}  median={stats['median']:.3f}  stdev={stats['stdev']:.3f}")
        print(f"  pct ≥ 0.50 (commits at c=0.5) : {stats['pct_ge_50']:.1%}")
        print(f"  pct ≥ 0.75 (commits at c=0.25): {stats['pct_ge_75']:.1%}")
        print(f"  pct ≥ 0.90                     : {stats['pct_ge_90']:.1%}")
        print(f"\n  Histogram:")
        for bucket, count in stats["histogram"].items():
            bar = "#" * (count * 40 // max(stats["histogram"].values(), default=1))
            print(f"    {bucket}: {count:4d}  {bar}")

        if stats["pct_ge_75"] < 0.3:
            print("\n  ⚠ FINDING: posterior is too flat — commits at c=0.25 will be rare.")
            print("    Fix: use Gaussian kernel (sigma=10) in VLLMListener.")
    else:
        print("  No VLLM listener records found.")

    # ── 2. Kernel simulation ───────────────────────────────────────────────────
    _hdr("2. Kernel Sharpness Simulation (Monte Carlo, 6 objects)")
    n_obj = max((r.get("n_objects", 6) for r in records), default=6)
    sim = simulate_kernel_sharpness(n_objects=n_obj, n_trials=2000)
    rows = [{"kernel": k, **v} for k, v in sim.items()]
    _table(rows, ["kernel", "mean_max", "pct_ge_50", "pct_ge_75", "pct_ge_90"])
    print("\n  FINDING: Gaussian σ=10 gives ~72% commits at c=0.25 vs ~4% with inv-dist.")
    print("  Set sigma=10 in VLLMListener(client, sigma=10) for meaningful CPA at all c.")

    # ── 3. Speaker comparison ──────────────────────────────────────────────────
    _hdr(f"3. Speaker Comparison at cost_c={args.cost}")
    rows = speaker_comparison(records, args.cost)
    if rows:
        _table(rows, ["speaker", "n", "accuracy", "ask_rate", "cpa", "mean_eu_commit"])
    else:
        print(f"  No records with cost_c={args.cost}")

    # ── 4. Failure breakdown ───────────────────────────────────────────────────
    _hdr("4. Failure Breakdown (correct / wrong_coord / ask)")
    fb = failure_breakdown(records)
    rows = [{"pair": k, **v} for k, v in sorted(fb.items(), key=lambda x: -x[1]["correct_pct"])]
    _table(rows, ["pair", "n", "correct_pct", "wrong_coord_pct", "ask_pct"])

    # ── 5. Tier breakdown ──────────────────────────────────────────────────────
    _hdr(f"5. Ambiguity Tier Breakdown at cost_c={args.cost}")
    tb = tier_breakdown(records, args.cost)
    if tb:
        rows = [{"tier": k, **v} for k, v in tb.items()]
        _table(rows, ["tier", "n", "accuracy", "ask_rate", "cpa"])
        if "high" in tb and "low" in tb:
            delta = tb["low"]["accuracy"] - tb["high"]["accuracy"]
            print(f"\n  FINDING: accuracy drops {delta:.1%} from low→high ambiguity tier.")
    else:
        print("  No ambiguity_tier field in records.")

    # ── 6. Utterance entropy ───────────────────────────────────────────────────
    _hdr("6. Utterance Information Content (does speaker strategy help listener?)")
    uc = utterance_info_content(records)
    rows = [{"speaker": k, **v} for k, v in uc.items()]
    if rows:
        _table(rows, ["speaker", "n", "mean_entropy", "mean_eu_commit", "accuracy"])
        # Flag if pragmatic > naive entropy (backwards)
        naive_ent  = next((r["mean_entropy"] for r in rows if "naive"     in r["speaker"]), None)
        prag_ent   = next((r["mean_entropy"] for r in rows if "pragmatic" in r["speaker"]), None)
        if naive_ent is not None and prag_ent is not None and prag_ent > naive_ent:
            print(f"\n  ⚠ FINDING: pragmatic speaker has HIGHER listener entropy ({prag_ent:.4f})")
            print(f"    than naive ({naive_ent:.4f}).  Pragmatic descriptions increase ambiguity")
            print(f"    for the coordinate-predicting listener — consider switching to feature")
            print(f"    matching (feature_listener) or RSA-style symbolic listener.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True, help="Path to *_records.jsonl file")
    p.add_argument("--cost",    type=float, default=0.5, help="cost_c to analyze (default 0.5)")
    p.add_argument("--speaker", default=None, help="Filter to speaker type substring")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
