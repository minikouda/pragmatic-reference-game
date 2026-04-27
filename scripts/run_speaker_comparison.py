"""
Speaker comparison experiment.

Runs all speaker variants against the VLLMListener on a subset of scenes,
then prints a ranked summary table.

Speakers evaluated
------------------
Baselines (VLLM):
  vllm-naive       : target properties only, no distractor context
  vllm-pragmatic   : chain-of-thought over target properties only

New strategies (VLLM):
  scene-aware      : target + full distractor list, model picks minimal features
  scene-ranked     : scene-aware + explicit feature ranking step
  landmark-vllm    : rule landmark suggestion + LLM polish
  contrastive-vllm : foil-aware contrast + LLM polish

Rule-based (free, no API):
  ordinal          : superlatives / uniqueness ("the only triangle", "the leftmost circle")
  landmark         : pure rule-based spatial reference
  contrastive      : pure rule-based feature contrast

Usage
-----
python scripts/run_speaker_comparison.py \
    --jsonl data/scenes_6_force.jsonl \
    --scene_size 6 \
    --condition force \
    --n 30 \
    --model gemini-flash \
    --costs 0.1 0.25 \
    --workers 4 \
    --out results/speaker_comparison/

Writes results/speaker_comparison/experiment_records.jsonl and
results/speaker_comparison/experiment_summary.json (single unified files).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import load_jsonl
from src.refgame.speakers.vllm import VLLMSpeaker
from src.refgame.speakers.scene_aware import SceneAwareSpeaker
from src.refgame.speakers.landmark import LandmarkSpeaker, LandmarkVLLMSpeaker
from src.refgame.speakers.contrastive import ContrastiveSpeaker, ContrastiveVLLMSpeaker
from src.refgame.speakers.ordinal import OrdinalSpeaker
from src.refgame.listeners.vllm import VLLMListener
from src.refgame.eval.harness import run_grid
from src.refgame.eval.reporter import (
    DEFAULT_RECORDS_FILE,
    append_records, compute_summary, reset_records_file, write_summary,
)
from src.refgame.utils.llm_client import openrouter

VLLM_MODEL_PRESETS = {
    "gemini-flash":     "google/gemini-2.0-flash-001",
    "llama-4-scout":    "meta-llama/llama-4-scout",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "claude-haiku":     "anthropic/claude-haiku-4-5",
    "gpt-4o-mini":      "openai/gpt-4o-mini",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl",       type=str, required=True, help="Scene JSONL file")
    p.add_argument("--scene_size",  type=int, required=True,
                   help="Number of objects per scene (tag attached to every record).")
    p.add_argument("--condition",   type=str, required=True,
                   help="Per-dataset condition tag (e.g. 'none' or 'force' overlap).")
    p.add_argument("--n",           type=int, default=30,    help="Number of scenes to use")
    p.add_argument("--model",       type=str, default="gemini-flash")
    p.add_argument("--costs",       type=float, nargs="+", default=[0.1, 0.25])
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--out",         type=str, default="results/speaker_comparison/")
    p.add_argument("--no_rule",     action="store_true", help="Skip rule-based speakers")
    args = p.parse_args()

    model_id = VLLM_MODEL_PRESETS.get(args.model, args.model)
    client   = openrouter(model=model_id)

    scenes = load_jsonl(args.jsonl)[: args.n]
    stem   = Path(args.jsonl).stem
    meta   = {"scene_size": args.scene_size, "condition": args.condition}
    logging.info(
        f"Loaded {len(scenes)} scenes from {args.jsonl} (meta={meta})"
    )

    # ── Speakers ──────────────────────────────────────────────────────────────
    vllm_speakers = [
        VLLMSpeaker(client=client, pragmatic=False),          # naive baseline
        VLLMSpeaker(client=client, pragmatic=True),           # pragmatic baseline
        SceneAwareSpeaker(client=client, ranked=False),       # scene-aware
        SceneAwareSpeaker(client=client, ranked=True),        # scene-aware + ranking
        LandmarkVLLMSpeaker(client=client),                   # landmark + LLM
        ContrastiveVLLMSpeaker(client=client),                # contrastive + LLM
    ]

    rule_speakers = [] if args.no_rule else [
        OrdinalSpeaker(),
        LandmarkSpeaker(),
        ContrastiveSpeaker(),
    ]

    speakers = vllm_speakers + rule_speakers

    # ── Listener ──────────────────────────────────────────────────────────────
    listeners = [VLLMListener(client=client)]

    n_vllm_spk   = len(vllm_speakers)
    n_rule_spk   = len(rule_speakers)
    n_vllm_calls = len(scenes) * n_vllm_spk + len(scenes) * len(speakers)
    logging.info(
        f"Speakers: {n_vllm_spk} VLLM + {n_rule_spk} rule-based | "
        f"Est. VLLM calls: ~{n_vllm_calls} | Model: {model_id}"
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    records = run_grid(
        scenes=scenes,
        speakers=speakers,
        listeners=listeners,
        cost_values=args.costs,
        n_workers=args.workers,
        verbose=True,
        meta=meta,
    )

    out_dir      = Path(args.out)
    records_path = out_dir / DEFAULT_RECORDS_FILE
    reset_records_file(records_path)
    append_records(records, records_path)
    summary_path = write_summary(records, out_dir=out_dir)
    logging.info(f"Wrote {records_path} and {summary_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = compute_summary(records)

    # Group by cost_c, sort by CPA descending within each group
    from collections import defaultdict
    by_cost: dict[float, list] = defaultdict(list)
    for row in summary:
        by_cost[row["cost_c"]].append(row)

    for cost_c, rows in sorted(by_cost.items()):
        rows_sorted = sorted(rows, key=lambda r: -r.get("cpa", 0))
        print(f"\n{'─'*92}")
        print(f"  cost_c = {cost_c:.2f}   (ranked by CPA)")
        print(f"{'─'*92}")
        print(f"  {'Speaker':<38} {'CPA':>6}  {'Acc':>6}  {'Ask%':>6}  "
              f"{'HiConf%':>7}  {'PreCom%':>7}  {'Words':>5}  n")
        print(f"  {'─'*38} {'─'*6}  {'─'*6}  {'─'*6}  "
              f"{'─'*7}  {'─'*7}  {'─'*5}  {'─'*4}")
        for row in rows_sorted:
            words = row.get("utt_word_count_mean")
            words_s = f"{words:>5.1f}" if isinstance(words, (int, float)) else f"{'':>5}"
            print(
                f"  {row.get('speaker_type',''):<38} "
                f"{row.get('cpa', 0):>6.3f}  "
                f"{row.get('accuracy', 0):>6.3f}  "
                f"{row.get('clarification_rate', 0):>6.1%}  "
                f"{row.get('pct_high_conf', 0):>7.1%}  "
                f"{row.get('premature_commit_rate', 0):>7.1%}  "
                f"{words_s}  "
                f"{row.get('n', 0)}"
            )
    print(f"\nResults saved to {out_dir}/\n")


if __name__ == "__main__":
    main()
