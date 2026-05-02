"""
Final experiment: all VLLM speaker strategies × all listener variants × all scene datasets.

Speakers
--------
VLLM strategic (text properties + raw image, no bounding-box annotation):
  strategic-natural        : unconstrained — model describes as it sees fit
  strategic-contrastive    : find most-confusable distractor, highlight difference
  strategic-landmark       : describe relative to a distinctive nearby object
  strategic-superlative    : use extreme/uniqueness property (only, largest, leftmost)
  strategic-pragmatic      : RSA-style chain-of-thought, minimal discriminating expression
  strategic-scene_first    : inventory all objects, then identify target
  strategic-listener_aware : optimize for listener confidence

Rule-based baseline (LLM used only for clarification answers):
  feature-canonical        : minimal canonical feature description (rule-based);
                             answers clarification questions via LLM

Listeners
---------
Image-only (annotated image, no feature text — honest listeners):
  direct                   : direct probability array from indexed image
  cot                      : observe → score → assign from indexed image
  elimination              : visually rule out → assign from indexed image
  index                    : hard commit to single index (CPA == Accuracy)

Coordinate-based:
  vllm-listener(σ=10)      : predict (x,y), Gaussian posterior

Feature-based:
  feature-match            : VLM extracts features from utterance, scores by overlap

Text-assisted listeners (leaky — receive full object feature list) are excluded.

Datasets
--------
Runs over all 6 scene files: {6,8,10}_objects × {none,force}_overlap.

Usage
-----
python scripts/run_final_experiment.py \\
    --model gemini-flash \\
    --n 50 \\
    --costs 0.1 0.25 0.5 \\
    --workers 8 \\
    --out results/final_experiment/

    # Run a single dataset only
    --jsonl data/scenes_6_none.jsonl

    # Skip rule-based speakers
    --no_rule

    # Restrict listeners
    --listeners direct cot vllm
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
from src.refgame.speakers.strategic import StrategicVLLMSpeaker
from src.refgame.speakers.feature_canonical import FeatureCanonicalSpeaker
from src.refgame.listeners.vllm import VLLMListener
from src.refgame.listeners.feature_match import FeatureMatchListener
from src.refgame.listeners.direct_rank import (
    DirectRankListener, CoTRankListener,
    EliminationListener, IndexListener,
)
from src.refgame.listeners.dialogue import DialogueListener
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

ALL_LISTENER_CHOICES = [
    "direct", "cot", "elimination", "index",
    "vllm", "feature_match", "dialogue",
]

# Default datasets — relative to project root
DEFAULT_DATASETS = [
    ("data/scenes_6_none.jsonl",  6,  "none"),
    ("data/scenes_6_force.jsonl", 6,  "force"),
    ("data/scenes_8_none.jsonl",  8,  "none"),
    ("data/scenes_8_force.jsonl", 8,  "force"),
    ("data/scenes_10_none.jsonl", 10, "none"),
    ("data/scenes_10_force.jsonl",10, "force"),
]


def build_listeners(args, client) -> list:
    choices = set(args.listeners)
    listeners = []
    if "direct"      in choices: listeners.append(DirectRankListener(client=client))
    if "cot"         in choices: listeners.append(CoTRankListener(client=client))
    if "elimination" in choices: listeners.append(EliminationListener(client=client))
    if "index"       in choices: listeners.append(IndexListener(client=client))
    if "vllm"        in choices: listeners.append(VLLMListener(client=client, sigma=10.0))
    if "feature_match" in choices: listeners.append(FeatureMatchListener(client=client))
    if "dialogue"    in choices:
        # DialogueListener runs actual Q&A: listener asks, speaker answers
        # Uses FeatureCanonicalSpeaker for answering (rule-based + LLM fallback)
        speaker_for_dialogue = FeatureCanonicalSpeaker(client=client)
        listeners.append(DialogueListener(
            listener_client=client,
            speaker=speaker_for_dialogue,
            cost_c=min(args.costs),
            max_rounds=2,
        ))
    return listeners


def build_speakers(args, client) -> tuple[list, list]:
    """Returns (vllm_speakers, rule_speakers)."""
    vllm_speakers = [
        StrategicVLLMSpeaker(client=client, strategy="natural"),
        StrategicVLLMSpeaker(client=client, strategy="contrastive"),
        StrategicVLLMSpeaker(client=client, strategy="landmark"),
        StrategicVLLMSpeaker(client=client, strategy="superlative"),
        StrategicVLLMSpeaker(client=client, strategy="pragmatic"),
        StrategicVLLMSpeaker(client=client, strategy="scene_first"),
        StrategicVLLMSpeaker(client=client, strategy="listener_aware"),
    ]
    # FeatureCanonicalSpeaker: rule-based description + LLM for clarification answers
    rule_speakers = [] if args.no_rule else [
        FeatureCanonicalSpeaker(client=client),
    ]
    return vllm_speakers, rule_speakers


def print_summary_table(summary: list[dict], out_dir: Path) -> None:
    from collections import defaultdict
    by_cost: dict[float, list] = defaultdict(list)
    for row in summary:
        by_cost[row["cost_c"]].append(row)

    for cost_c, rows in sorted(by_cost.items()):
        rows_sorted = sorted(rows, key=lambda r: -r.get("cpa", 0))
        print(f"\n{'─'*110}")
        print(f"  cost_c = {cost_c:.2f}   (ranked by CPA)")
        print(f"{'─'*110}")
        print(f"  {'Speaker':<38} {'Listener':<24} {'CPA':>6}  {'Acc':>6}  "
              f"{'Ask%':>6}  {'Words':>5}  n")
        print(f"  {'─'*38} {'─'*24} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*4}")
        for row in rows_sorted:
            words = row.get("utt_word_count_mean")
            words_s = f"{words:>5.1f}" if isinstance(words, (int, float)) else f"{'':>5}"
            listener = row.get("listener_type", "")[:24]
            print(
                f"  {row.get('speaker_type',''):<38} "
                f"{listener:<24} "
                f"{row.get('cpa', 0):>6.3f}  "
                f"{row.get('accuracy', 0):>6.3f}  "
                f"{row.get('clarification_rate', 0):>6.1%}  "
                f"{words_s}  "
                f"{row.get('n', 0)}"
            )
    print(f"\nResults saved to {out_dir}/\n")


def run_dataset(jsonl: str, scene_size: int, condition: str, args, client,
                speakers: list, listeners: list, out_dir: Path) -> list:
    scenes = load_jsonl(jsonl)[: args.n]
    meta = {"scene_size": scene_size, "condition": condition}
    logging.info(
        f"Dataset: {jsonl} | scenes={len(scenes)} | "
        f"speakers={len(speakers)} | listeners={len(listeners)}"
    )
    return run_grid(
        scenes=scenes,
        speakers=speakers,
        listeners=listeners,
        cost_values=args.costs,
        n_workers=args.workers,
        verbose=True,
        meta=meta,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl",    type=str, default=None,
                   help="Run a single dataset instead of all six. "
                        "Must also pass --scene_size and --condition.")
    p.add_argument("--scene_size", type=int, default=None,
                   help="Required when --jsonl is set.")
    p.add_argument("--condition",  type=str, default=None,
                   help="Required when --jsonl is set (e.g. 'none' or 'force').")
    p.add_argument("--n",        type=int, default=100,
                   help="Max scenes per dataset (default: 100)")
    p.add_argument("--model",    type=str, default="gemini-flash",
                   help=f"Model preset or full ID. Presets: {list(VLLM_MODEL_PRESETS)}")
    p.add_argument("--costs",    type=float, nargs="+", default=[0.25])
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--out",      type=str, default="results/final_experiment/")
    p.add_argument("--no_rule",  action="store_true", help="Skip rule-based speakers")
    p.add_argument("--listeners", type=str, nargs="+", default=ALL_LISTENER_CHOICES,
                   choices=ALL_LISTENER_CHOICES,
                   help="Listener variants to include (default: all)")
    args = p.parse_args()

    model_id: str = VLLM_MODEL_PRESETS.get(args.model, args.model)
    client   = openrouter(model=model_id)
    out_dir  = Path(args.out)

    vllm_speakers, rule_speakers = build_speakers(args, client)
    speakers = vllm_speakers + rule_speakers
    listeners = build_listeners(args, client)

    logging.info(
        f"Model: {model_id} | "
        f"Speakers: {len(vllm_speakers)} VLLM + {len(rule_speakers)} rule-based | "
        f"Listeners: {len(listeners)} | Costs: {args.costs}"
    )

    # Determine datasets to run
    if args.jsonl:
        if args.scene_size is None or args.condition is None:
            p.error("--jsonl requires --scene_size and --condition")
        datasets = [(args.jsonl, args.scene_size, args.condition)]
    else:
        datasets = [(str(Path(j)), s, c) for j, s, c in DEFAULT_DATASETS if Path(j).exists()]
        missing  = [j for j, _, _ in DEFAULT_DATASETS if not Path(j).exists()]
        if missing:
            logging.warning(f"Skipping missing datasets: {missing}")

    # Unified output files accumulate records from all datasets
    records_path = out_dir / DEFAULT_RECORDS_FILE
    reset_records_file(records_path)
    all_records = []

    for jsonl, scene_size, condition in datasets:
        records = run_dataset(jsonl, scene_size, condition, args, client,
                              speakers, listeners, out_dir)
        append_records(records, records_path)
        all_records.extend(records)
        logging.info(f"  → {len(records)} records for {Path(jsonl).stem}")

    summary_path = write_summary(all_records, out_dir=out_dir)
    logging.info(f"Wrote {len(all_records)} total records → {records_path}")
    logging.info(f"Summary → {summary_path}")

    summary = compute_summary(all_records)
    print_summary_table(summary, out_dir)


if __name__ == "__main__":
    main()
