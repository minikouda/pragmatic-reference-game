"""
Demo: DialogueListener on a small slice of scenes.

Tests the full multi-turn loop:
  speaker → utterance
  listener → posterior
  if uncertain: listener asks → speaker answers → listener updates posterior
  commit to argmax

Usage:
  python scripts/run_dialogue_demo.py --n 20 --cost 0.5 --rounds 2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import load_jsonl
from src.refgame.speakers.strategic import StrategicVLLMSpeaker
from src.refgame.speakers.feature_canonical import FeatureCanonicalSpeaker
from src.refgame.listeners.dialogue import DialogueListener
from src.refgame.listeners.direct_rank import ImageOnlyDirectRankListener
from src.refgame.utils.llm_client import openrouter

MODEL = "google/gemini-2.0-flash-001"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl",   default="data/scenes_6_force.jsonl")
    p.add_argument("--n",       type=int,   default=20)
    p.add_argument("--cost",    type=float, default=0.5)
    p.add_argument("--rounds",  type=int,   default=2)
    p.add_argument("--strategy", default="listener_aware")
    args = p.parse_args()

    client = openrouter(model=MODEL)

    scenes  = load_jsonl(args.jsonl)[: args.n]
    speaker = StrategicVLLMSpeaker(client=client, strategy=args.strategy)

    # Baseline: io-direct (no dialogue)
    baseline = ImageOnlyDirectRankListener(client=client)

    # Dialogue listener — speaker answers questions via its own visual reasoning
    dialogue = DialogueListener(
        listener_client=client,
        speaker=speaker,
        cost_c=args.cost,
        max_rounds=args.rounds,
    )

    results = {"baseline": [], "dialogue": []}

    for i, scene in enumerate(scenes):
        target_idx = scene.target_idx
        utterance = speaker.speak(scene, target_idx)
        logging.info(f"[{i+1}/{args.n}] scene={scene.id}  utt='{utterance.text}'")

        # Baseline
        b_out = baseline.listen(scene, utterance)
        b_correct = (b_out.predicted_idx == target_idx)
        b_conf    = max(b_out.posterior)
        b_ask     = b_conf < (1.0 - args.cost)
        results["baseline"].append({"correct": b_correct, "ask": b_ask, "conf": b_conf})

        # Dialogue
        d_out = dialogue.listen(scene, utterance)
        d_correct = (d_out.predicted_idx == target_idx)
        d_rounds  = d_out.listener_meta["rounds"]
        d_conf    = d_out.listener_meta["final_max_p"]
        results["dialogue"].append({"correct": d_correct, "rounds": d_rounds, "conf": d_conf})

        if d_rounds > 0:
            for j, (q, a) in enumerate(d_out.listener_meta["qa_history"], 1):
                logging.info(f"    turn {j}  Q: {q}")
                logging.info(f"    turn {j}  A: {a}")
        logging.info(
            f"  baseline: pred={b_out.predicted_idx} correct={b_correct} conf={b_conf:.2f} ask={b_ask}"
            f"  |  dialogue: pred={d_out.predicted_idx} correct={d_correct} "
            f"conf={d_conf:.2f} rounds={d_rounds}"
        )

    n = len(scenes)
    b_acc  = sum(r["correct"] for r in results["baseline"]) / n
    b_ask  = sum(r["ask"]     for r in results["baseline"]) / n
    b_cpa  = b_acc - args.cost * b_ask

    d_acc  = sum(r["correct"] for r in results["dialogue"]) / n
    d_ask  = sum(r["rounds"] > 0 for r in results["dialogue"]) / n
    d_cpa  = d_acc - args.cost * d_ask  # each round = 1 clarification cost

    print(f"\n{'─'*60}")
    print(f"  n={n}  cost={args.cost}  max_rounds={args.rounds}")
    print(f"{'─'*60}")
    print(f"  {'Listener':<20} {'Acc':>6}  {'Ask%':>6}  {'CPA':>6}")
    print(f"  {'─'*20} {'─'*6}  {'─'*6}  {'─'*6}")
    print(f"  {'io-direct (no Q)':20} {b_acc:>6.3f}  {b_ask:>6.1%}  {b_cpa:>6.3f}")
    print(f"  {'dialogue':20} {d_acc:>6.3f}  {d_ask:>6.1%}  {d_cpa:>6.3f}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
