"""
Script: run the full speaker × listener × cost grid evaluation.

Usage
-----
# Rule-based only (no API calls):
python scripts/run_eval.py --scenes data/scenes_test.jsonl --out results/

# With LLM speaker via OpenRouter:
export OPENROUTER_API_KEY=sk-or-...
python scripts/run_eval.py --scenes data/scenes_test.jsonl \
    --llm_model anthropic/claude-haiku-4-5 --workers 8 --out results/
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import load_jsonl
from src.refgame.speakers.literal import LiteralSpeaker
from src.refgame.speakers.rsa import RSASpeaker
from src.refgame.speakers.llm import LLMSpeaker
from src.refgame.listeners.literal import LiteralListener
from src.refgame.listeners.rsa import RSAListener
from src.refgame.eval.harness import run_grid
from src.refgame.eval.reporter import save_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes",    required=True,            help="Path to test scenes JSONL")
    p.add_argument("--out",       default="results/",       help="Output directory")
    p.add_argument("--costs",     nargs="+", type=float,    default=[0.1, 0.25, 0.5])
    p.add_argument("--alpha",     type=float, default=4.0,  help="RSA rationality")
    p.add_argument("--llm_model", type=str,   default=None, help="OpenRouter model for LLM speaker")
    p.add_argument("--workers",   type=int,   default=1,    help="Thread pool size for LLM calls")
    p.add_argument("--prefix",    type=str,   default="eval")
    return p.parse_args()


def main():
    args = parse_args()

    scenes = load_jsonl(args.scenes)
    logging.info(f"Loaded {len(scenes)} scenes from {args.scenes}")

    # ── Speakers ──────────────────────────────────────────────────────────────
    speakers = [
        LiteralSpeaker(),
        RSASpeaker(alpha=args.alpha),
    ]
    if args.llm_model:
        from src.refgame.utils.llm_client import openrouter
        client = openrouter(model=args.llm_model)
        speakers.append(LLMSpeaker(client=client, pragmatic=False))
        speakers.append(LLMSpeaker(client=client, pragmatic=True))

    # ── Listeners ─────────────────────────────────────────────────────────────
    listeners = [
        LiteralListener(),
        RSAListener(alpha=args.alpha),
    ]

    # ── Run ───────────────────────────────────────────────────────────────────
    records = run_grid(
        scenes=scenes,
        speakers=speakers,
        listeners=listeners,
        cost_values=args.costs,
        n_workers=args.workers,
        verbose=True,
    )

    logging.info(f"Collected {len(records)} evaluation records")
    save_results(records, out_dir=args.out, prefix=args.prefix)


if __name__ == "__main__":
    main()
