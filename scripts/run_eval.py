"""
Script: run the full speaker × listener × cost grid evaluation.

Usage
-----
# VLLM only (requires OPENROUTER_API_KEY):
export OPENROUTER_API_KEY=sk-or-...
python scripts/run_eval.py \
    --dataset reference_game_dataset/dataset.json \
    --vllm_model qwen2-vl-72b \
    --workers 8 \
    --out results/

# List available built-in model aliases:
python scripts/run_eval.py --list_vllm_models --dataset reference_game_dataset/dataset.json

# Include symbolic baselines (literal + RSA, no API key needed):
python scripts/run_eval.py \
    --dataset reference_game_dataset/dataset.json \
    --vllm_model llava-1.6-7b \
    --symbolic_baselines \
    --workers 8 \
    --out results/

# Custom JSONL dataset (e.g. generated scenes):
python scripts/run_eval.py \
    --jsonl data/scenes_test.jsonl \
    --vllm_model qwen/qwen2-vl-72b-instruct \
    --out results/
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import (
    load_image_dataset, load_jsonl, dataset_stats, split_dataset,
)
from src.refgame.speakers.literal import LiteralSpeaker
from src.refgame.speakers.rsa import RSASpeaker
from src.refgame.speakers.vllm import VLLMSpeaker
from src.refgame.listeners.literal import LiteralListener
from src.refgame.listeners.rsa import RSAListener
from src.refgame.listeners.vllm import VLLMListener
from src.refgame.eval.harness import run_grid
from src.refgame.eval.reporter import save_results, summarize


VLLM_MODEL_PRESETS = {
    # Open-source vision-language models commonly served with vLLM.
    "llava-1.6-7b": "liuhaotian/llava-1.6-7b",
    "llava-1.6-13b": "liuhaotian/llava-1.6-13b",
    "qwen2-vl-7b": "qwen/qwen2-vl-7b-instruct",
    "qwen2-vl-72b": "qwen/qwen2-vl-72b-instruct",
    "internvl2-8b": "openbmb/internvl2-8b",
    # Strong proprietary baselines available via OpenRouter.
    "claude-haiku": "anthropic/claude-haiku-4-5",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}


def _resolve_vllm_model(model_arg: str | None) -> str | None:
    """Resolve short alias to full OpenRouter model ID; pass through unknown IDs."""
    if model_arg is None:
        return None
    return VLLM_MODEL_PRESETS.get(model_arg, model_arg)


def parse_args():
    p = argparse.ArgumentParser()

    # Dataset source (mutually exclusive)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=str,
                     help="Path to reference_game_dataset/dataset.json (image-backed)")
    src.add_argument("--jsonl", type=str,
                     help="Path to a JSONL scene file (generated or augmented)")

    # Image dataset options
    p.add_argument("--target_selection", default="random",
                   choices=["random", "first", "hardest"],
                   help="How to assign target_idx when loading image dataset")
    p.add_argument("--split", action="store_true",
                   help="Evaluate on the test split only (70/15/15 stratified split)")

    # Models
    p.add_argument("--vllm_model", type=str, default=None,
                   help=(
                       "OpenRouter model ID or preset alias. "
                       "Examples: anthropic/claude-haiku-4-5, "
                       "qwen/qwen2-vl-72b-instruct, llava-1.6-7b"
                   ))
    p.add_argument("--list_vllm_models", action="store_true",
                   help="Print built-in vLLM/OpenRouter model presets and exit")
    p.add_argument("--symbolic_baselines", action="store_true",
                   help="Also run Literal and RSA speaker/listener baselines")
    p.add_argument("--alpha", type=float, default=4.0,
                   help="RSA rationality parameter")

    # Eval settings
    p.add_argument("--costs", nargs="+", type=float, default=[0.1, 0.25, 0.5],
                   help="Clarification cost values to sweep")
    p.add_argument("--workers", type=int, default=4,
                   help="Thread pool size for concurrent VLLM calls")
    p.add_argument("--out", type=str, default="results/")
    p.add_argument("--prefix", type=str, default="eval")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_vllm_models:
        print("Built-in model presets (alias -> OpenRouter model ID):")
        for alias, model_id in sorted(VLLM_MODEL_PRESETS.items()):
            print(f"  {alias:<16} -> {model_id}")
        return

    args.vllm_model = _resolve_vllm_model(args.vllm_model)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if args.dataset:
        scenes = load_image_dataset(
            json_path=args.dataset,
            target_selection=args.target_selection,
        )
        logging.info(f"Loaded {len(scenes)} image-backed scenes from {args.dataset}")
    else:
        scenes = load_jsonl(args.jsonl)
        logging.info(f"Loaded {len(scenes)} scenes from {args.jsonl}")

    logging.info(f"Dataset stats: {dataset_stats(scenes)}")

    if args.split:
        _, _, scenes = split_dataset(scenes)
        logging.info(f"Using test split: {len(scenes)} scenes")

    # ── Build speakers ────────────────────────────────────────────────────────
    speakers = []

    if args.symbolic_baselines:
        speakers += [LiteralSpeaker(), RSASpeaker(alpha=args.alpha)]

    if args.vllm_model:
        from src.refgame.utils.llm_client import openrouter
        client = openrouter(model=args.vllm_model)
        speakers += [
            VLLMSpeaker(client=client, pragmatic=False),
            VLLMSpeaker(client=client, pragmatic=True),
        ]

    if not speakers:
        logging.error("No speakers configured. Provide --vllm_model or --symbolic_baselines.")
        sys.exit(1)

    # ── Build listeners ───────────────────────────────────────────────────────
    listeners = []

    if args.symbolic_baselines:
        listeners += [LiteralListener(), RSAListener(alpha=args.alpha)]

    if args.vllm_model:
        from src.refgame.utils.llm_client import openrouter
        client = openrouter(model=args.vllm_model)
        listeners += [VLLMListener(client=client)]

    # ── Run grid ──────────────────────────────────────────────────────────────
    records = run_grid(
        scenes=scenes,
        speakers=speakers,
        listeners=listeners,
        cost_values=args.costs,
        n_workers=args.workers,
        verbose=True,
    )

    logging.info(f"Collected {len(records)} evaluation records")

    # ── Print summary to console ──────────────────────────────────────────────
    summary = summarize(records)
    print(f"\n{'─'*80}")
    print(f"{'Speaker':<30} {'Listener':<30} {'cost_c':>6}  {'CPA':>6}  {'Acc':>6}  {'Clarif':>6}")
    print(f"{'─'*80}")
    for row in summary:
        print(f"{row.get('speaker_type',''):<30} {row.get('listener_type',''):<30} "
              f"{row.get('cost_c',0):>6.2f}  {row.get('cpa',0):>6.3f}  "
              f"{row.get('accuracy',0):>6.3f}  {row.get('clarification_rate',0):>6.3f}")
    print(f"{'─'*80}")

    save_results(records, out_dir=args.out, prefix=args.prefix)


if __name__ == "__main__":
    main()
