"""
Script: run the full speaker × listener × cost grid evaluation.

Typical experiment sizes and estimated costs
--------------------------------------------
All cost estimates assume Llama-4-Maverick via OpenRouter
($0.18/M input tokens, $0.72/M output tokens, ~1 300 tokens/image call).

  Smoke test  (5 scenes,   2 vllm spk, 1 vllm lst, 3 costs)
    LLM calls : 10 speaker + 15 listener = 25
    Est. cost : < $0.01

  Small run   (50 scenes,  2 vllm spk, 1 vllm lst, 3 costs)
    LLM calls : 100 speaker + 150 listener = 250
    Est. cost : ~$0.05

  Full run    (500 scenes, 2 vllm spk, 1 vllm lst, 3 costs)
    LLM calls : 1 000 speaker + 1 500 listener = 2 500
    Est. cost : ~$0.30 (gemini-flash) / ~$0.55 (llama-4-maverick)

  With symbolic baselines (+ literal + RSA speakers/listeners, no API cost):
    Same VLM call count; symbolic models are free.

Note: speaker utterances and listener posteriors are cached per scene so the
cost sweep over multiple `cost_c` values adds zero additional LLM calls.

Usage examples
--------------
# Full experiment with balanced synthetic scenes:
python scripts/generate_data.py --n_per_tier 167 --n_objects 6 --out data/scenes --split
python scripts/run_eval.py \\
    --jsonl data/scenes_test.jsonl \\
    --vllm_model meta-llama/llama-4-maverick \\
    --symbolic_baselines \\
    --workers 8 \\
    --out results/full_run/ \\
    --prefix full

# Quick smoke test (5 generated scenes, no data file needed):
python scripts/run_eval.py \\
    --smoke \\
    --vllm_model meta-llama/llama-4-maverick \\
    --out results/smoke/

# Proprietary model baseline:
python scripts/run_eval.py \\
    --jsonl data/scenes_test.jsonl \\
    --vllm_model anthropic/claude-haiku-4-5 \\
    --symbolic_baselines \\
    --out results/haiku/

# List available built-in model aliases:
python scripts/run_eval.py --list_vllm_models --smoke

Output files (in --out directory)
----------------------------------
  {prefix}_records.jsonl   one JSON record per (scene × speaker × listener × cost_c)
  {prefix}_summary.json    aggregated metrics grouped by (speaker, listener, cost_c)
  {prefix}_by_tier.json    same but further broken down by ambiguity_tier

Fields in each record
---------------------
  scene_id, speaker_type, listener_type, cost_c
  utterance          : the speaker's natural language expression
  action             : "commit" or "ask"
  predicted_idx      : listener's argmax prediction
  target_idx         : ground-truth target index
  correct            : bool
  eu_commit          : max posterior probability (E[U|commit])
  eu_ask             : 1 - cost_c (E[U|ask])
  entropy            : referential entropy H(T|u)
  brier_score        : (p_target - 1)^2 + sum of p_distractor^2
  min_desc_length    : scene-level ambiguity annotation
  ambiguity_tier     : "low" | "medium" | "high"
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import load_jsonl, dataset_stats, split_dataset
from src.refgame.data.generator import GeneratorConfig, SceneGenerator
from src.refgame.speakers.literal import LiteralSpeaker
from src.refgame.speakers.rsa import RSASpeaker
from src.refgame.speakers.vllm import VLLMSpeaker
from src.refgame.listeners.literal import LiteralListener
from src.refgame.listeners.rsa import RSAListener
from src.refgame.listeners.vllm import VLLMListener
from src.refgame.eval.harness import run_grid
from src.refgame.eval.reporter import save_results, summarize


# ── Model presets ─────────────────────────────────────────────────────────────
# Short aliases for commonly-used OpenRouter model IDs.
# Cost column: estimated $/1M input tokens (check openrouter.ai/models for current prices).
VLLM_MODEL_PRESETS = {
    # Vision-capable open-weight models
    "gemini-flash":      "google/gemini-2.0-flash-001",    # $0.10/M  — recommended: cheapest strong vision model (verified)
    "llama-4-scout":     "meta-llama/llama-4-scout",       # $0.11/M  — good balance of cost and quality
    "llama-4-maverick":  "meta-llama/llama-4-maverick",    # $0.18/M  — stronger, costlier
    "qwen2-vl-7b":       "qwen/qwen2-vl-7b-instruct",      # $0.10/M  — small Qwen vision model
    "qwen2-vl-72b":      "qwen/qwen2-vl-72b-instruct",     # $0.40/M  — large Qwen
    # Proprietary baselines (more expensive)
    "claude-haiku":      "anthropic/claude-haiku-4-5",     # $0.80/M
    "gpt-4o-mini":       "openai/gpt-4o-mini",             # $0.15/M
    "gpt-4o":            "openai/gpt-4o",                  # $2.50/M
}


def _resolve_model(alias: str | None) -> str | None:
    if alias is None:
        return None
    return VLLM_MODEL_PRESETS.get(alias, alias)


def _estimate_cost(n_scenes: int, n_vllm_spk: int, n_vllm_lst: int) -> str:
    """Back-of-envelope cost estimate assuming Llama-4-Maverick pricing."""
    spk_calls = n_scenes * n_vllm_spk
    lst_calls = n_scenes * (n_vllm_spk + 1) * n_vllm_lst  # +1 for literal speaker
    total_calls = spk_calls + lst_calls
    tokens_in  = total_calls * 1_300
    tokens_out = total_calls * 50
    cost = tokens_in / 1e6 * 0.10 + tokens_out / 1e6 * 0.40  # gemini-flash pricing
    return (
        f"Estimated LLM calls: {spk_calls} speaker + {lst_calls} listener = {total_calls}  "
        f"| Est. cost (Llama-4-Maverick): ${cost:.2f}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Dataset source
    src = p.add_mutually_exclusive_group()
    src.add_argument("--jsonl", type=str, nargs="+",
                     help="One or more JSONL scene files. Each is evaluated separately and results saved with the file stem as prefix.")
    src.add_argument("--smoke", action="store_true",
                     help="Generate 5 fresh scenes inline for a quick smoke test")

    p.add_argument("--split", action="store_true",
                   help="If using --jsonl, evaluate on the test split only (70/15/15)")

    # Models
    p.add_argument("--vllm_model", type=str, default=None,
                   help=(
                       "OpenRouter model ID or preset alias. "
                       "See --list_vllm_models for built-in aliases. "
                       "Example: meta-llama/llama-4-maverick"
                   ))
    p.add_argument("--list_vllm_models", action="store_true",
                   help="Print built-in model presets (alias → full ID) and exit")
    p.add_argument("--symbolic_baselines", action="store_true",
                   help="Also run Literal and RSA speaker/listener baselines (no API cost)")
    p.add_argument("--alpha", type=float, default=4.0,
                   help="RSA rationality parameter α (default: 4.0)")
    p.add_argument("--pragmatic_speaker", action="store_true",
                   help="Also run VLLMSpeaker in pragmatic (chain-of-thought) mode")

    # Eval settings
    p.add_argument("--costs", nargs="+", type=float, default=[0.1, 0.25, 0.5],
                   help="Clarification cost values to sweep (default: 0.1 0.25 0.5)")
    p.add_argument("--workers", type=int, default=4,
                   help="Thread pool size for concurrent LLM calls (default: 4)")
    p.add_argument("--out", type=str, default="results/",
                   help="Output directory (default: results/)")
    p.add_argument("--prefix", type=str, default="eval",
                   help="Output file prefix (default: eval)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print cost estimate and exit without making any LLM calls")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list_vllm_models:
        print("Built-in model presets (alias → OpenRouter model ID):")
        for alias, model_id in sorted(VLLM_MODEL_PRESETS.items()):
            print(f"  {alias:<20} → {model_id}")
        return

    args.vllm_model = _resolve_model(args.vllm_model)

    # ── Resolve dataset paths ─────────────────────────────────────────────────
    if args.smoke:
        logging.info("Smoke mode: generating 5 scenes inline")
        gen = SceneGenerator(GeneratorConfig(n_objects=4, seed=0))
        smoke_scenes = gen.generate_with_images(5, out_dir="data/smoke_images", prefix="smoke")
        jsonl_paths = None  # handled inline below
    elif args.jsonl:
        jsonl_paths = args.jsonl
    else:
        logging.error("Provide --jsonl <path> [<path> ...] or --smoke")
        sys.exit(1)

    # ── Build speakers ────────────────────────────────────────────────────────
    speakers = []

    if args.symbolic_baselines:
        speakers += [LiteralSpeaker(), RSASpeaker(alpha=args.alpha)]

    if args.vllm_model:
        from src.refgame.utils.llm_client import openrouter
        client = openrouter(model=args.vllm_model)
        speakers.append(VLLMSpeaker(client=client, pragmatic=False))
        if args.pragmatic_speaker:
            speakers.append(VLLMSpeaker(client=client, pragmatic=True))

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
        listeners.append(VLLMListener(client=client))

    if not listeners:
        logging.error("No listeners configured. Provide --vllm_model or --symbolic_baselines.")
        sys.exit(1)

    # ── Collect dataset scenes ────────────────────────────────────────────────
    if args.smoke:
        dataset_list = [("smoke", smoke_scenes)]
    else:
        dataset_list = []
        for path in jsonl_paths:
            scenes = load_jsonl(path)
            logging.info(f"Loaded {len(scenes)} scenes from {path}")
            if args.split:
                _, _, scenes = split_dataset(scenes)
                logging.info(f"Using test split: {len(scenes)} scenes")
            stem = Path(path).stem
            dataset_list.append((stem, scenes))

    total_scenes = sum(len(s) for _, s in dataset_list)

    # ── Cost / dry-run estimate ───────────────────────────────────────────────
    n_vllm_spk = sum(1 for s in speakers if "vllm" in s.name)
    n_vllm_lst = sum(1 for l in listeners if "vllm" in l.name)
    est = _estimate_cost(total_scenes, n_vllm_spk, n_vllm_lst)
    logging.info(est + " (gemini-flash pricing; scale by 1.8x for llama-4-maverick)")

    if args.dry_run:
        print(est)
        print(f"Speakers  : {[s.name for s in speakers]}")
        print(f"Listeners : {[l.name for l in listeners]}")
        print(f"Costs     : {args.costs}")
        print(f"Datasets  : {[stem for stem, _ in dataset_list]}")
        print(f"Scenes    : {total_scenes} total ({len(dataset_list)} datasets)")
        return

    # ── Run grid over each dataset ────────────────────────────────────────────
    all_records = []
    for stem, scenes in dataset_list:
        logging.info(f"Running grid on dataset '{stem}' ({len(scenes)} scenes)")
        records = run_grid(
            scenes=scenes,
            speakers=speakers,
            listeners=listeners,
            cost_values=args.costs,
            n_workers=args.workers,
            verbose=True,
        )
        logging.info(f"  → {len(records)} records for '{stem}'")
        all_records.extend(records)
        save_results(records, out_dir=args.out, prefix=stem)

    logging.info(f"All datasets done — {len(all_records)} records total")

    # ── Print combined summary ────────────────────────────────────────────────
    summary = summarize(all_records)
    print(f"\n{'─'*88}")
    print(f"{'Speaker':<32} {'Listener':<28} {'cost':>5}  {'CPA':>6}  {'Acc':>6}  {'Ask%':>6}  n")
    print(f"{'─'*88}")
    for row in summary:
        print(
            f"{row.get('speaker_type',''):<32} {row.get('listener_type',''):<28} "
            f"{row.get('cost_c', 0):>5.2f}  {row.get('cpa', 0):>6.3f}  "
            f"{row.get('accuracy', 0):>6.3f}  {row.get('clarification_rate', 0):>6.1%}  "
            f"{row.get('n', 0)}"
        )
    print(f"{'─'*88}\n")


if __name__ == "__main__":
    main()
