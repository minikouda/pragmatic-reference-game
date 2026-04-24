"""
Experiment: does the VLLMSpeaker hallucinate features not in the target's properties?

Since the speaker is now given the target's ground-truth properties in the prompt,
any feature word in its utterance that contradicts those properties is a hallucination.

For each scene we:
  1. Run the speaker to get an utterance.
  2. Extract all feature tokens (color/shape/size/location vocab words) from the utterance.
  3. Check each token against the target's ground-truth properties.
     - CORRECT   : token matches one of the target's property values
     - HALLUCINATED : token is a valid vocab word but does NOT match the target
     - IGNORED   : token is not a vocab word (articles, prepositions, etc.)

Usage
-----
python scripts/test_speaker_hallucination.py \
    --jsonl data/smoke_test.jsonl \
    --model gemini-flash \
    --mode naive \
    --out results/hallucination_test/ \
    --n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import load_jsonl
from src.refgame.data.schema import FEATURE_VOCAB
from src.refgame.data.generator import GeneratorConfig, SceneGenerator
from src.refgame.speakers.vllm import VLLMSpeaker
from src.refgame.utils.llm_client import openrouter

VLLM_MODEL_PRESETS = {
    "gemini-flash":     "google/gemini-2.0-flash-001",
    "llama-4-scout":    "meta-llama/llama-4-scout",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "claude-haiku":     "anthropic/claude-haiku-4-5",
    "gpt-4o-mini":      "openai/gpt-4o-mini",
}

# Flat set of all valid feature vocab words
_ALL_VOCAB: frozenset[str] = frozenset(
    v for vals in FEATURE_VOCAB.values() for v in vals
)
_STOP = frozenset({"the", "a", "an", "on", "at", "in", "of", "one"})


def extract_feature_tokens(text: str) -> list[str]:
    """Return all vocab words mentioned in the utterance (lowercased, stop-words removed)."""
    tokens = [t.strip(".,;:\"'") for t in text.lower().split()]
    return [t for t in tokens if t in _ALL_VOCAB and t not in _STOP]


def analyze_utterance(utterance: str, gt: dict) -> dict:
    """
    Compare feature tokens in the utterance against ground-truth properties.

    Returns counts and lists of correct / hallucinated tokens.
    """
    gt_values = set(gt.values())
    tokens = extract_feature_tokens(utterance)

    correct      = [t for t in tokens if t in gt_values]
    hallucinated = [t for t in tokens if t not in gt_values]

    return {
        "tokens":       tokens,
        "correct":      correct,
        "hallucinated": hallucinated,
        "n_tokens":     len(tokens),
        "n_correct":    len(correct),
        "n_hallucinated": len(hallucinated),
        "has_hallucination": len(hallucinated) > 0,
    }


def run(scenes, speaker: VLLMSpeaker) -> list[dict]:
    records = []
    for i, scene in enumerate(scenes):
        try:
            utt = speaker.speak(scene, scene.target_idx)
            target = scene.objects[scene.target_idx]
            gt = target.features()
            analysis = analyze_utterance(utt.text, gt)

            rec = {
                "scene_id":    scene.id,
                "target_idx":  scene.target_idx,
                "n_objects":   len(scene.objects),
                "ground_truth": gt,
                "utterance":   utt.text,
                **analysis,
            }
            records.append(rec)

            flag = "HALLUC" if analysis["has_hallucination"] else "ok    "
            logging.info(
                f"[{i+1}/{len(scenes)}] {flag}  "
                f"gt={list(gt.values())}  "
                f"utt='{utt.text}'  "
                f"hallucinated={analysis['hallucinated']}"
            )
        except Exception as e:
            logging.error(f"Scene {scene.id} failed: {e}")
    return records


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    halluc_rate = sum(r["has_hallucination"] for r in records) / n

    # Per-feature: how often does the utterance mention the wrong value for each feature
    feature_halluc: dict[str, int] = defaultdict(int)
    feature_total:  dict[str, int] = defaultdict(int)
    for r in records:
        utt_tokens = set(r["tokens"])
        for feat, val in r["ground_truth"].items():
            # Check if any OTHER value from this feature's vocab appears
            other_vals = set(FEATURE_VOCAB[feat]) - {val}
            if utt_tokens & other_vals:
                feature_halluc[feat] += 1
            feature_total[feat] += 1

    # Most common hallucinated tokens
    halluc_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for t in r["hallucinated"]:
            halluc_counts[t] += 1

    return {
        "n": n,
        "hallucination_rate": halluc_rate,
        "per_feature_hallucination_rate": {
            k: feature_halluc[k] / feature_total[k]
            for k in FEATURE_VOCAB
        },
        "top_hallucinated_tokens": sorted(
            halluc_counts.items(), key=lambda x: -x[1]
        )[:10],
        "mean_tokens_per_utterance": sum(r["n_tokens"] for r in records) / n,
        "mean_hallucinated_per_utterance": sum(r["n_hallucinated"] for r in records) / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--jsonl", type=str, help="JSONL scene file")
    src.add_argument("--smoke", action="store_true", help="Generate 10 scenes inline")
    p.add_argument("--model", type=str, default="gemini-flash")
    p.add_argument("--mode",  type=str, default="naive", choices=["naive", "pragmatic"])
    p.add_argument("--n",     type=int, default=None, help="Max scenes to test")
    p.add_argument("--out",   type=str, default="results/hallucination_test/")
    args = p.parse_args()

    model_id = VLLM_MODEL_PRESETS.get(args.model, args.model)
    client   = openrouter(model=model_id)
    speaker  = VLLMSpeaker(client=client, pragmatic=(args.mode == "pragmatic"))

    if args.smoke:
        gen    = SceneGenerator(GeneratorConfig(n_objects=4, seed=99))
        scenes = gen.generate_with_images(10, out_dir="data/smoke_images", prefix="halluc_smoke")
    elif args.jsonl:
        scenes = load_jsonl(args.jsonl)
    else:
        p.error("Provide --jsonl or --smoke")

    if args.n:
        scenes = scenes[: args.n]

    logging.info(f"Testing {len(scenes)} scenes | model={model_id} | mode={args.mode}")

    records = run(scenes, speaker)
    summary = summarize(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "hallucination_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(out_dir / "hallucination_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*60}")
    print(f"Model: {model_id}  |  Mode: {args.mode}  |  N: {summary['n']}")
    print(f"{'─'*60}")
    print(f"  Hallucination rate          : {summary['hallucination_rate']:.1%}")
    print(f"  Mean tokens / utterance     : {summary['mean_tokens_per_utterance']:.1f}")
    print(f"  Mean hallucinated / utt     : {summary['mean_hallucinated_per_utterance']:.2f}")
    print(f"  Per-feature hallucination rate:")
    for feat, rate in summary["per_feature_hallucination_rate"].items():
        print(f"    {feat:<10}: {rate:.1%}")
    print(f"  Top hallucinated tokens:")
    for tok, cnt in summary["top_hallucinated_tokens"]:
        print(f"    '{tok}' × {cnt}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
