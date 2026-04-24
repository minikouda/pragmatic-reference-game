"""
Experiment: does the VLLMSpeaker correctly perceive the target object?

For each scene, show the model the annotated image (red TARGET box) and ask it
to output the target's features as structured JSON.  Compare against ground
truth to compute per-feature accuracy and a "fully correct" rate.

This isolates visual perception from language generation — if feature accuracy
is low, the red box is misleading the model (e.g. blending with red objects).

Usage
-----
python scripts/test_speaker_perception.py \
    --jsonl data/smoke_test.jsonl \
    --model gemini-flash \
    --out results/perception_test/ \
    --n 20

Output
------
  results/perception_test/perception_records.jsonl   — per-scene results
  results/perception_test/perception_summary.json    — per-feature accuracy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.refgame.data.dataset import load_jsonl
from src.refgame.data.generator import GeneratorConfig, SceneGenerator
from src.refgame.speakers.vllm import _annotate_image
from src.refgame.utils.llm_client import openrouter

VLLM_MODEL_PRESETS = {
    "gemini-flash":     "google/gemini-2.0-flash-001",
    "llama-4-scout":    "meta-llama/llama-4-scout",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "claude-haiku":     "anthropic/claude-haiku-4-5",
    "gpt-4o-mini":      "openai/gpt-4o-mini",
}

_SYSTEM_PROBE = """\
You are examining a scene image. The TARGET object is highlighted with a red \
bounding box labeled "TARGET".

Output a JSON object with exactly these keys describing the TARGET object:
  "color"    : one of black, blue, green, red, yellow
  "shape"    : one of circle, square, triangle
  "size"     : one of small, medium, large
  "location" : one of top-left, top-right, bottom-left, bottom-right, top, bottom, left, right, center

Output ONLY valid JSON. No explanation."""


def probe_scene(client, scene) -> dict:
    """Ask the VLM to output structured features for the target object."""
    from src.refgame.utils.llm_client import ChatMessage

    target = scene.objects[scene.target_idx]
    annotated = _annotate_image(scene.image_path, scene.objects, scene.target_idx)

    raw = client.complete(
        messages=[
            ChatMessage(role="system", content=_SYSTEM_PROBE),
            ChatMessage(role="user",   content="What are the features of the TARGET object?"),
        ],
        image_path=annotated,
    )

    # Parse predicted features
    pred = _parse_features(raw)

    # Compare against ground truth
    gt = {
        "color":    target.color,
        "shape":    target.shape,
        "size":     target.size,
        "location": target.location,
    }

    matches = {k: (pred.get(k) == gt[k]) for k in gt}
    fully_correct = all(matches.values())

    return {
        "scene_id":      scene.id,
        "target_idx":    scene.target_idx,
        "n_objects":     len(scene.objects),
        "ground_truth":  gt,
        "predicted":     pred,
        "matches":       matches,
        "fully_correct": fully_correct,
        "raw_response":  raw,
    }


def _parse_features(raw: str) -> dict:
    """Extract JSON feature dict from model response."""
    import re
    try:
        blob = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if blob:
            return json.loads(blob.group())
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    features = ["color", "shape", "size", "location"]
    per_feature = {
        k: sum(r["matches"].get(k, False) for r in records) / n
        for k in features
    }
    fully_correct = sum(r["fully_correct"] for r in records) / n

    # Break down fully_correct by whether the target is red
    red_target   = [r for r in records if r["ground_truth"]["color"] == "red"]
    other_target = [r for r in records if r["ground_truth"]["color"] != "red"]

    return {
        "n": n,
        "fully_correct": fully_correct,
        "per_feature_accuracy": per_feature,
        "fully_correct_red_target":   (
            sum(r["fully_correct"] for r in red_target) / len(red_target)
            if red_target else None
        ),
        "fully_correct_non_red_target": (
            sum(r["fully_correct"] for r in other_target) / len(other_target)
            if other_target else None
        ),
        "n_red_target": len(red_target),
        "n_non_red_target": len(other_target),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--jsonl", type=str, help="JSONL scene file")
    src.add_argument("--smoke", action="store_true", help="Generate 10 scenes inline")
    p.add_argument("--model",  type=str, default="gemini-flash",
                   help="Model alias or full OpenRouter model ID")
    p.add_argument("--n",      type=int, default=None,
                   help="Max number of scenes to probe (default: all)")
    p.add_argument("--out",    type=str, default="results/perception_test/")
    args = p.parse_args()

    model_id = VLLM_MODEL_PRESETS.get(args.model, args.model)
    client   = openrouter(model=model_id)

    if args.smoke:
        gen    = SceneGenerator(GeneratorConfig(n_objects=4, seed=42))
        scenes = gen.generate_with_images(10, out_dir="data/smoke_images", prefix="perc_smoke")
    elif args.jsonl:
        scenes = load_jsonl(args.jsonl)
    else:
        p.error("Provide --jsonl or --smoke")

    if args.n:
        scenes = scenes[: args.n]

    logging.info(f"Probing {len(scenes)} scenes with {model_id}")

    records = []
    for i, scene in enumerate(scenes):
        try:
            rec = probe_scene(client, scene)
            records.append(rec)
            status = "✓" if rec["fully_correct"] else "✗"
            logging.info(
                f"[{i+1}/{len(scenes)}] scene={scene.id}  {status}  "
                f"gt={rec['ground_truth']}  pred={rec['predicted']}"
            )
        except Exception as e:
            logging.error(f"Scene {scene.id} failed: {e}")

    summary = summarize(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "perception_records.jsonl"
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary_path = out_dir / "perception_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'─'*55}")
    print(f"Model: {model_id}   Scenes: {summary['n']}")
    print(f"{'─'*55}")
    print(f"  Fully correct          : {summary['fully_correct']:.1%}")
    print(f"  Fully correct (red tgt): {summary['fully_correct_red_target']}")
    print(f"  Fully correct (non-red): {summary['fully_correct_non_red_target']}")
    print(f"  Per-feature accuracy:")
    for feat, acc in summary["per_feature_accuracy"].items():
        print(f"    {feat:<10}: {acc:.1%}")
    print(f"{'─'*55}")
    print(f"Records → {records_path}")
    print(f"Summary → {summary_path}\n")


if __name__ == "__main__":
    main()
