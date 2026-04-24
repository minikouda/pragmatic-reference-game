"""
Diagnostic baseline: per-scene verbose trace to identify accuracy failure modes.

For each scene prints:
  - Target properties + all distractors
  - Speaker utterance (with strategy meta)
  - Listener predicted (x,y) → snapped object vs target
  - Pass/fail + failure category

Failure categories
------------------
  PASS          : listener correctly identified target
  COORD_MISS    : listener's (x,y) was off; nearest object was wrong
  PARSE_FAIL    : listener output could not be parsed (fell back to center)
  UTTERANCE_OK  : utterance was unambiguous but listener still failed
  AMB_UTTERANCE : utterance did not include enough discriminating features

Run:
  python scripts/run_diagnostic.py \
      --jsonl data/scenes_6_none.jsonl \
      --n 20 \
      --model gemini-flash \
      --speaker ordinal \
      --workers 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.WARNING)  # suppress LLM noise; we print our own

from src.refgame.data.dataset import load_jsonl
from src.refgame.speakers.vllm import VLLMSpeaker
from src.refgame.speakers.scene_aware import SceneAwareSpeaker
from src.refgame.speakers.landmark import LandmarkSpeaker, LandmarkVLLMSpeaker
from src.refgame.speakers.contrastive import ContrastiveSpeaker, ContrastiveVLLMSpeaker
from src.refgame.speakers.ordinal import OrdinalSpeaker
from src.refgame.listeners.vllm import VLLMListener, _parse_coords
from src.refgame.utils.llm_client import openrouter

VLLM_MODEL_PRESETS = {
    "gemini-flash":     "google/gemini-2.0-flash-001",
    "llama-4-scout":    "meta-llama/llama-4-scout",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "claude-haiku":     "anthropic/claude-haiku-4-5",
    "gpt-4o-mini":      "openai/gpt-4o-mini",
}

SPEAKER_REGISTRY = {
    "naive":            lambda c: VLLMSpeaker(client=c, pragmatic=False),
    "pragmatic":        lambda c: VLLMSpeaker(client=c, pragmatic=True),
    "scene-aware":      lambda c: SceneAwareSpeaker(client=c, ranked=False),
    "scene-ranked":     lambda c: SceneAwareSpeaker(client=c, ranked=True),
    "landmark-vllm":    lambda c: LandmarkVLLMSpeaker(client=c),
    "contrastive-vllm": lambda c: ContrastiveVLLMSpeaker(client=c),
    "ordinal":          lambda _: OrdinalSpeaker(),
    "landmark":         lambda _: LandmarkSpeaker(),
    "contrastive":      lambda _: ContrastiveSpeaker(),
}


def _dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _object_summary(obj) -> str:
    f = obj.features()
    return f"{f['size']} {f['color']} {f['shape']} @ ({obj.x_loc},{obj.y_loc}) [{f['location']}]"


def _is_ambiguous(utterance_text: str, scene, target_idx: int) -> bool:
    """True if the utterance matches more than one object (simple keyword check)."""
    target = scene.objects[target_idx]
    tf = target.features()
    utt = utterance_text.lower()

    # Count how many objects all mentioned keywords in utterance match
    matches = 0
    for obj in scene.objects:
        f = obj.features()
        # An object "matches" if none of its key features contradict the utterance
        color_ok  = f["color"]  in utt or f["color"]  not in " ".join(o.features()["color"]  for o in scene.objects)
        shape_ok  = f["shape"]  in utt or f["shape"]  not in " ".join(o.features()["shape"]  for o in scene.objects)
        # At least color+shape must be consistent with utterance
        if f["color"] in utt and f["shape"] in utt:
            matches += 1
    return matches > 1


def run_diagnostic(args):
    model_id = VLLM_MODEL_PRESETS.get(args.model, args.model)
    client   = openrouter(model=model_id)

    scenes = load_jsonl(args.jsonl)[: args.n]
    print(f"\n{'='*80}")
    print(f"  DIAGNOSTIC BASELINE")
    print(f"  Dataset : {args.jsonl}  ({len(scenes)} scenes)")
    print(f"  Speaker : {args.speaker}")
    print(f"  Model   : {model_id}")
    print(f"{'='*80}\n")

    if args.speaker not in SPEAKER_REGISTRY:
        print(f"Unknown speaker '{args.speaker}'. Options: {list(SPEAKER_REGISTRY)}")
        sys.exit(1)

    speaker  = SPEAKER_REGISTRY[args.speaker](client)
    listener = VLLMListener(client=client)

    results = []
    n_pass = n_coord_miss = n_parse_fail = n_amb = 0

    for i, scene in enumerate(scenes):
        target     = scene.objects[scene.target_idx]
        tf         = target.features()
        distractors = [o for j, o in enumerate(scene.objects) if j != scene.target_idx]

        print(f"{'─'*80}")
        print(f"Scene {scene.id}  (target_idx={scene.target_idx})")
        print(f"  TARGET   : {_object_summary(target)}")
        print(f"  SCENE    :")
        for j, obj in enumerate(scene.objects):
            marker = " ◀ TARGET" if j == scene.target_idx else ""
            print(f"    [{j}] {_object_summary(obj)}{marker}")

        # ── Speaker ──────────────────────────────────────────────────────────
        try:
            utt = speaker.speak(scene, scene.target_idx)
        except Exception as e:
            print(f"  SPEAKER  : ERROR — {e}")
            continue

        meta_str = ""
        if utt.speaker_meta:
            relevant = {k: v for k, v in utt.speaker_meta.items()
                        if k not in ("raw_response",)}
            if relevant:
                meta_str = f"  [meta: {relevant}]"

        print(f"  UTTERANCE: \"{utt.text}\"{meta_str}")

        # Quick ambiguity check
        amb = _is_ambiguous(utt.text, scene, scene.target_idx)
        if amb:
            print(f"  ⚠ AMBIGUOUS utterance — color+shape matches multiple objects")

        # ── Listener ─────────────────────────────────────────────────────────
        try:
            l_out = listener.listen(scene, utt)
        except Exception as e:
            print(f"  LISTENER : ERROR — {e}")
            continue

        raw_resp  = (l_out.listener_meta or {}).get("raw_response", "")
        pred_x    = (l_out.listener_meta or {}).get("pred_x", 50.0)
        pred_y    = (l_out.listener_meta or {}).get("pred_y", 50.0)
        pred_idx  = l_out.predicted_idx
        correct   = pred_idx == scene.target_idx

        # Both coords in image convention (top-left=0,0), no flip needed
        dist_to_target = _dist(pred_x, pred_y, target.x_loc, target.y_loc)

        print(f"  LISTENER :")
        print(f"    raw output : {raw_resp.strip()[:120]}")
        print(f"    predicted  : ({pred_x:.1f}, {pred_y:.1f})")
        print(f"    snapped to : [{pred_idx}] {_object_summary(scene.objects[pred_idx])}")
        print(f"    dist_to_target = {dist_to_target:.1f} px-units")

        # Show distances to all objects
        print(f"    distances  :", end="")
        for j, obj in enumerate(scene.objects):
            d = _dist(pred_x, pred_y, obj.x_loc, obj.y_loc)
            marker = "*" if j == scene.target_idx else ""
            print(f"  [{j}]={d:.1f}{marker}", end="")
        print()

        # Failure category
        parsed_ok = not (pred_x == 50.0 and pred_y == 50.0 and "x" not in raw_resp.lower())
        if correct:
            category = "PASS"
            n_pass  += 1
        elif not parsed_ok:
            category = "PARSE_FAIL"
            n_parse_fail += 1
        elif amb:
            category = "AMB_UTTERANCE"
            n_amb += 1
        else:
            category = "COORD_MISS"
            n_coord_miss += 1

        symbol = "✓" if correct else "✗"
        print(f"  RESULT   : {symbol} {category}")
        print()

        results.append({
            "scene_id":    scene.id,
            "target_idx":  scene.target_idx,
            "utterance":   utt.text,
            "pred_x":      pred_x,
            "pred_y":      pred_y,
            "pred_idx":    pred_idx,
            "correct":     correct,
            "ambiguous":   amb,
            "category":    category,
            "dist_to_tgt": round(dist_to_target, 1),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    print(f"{'='*80}")
    print(f"  SUMMARY  ({n} scenes)")
    print(f"{'='*80}")
    print(f"  PASS          : {n_pass}/{n}  ({n_pass/n:.1%})")
    print(f"  COORD_MISS    : {n_coord_miss}/{n}  ({n_coord_miss/n:.1%})")
    print(f"  AMB_UTTERANCE : {n_amb}/{n}  ({n_amb/n:.1%})")
    print(f"  PARSE_FAIL    : {n_parse_fail}/{n}  ({n_parse_fail/n:.1%})")
    print()

    # Breakdown by number of objects sharing same color+shape as target
    from collections import Counter
    amb_counts = Counter(r["ambiguous"] for r in results)
    print(f"  Ambiguous utterances: {amb_counts[True]}/{n}")

    # Avg distance to target on failures
    fail_dists = [r["dist_to_tgt"] for r in results if not r["correct"]]
    if fail_dists:
        print(f"  Avg dist-to-target on failures: {sum(fail_dists)/len(fail_dists):.1f}")

    correct_dists = [r["dist_to_tgt"] for r in results if r["correct"]]
    if correct_dists:
        print(f"  Avg dist-to-target on passes  : {sum(correct_dists)/len(correct_dists):.1f}")

    print()

    if args.out:
        out_path = Path(args.out)
        out_path.write_text("\n".join(json.dumps(r) for r in results) + "\n")
        print(f"  Results written to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl",    required=True,         help="Scene JSONL file")
    p.add_argument("--n",        type=int, default=20,  help="Number of scenes")
    p.add_argument("--model",    default="gemini-flash", help="Model preset or full ID")
    p.add_argument("--speaker",  default="ordinal",     help="Speaker type")
    p.add_argument("--workers",  type=int, default=1)
    p.add_argument("--out",      default=None,          help="Write JSONL results here")
    args = p.parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()
