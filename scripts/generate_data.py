"""
Script: generate a synthetic reference game dataset with PNG images.

Usage
-----
# 500 unconstrained scenes, images written to data/generated_scenes/
python scripts/generate_data.py --n 500 --n_objects 6 --out data/scenes --split

# 167 scenes per ambiguity tier (low / medium / high)
python scripts/generate_data.py --n_per_tier 167 --n_objects 6 --out data/scenes --split

Output files (with --split):
  {out}_train.jsonl, {out}_val.jsonl, {out}_test.jsonl
  {out}_images/scene_0.png, scene_1.png, ...
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.refgame.data.generator import GeneratorConfig, SceneGenerator
from src.refgame.data.dataset import save_jsonl, split_dataset, dataset_stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",          type=int, default=None, help="Total scenes (flat)")
    p.add_argument("--n_per_tier", type=int, default=None, help="Scenes per ambiguity tier")
    p.add_argument("--n_objects",  type=int, default=6,    help="Objects per scene")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--overlap_mode", type=str, default="none",
                   choices=["none", "allow", "force"],
                   help="Physical-overlap policy between rendered objects")
    p.add_argument("--out",        type=str, default="data/scenes",
                   help="Output path stem (images go to {out}_images/, JSONL to {out}.jsonl)")
    p.add_argument("--split",      action="store_true",
                   help="Write train/val/test splits in addition to the full JSONL")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = GeneratorConfig(
        n_objects=args.n_objects, seed=args.seed, overlap_mode=args.overlap_mode,
    )
    gen  = SceneGenerator(cfg)

    img_dir = args.out + "_images"

    if args.n_per_tier is not None:
        # Stratified: equal scenes per tier
        scenes = []
        for tier in ("low", "medium", "high"):
            tier_cfg = GeneratorConfig(
                n_objects=args.n_objects,
                ambiguity_tier=tier,
                overlap_mode=args.overlap_mode,
                seed=args.seed + hash(tier) % 1000,
            )
            tier_gen = SceneGenerator(tier_cfg)
            tier_scenes = tier_gen.generate_with_images(
                n=args.n_per_tier,
                out_dir=img_dir,
                prefix=f"scene_{tier}",
            )
            scenes.extend(tier_scenes)
    elif args.n is not None:
        scenes = gen.generate_with_images(n=args.n, out_dir=img_dir, prefix="scene")
    else:
        print("Error: specify --n or --n_per_tier")
        sys.exit(1)

    print(f"Generated {len(scenes)} scenes  →  images in {img_dir}/")
    print(dataset_stats(scenes))

    save_jsonl(scenes, args.out + ".jsonl")
    print(f"Saved {args.out}.jsonl")

    if args.split:
        train, val, test = split_dataset(scenes, seed=args.seed)
        save_jsonl(train, args.out + "_train.jsonl")
        save_jsonl(val,   args.out + "_val.jsonl")
        save_jsonl(test,  args.out + "_test.jsonl")
        print(f"Splits: {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    main()
