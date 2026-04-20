"""
Script: generate a reference game dataset.

Usage
-----
# 500 scenes, balanced across ambiguity tiers, saved to data/
python scripts/generate_data.py --n 500 --n_objects 4 --seed 42 --out data/scenes.jsonl

# Generate stratified: 200 per tier (low/medium/high)
python scripts/generate_data.py --n_per_tier 200 --n_objects 5 --out data/scenes_stratified.jsonl
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.refgame.data.generator import GeneratorConfig, SceneGenerator
from src.refgame.data.dataset import save_jsonl, split_dataset, dataset_stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",          type=int, default=None, help="Total scenes (unconstrained)")
    p.add_argument("--n_per_tier", type=int, default=None, help="Scenes per tier (stratified)")
    p.add_argument("--n_objects",  type=int, default=4,    help="Objects per scene")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        type=str, default="data/scenes.jsonl")
    p.add_argument("--split",      action="store_true",    help="Also write train/val/test splits")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = GeneratorConfig(n_objects=args.n_objects, seed=args.seed)
    gen  = SceneGenerator(cfg)

    if args.n_per_tier is not None:
        scenes = gen.generate_stratified(n_per_tier=args.n_per_tier)
    elif args.n is not None:
        scenes = list(gen.generate(n=args.n))
    else:
        raise ValueError("Specify --n or --n_per_tier")

    print(f"Generated {len(scenes)} scenes")
    print(dataset_stats(scenes))

    save_jsonl(scenes, args.out)
    print(f"Saved → {args.out}")

    if args.split:
        train, val, test = split_dataset(scenes, seed=args.seed)
        base = args.out.replace(".jsonl", "")
        save_jsonl(train, f"{base}_train.jsonl")
        save_jsonl(val,   f"{base}_val.jsonl")
        save_jsonl(test,  f"{base}_test.jsonl")
        print(f"Splits: {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    main()
