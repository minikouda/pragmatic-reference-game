"""
Dataset I/O: loading the reference_game_dataset image scenes and JSONL I/O.

Primary loader: `load_image_dataset` reads reference_game_dataset/dataset.json
and converts each scene into a typed Scene object, assigning a target_idx.

Secondary: JSONL helpers for generated/augmented datasets.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterator

from .schema import (
    FEATURE_KEYS, SIZE_MAP, EntropyAnnotation,
    Object, Scene, loc_label,
)


# ── Image dataset loader ─────────────────────────────────────────────────────

def load_image_dataset(
    json_path: str | Path = "reference_game_dataset/dataset.json",
    target_selection: str = "random",
    seed: int = 42,
) -> list[Scene]:
    """
    Load scenes from the rendered reference_game_dataset.

    Parameters
    ----------
    json_path        : path to dataset.json
    target_selection : how to assign target_idx per scene
                       "random"   — pick uniformly at random (reproducible via seed)
                       "first"    — always use index 0
                       "hardest"  — pick the object with the highest min_desc_length
                                    (most ambiguous target, hardest for the listener)
    seed             : RNG seed for "random" mode

    Returns
    -------
    List of Scene objects with image_path and entropy_annotation set.
    """
    rng   = random.Random(seed)
    raw   = json.loads(Path(json_path).read_text())
    scenes: list[Scene] = []

    for i, entry in enumerate(raw):
        objects = [
            _convert_object(j, obj)
            for j, obj in enumerate(entry["objects"])
        ]
        if target_selection == "random":
            target_idx = rng.randrange(len(objects))
        elif target_selection == "first":
            target_idx = 0
        elif target_selection == "hardest":
            target_idx = _hardest_target(objects)
        else:
            raise ValueError(f"Unknown target_selection: {target_selection!r}")

        ann = _compute_annotation(objects, target_idx)
        scenes.append(Scene(
            id=str(i),
            objects=objects,
            target_idx=target_idx,
            image_path=entry["image"],
            entropy_annotation=ann,
        ))

    return scenes


def _convert_object(idx: int, raw: dict) -> Object:
    """Convert a raw dataset.json object entry to a typed Object."""
    size_str = SIZE_MAP.get(raw["size"], "medium")
    loc      = loc_label(raw["x_loc"], raw["y_loc"], canvas=100)
    return Object(
        id=str(idx),
        color=raw["color"],
        shape=raw["type"],
        size=size_str,
        location=loc,
        x_loc=raw["x_loc"],
        y_loc=raw["y_loc"],
    )


def _hardest_target(objects: list[Object]) -> int:
    """Return the index of the object with the largest min_desc_length."""
    best_idx, best_mdl = 0, 0
    for i, obj in enumerate(objects):
        distractors = [o for j, o in enumerate(objects) if j != i]
        scene = Scene(id="_tmp", objects=[obj] + distractors, target_idx=0)
        mdl   = scene.min_description_length()
        if mdl > best_mdl:
            best_mdl, best_idx = mdl, i
    return best_idx


def _compute_annotation(objects: list[Object], target_idx: int) -> EntropyAnnotation:
    """Compute EntropyAnnotation for a loaded scene."""
    from itertools import combinations

    target     = objects[target_idx]
    distractors = [o for i, o in enumerate(objects) if i != target_idx]

    # min_desc_length
    t_feats = target.features()
    mdl = len(FEATURE_KEYS) + 1
    for r in range(1, len(FEATURE_KEYS) + 1):
        for combo in combinations(FEATURE_KEYS, r):
            vals = {k: t_feats[k] for k in combo}
            if not any(
                all(d.features()[k] == v for k, v in vals.items())
                for d in distractors
            ):
                mdl = r
                break
        if mdl <= r:
            break

    # max overlap
    def overlap(a: Object, b: Object) -> int:
        fa, fb = a.features(), b.features()
        return sum(fa[k] == fb[k] for k in FEATURE_KEYS)

    max_ov = max(overlap(target, d) for d in distractors) if distractors else 0
    tier   = "low" if mdl <= 1 else ("medium" if mdl == 2 else "high")

    return EntropyAnnotation(
        n_objects=len(objects),
        min_desc_length=mdl,
        max_overlap=max_ov,
        h_uniform=math.log(len(objects)),
        ambiguity_tier=tier,
    )


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def save_jsonl(scenes: list[Scene], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for scene in scenes:
            f.write(json.dumps(scene.to_dict()) + "\n")


def load_jsonl(path: str | Path) -> list[Scene]:
    return [Scene.from_dict(json.loads(line)) for line in Path(path).open()]


def stream_jsonl(path: str | Path) -> Iterator[Scene]:
    with Path(path).open() as f:
        for line in f:
            yield Scene.from_dict(json.loads(line))


# ── Train / val / test split ──────────────────────────────────────────────────

def split_dataset(
    scenes: list[Scene],
    train: float = 0.7,
    val:   float = 0.15,
    test:  float = 0.15,
    seed:  int   = 42,
) -> tuple[list[Scene], list[Scene], list[Scene]]:
    """Stratified split by ambiguity_tier."""
    assert abs(train + val + test - 1.0) < 1e-6
    rng   = random.Random(seed)
    tiers: dict[str, list[Scene]] = {}
    for s in scenes:
        tier = s.entropy_annotation.ambiguity_tier if s.entropy_annotation else "unknown"
        tiers.setdefault(tier, []).append(s)

    tr, va, te = [], [], []
    for group in tiers.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n      = len(shuffled)
        n_tr   = int(n * train)
        n_va   = int(n * val)
        tr.extend(shuffled[:n_tr])
        va.extend(shuffled[n_tr:n_tr + n_va])
        te.extend(shuffled[n_tr + n_va:])
    return tr, va, te


def dataset_stats(scenes: list[Scene]) -> dict:
    from collections import Counter
    tiers = Counter(
        s.entropy_annotation.ambiguity_tier for s in scenes if s.entropy_annotation
    )
    mdls = [s.entropy_annotation.min_desc_length for s in scenes if s.entropy_annotation]
    return {
        "n_scenes":   len(scenes),
        "has_images": sum(1 for s in scenes if s.image_path),
        "tier_counts": dict(tiers),
        "mdl_mean":   round(sum(mdls) / len(mdls), 2) if mdls else None,
        "mdl_min":    min(mdls) if mdls else None,
        "mdl_max":    max(mdls) if mdls else None,
    }
