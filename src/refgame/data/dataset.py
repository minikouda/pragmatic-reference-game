"""
Dataset I/O and train/val/test splitting.

Scenes are stored as JSONL (one JSON object per line) for streaming-friendly
access on large datasets without loading everything into RAM.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from .schema import Scene


def save_jsonl(scenes: list[Scene], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for scene in scenes:
            f.write(json.dumps(scene.to_dict()) + "\n")


def load_jsonl(path: str | Path) -> list[Scene]:
    path = Path(path)
    return [Scene.from_dict(json.loads(line)) for line in path.open()]


def stream_jsonl(path: str | Path) -> Iterator[Scene]:
    """Lazy loading for large datasets."""
    with Path(path).open() as f:
        for line in f:
            yield Scene.from_dict(json.loads(line))


def split_dataset(
    scenes: list[Scene],
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[list[Scene], list[Scene], list[Scene]]:
    """
    Stratified split by ambiguity_tier so each split has balanced difficulty.

    Returns (train_scenes, val_scenes, test_scenes).
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

    rng = random.Random(seed)

    # Group by tier
    tiers: dict[str, list[Scene]] = {}
    for s in scenes:
        tier = s.entropy_annotation.ambiguity_tier if s.entropy_annotation else "unknown"
        tiers.setdefault(tier, []).append(s)

    train_s, val_s, test_s = [], [], []
    for tier_scenes in tiers.values():
        shuffled = tier_scenes[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train)
        n_val   = int(n * val)
        train_s.extend(shuffled[:n_train])
        val_s.extend(shuffled[n_train:n_train + n_val])
        test_s.extend(shuffled[n_train + n_val:])

    return train_s, val_s, test_s


def dataset_stats(scenes: list[Scene]) -> dict:
    """Summary statistics for a scene list."""
    from collections import Counter
    tiers = Counter(
        s.entropy_annotation.ambiguity_tier
        for s in scenes
        if s.entropy_annotation
    )
    mdls = [
        s.entropy_annotation.min_desc_length
        for s in scenes
        if s.entropy_annotation
    ]
    return {
        "n_scenes": len(scenes),
        "tier_counts": dict(tiers),
        "mdl_mean": sum(mdls) / len(mdls) if mdls else None,
        "mdl_min": min(mdls) if mdls else None,
        "mdl_max": max(mdls) if mdls else None,
    }
