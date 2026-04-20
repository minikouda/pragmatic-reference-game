"""
Scene generator with explicit control over referential entropy.

Design choices
--------------
- All randomness flows through a single `numpy.random.Generator` (seeded),
  making experiments reproducible.
- Ambiguity is controlled by `min_desc_length`: the minimum number of
  features required to uniquely identify the target among distractors.
  Setting this to 1 → easy; 3–4 → hard.
- `max_overlap` controls how many features the most similar distractor
  shares with the target (independent knob from min_desc_length).
- The generator rejects and resamples scenes that don't meet the requested
  constraints (rejection sampling with a max-iteration guard).
"""

from __future__ import annotations

import uuid
import random as _random
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .schema import (
    COLORS, SHAPES, SIZES, LOCATIONS, FEATURE_KEYS,
    Object, Scene, EntropyAnnotation,
)


# ── Generator config ────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    """
    Parameters controlling scene complexity.

    n_objects          : number of objects per scene (including target)
    min_desc_length    : target minimum description length (1–4)
                         1 = one feature suffices to identify target (easy)
                         3 = three features needed (hard)
                         None = unconstrained (random)
    max_overlap        : maximum feature overlap any distractor may share
                         with the target.  None = unconstrained.
    ambiguity_tier     : convenience shorthand ("low"|"medium"|"high"|None)
                         overrides min_desc_length if set
    seed               : RNG seed (None = non-deterministic)
    max_rejection_iters: rejection sampling cap before raising
    """
    n_objects:           int         = 4
    min_desc_length:     int | None  = None
    max_overlap:         int | None  = None
    ambiguity_tier:      str | None  = None   # "low" | "medium" | "high"
    seed:                int | None  = 42
    max_rejection_iters: int         = 500

    def __post_init__(self) -> None:
        # Resolve ambiguity_tier → min_desc_length
        _tier_map = {"low": 1, "medium": 2, "high": 3}
        if self.ambiguity_tier is not None:
            if self.ambiguity_tier not in _tier_map:
                raise ValueError(f"ambiguity_tier must be one of {list(_tier_map)}")
            self.min_desc_length = _tier_map[self.ambiguity_tier]


# ── Core generator ───────────────────────────────────────────────────────────

class SceneGenerator:
    """
    Generates reference game scenes with controllable referential entropy.

    Usage
    -----
    >>> cfg = GeneratorConfig(n_objects=5, ambiguity_tier="high", seed=0)
    >>> gen = SceneGenerator(cfg)
    >>> scenes = list(gen.generate(n=500))
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._rng = _random.Random(config.seed)

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, n: int) -> Iterator[Scene]:
        """Yield `n` scenes satisfying the configured constraints."""
        for i in range(n):
            yield self._sample_scene(scene_id=str(i))

    def generate_stratified(
        self,
        n_per_tier: int,
        tiers: Iterable[str] = ("low", "medium", "high"),
    ) -> list[Scene]:
        """
        Generate a balanced dataset across ambiguity tiers.

        Each tier uses the same n_objects and seed offset so the dataset
        is reproducible but the tiers don't share RNG state.
        """
        scenes: list[Scene] = []
        for tier in tiers:
            cfg = GeneratorConfig(
                n_objects=self.config.n_objects,
                ambiguity_tier=tier,
                seed=(self.config.seed or 0) + hash(tier) % 1000,
                max_rejection_iters=self.config.max_rejection_iters,
            )
            sub = SceneGenerator(cfg)
            scenes.extend(sub.generate(n_per_tier))
        return scenes

    # ── Internals ────────────────────────────────────────────────────────────

    def _sample_object(self, obj_id: str) -> Object:
        return Object(
            id=obj_id,
            color=self._rng.choice(COLORS),
            shape=self._rng.choice(SHAPES),
            size=self._rng.choice(SIZES),
            location=self._rng.choice(LOCATIONS),
        )

    def _feature_overlap(self, a: Object, b: Object) -> int:
        """Number of feature dimensions with the same value."""
        fa, fb = a.features(), b.features()
        return sum(fa[k] == fb[k] for k in FEATURE_KEYS)

    def _compute_min_desc_length(self, target: Object, distractors: list[Object]) -> int:
        """Wrapper around Scene helper; avoids constructing a full Scene."""
        scene = Scene(id="_tmp", objects=[target] + distractors, target_idx=0)
        return scene.min_description_length()

    def _meets_constraints(
        self,
        target: Object,
        distractors: list[Object],
    ) -> bool:
        cfg = self.config

        mdl = self._compute_min_desc_length(target, distractors)
        max_ov = max(self._feature_overlap(target, d) for d in distractors)

        if cfg.min_desc_length is not None and mdl != cfg.min_desc_length:
            return False
        if cfg.max_overlap is not None and max_ov > cfg.max_overlap:
            return False

        # Sanity: target must be uniquely identifiable by full description
        if mdl > len(FEATURE_KEYS):
            return False

        return True

    def _sample_scene(self, scene_id: str) -> Scene:
        cfg = self.config
        for _ in range(cfg.max_rejection_iters):
            # Sample target
            target = self._sample_object("T")
            # Sample distractors (allow attribute collisions — that's the point)
            distractors = [
                self._sample_object(f"D{i}")
                for i in range(cfg.n_objects - 1)
            ]
            # Ensure no two objects are identical across all features
            all_objs = [target] + distractors
            feature_sets = [frozenset(o.features().items()) for o in all_objs]
            if len(feature_sets) != len(set(feature_sets)):
                continue
            if not self._meets_constraints(target, distractors):
                continue

            # Assign stable IDs and shuffle order
            combined = all_objs[:]
            self._rng.shuffle(combined)
            # Reassign IDs after shuffle for cleanliness
            combined = [
                Object(id=f"obj_{i}", color=o.color, shape=o.shape,
                       size=o.size, location=o.location)
                for i, o in enumerate(combined)
            ]
            target_idx = next(
                i for i, o in enumerate(combined) if o.color == target.color
                and o.shape == target.shape and o.size == target.size
                and o.location == target.location
            )

            target_obj = combined[target_idx]
            distractor_objs = [o for i, o in enumerate(combined) if i != target_idx]
            mdl     = self._compute_min_desc_length(target_obj, distractor_objs)
            max_ov  = max(self._feature_overlap(target_obj, d) for d in distractor_objs)
            h_max   = __import__("math").log(len(combined))
            tier    = self._tier_label(mdl)

            ann = EntropyAnnotation(
                n_objects=len(combined),
                min_desc_length=mdl,
                max_overlap=max_ov,
                h_uniform=h_max,
                ambiguity_tier=tier,
            )
            return Scene(
                id=scene_id,
                objects=combined,
                target_idx=target_idx,
                entropy_annotation=ann,
            )

        raise RuntimeError(
            f"Could not sample a valid scene after {cfg.max_rejection_iters} attempts. "
            f"Config: {cfg}"
        )

    @staticmethod
    def _tier_label(mdl: int) -> str:
        if mdl <= 1:
            return "low"
        if mdl == 2:
            return "medium"
        return "high"
