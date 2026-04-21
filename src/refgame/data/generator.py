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
    COLORS, SHAPES, SIZES, LOCATIONS, FEATURE_KEYS, FEATURE_VOCAB,
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
    max_rejection_iters: int         = 2000

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

        # Treat min_desc_length as a lower bound: scenes that require *more*
        # features are at least as hard, so they belong in the same tier.
        if cfg.min_desc_length is not None and mdl < cfg.min_desc_length:
            return False
        if cfg.max_overlap is not None and max_ov > cfg.max_overlap:
            return False

        # Sanity: target must be uniquely identifiable by full description
        if mdl > len(FEATURE_KEYS):
            return False

        return True

    def _sample_scene(self, scene_id: str) -> Scene:
        cfg = self.config
        # For high ambiguity (MDL >= 3), pure rejection sampling almost never
        # succeeds with few objects because covering all C(4,2)=6 feature-pairs
        # requires deliberately structured overlap.  Use the constructive path.
        if cfg.min_desc_length is not None and cfg.min_desc_length >= 3:
            return self._sample_scene_constructive(scene_id)

        for _ in range(cfg.max_rejection_iters):
            target = self._sample_object("T")
            distractors = [
                self._sample_object(f"D{i}")
                for i in range(cfg.n_objects - 1)
            ]
            all_objs = [target] + distractors
            feature_sets = [frozenset(o.features().items()) for o in all_objs]
            if len(feature_sets) != len(set(feature_sets)):
                continue
            if not self._meets_constraints(target, distractors):
                continue
            return self._finalize_scene(scene_id, target, distractors)

        raise RuntimeError(
            f"Could not sample a valid scene after {cfg.max_rejection_iters} attempts. "
            f"Config: {cfg}"
        )

    def _sample_scene_constructive(self, scene_id: str) -> Scene:
        """
        Build a high-ambiguity scene by design rather than rejection sampling.

        For MDL >= 3 we must block every one of the C(4,2)=6 feature-pairs from
        distinguishing the target.  Each distractor that matches the target on k
        features blocks C(k,2) pairs.  With k=3 each distractor blocks 3 pairs,
        so three distractors with the right triples block all 6 exactly:

            D0 locks {color, shape, size}     → blocks {cs, csz, ssz}
            D1 locks {shape, size, location}  → blocks {ssz, sl, szl}
            D2 locks {color, size, location}  → blocks {csz, cl, szl} + {cs}

        All 6 pairs are covered.  Additional distractors (n_objects > 4)
        use further triples chosen round-robin from all C(4,3)=4 triples.

        For each distractor the one un-locked feature dimension is randomized
        to a value != target's, ensuring the full description still identifies
        the target uniquely (MDL <= 4 always).
        """
        from itertools import combinations as _comb

        cfg = self.config
        n_d = cfg.n_objects - 1

        # All C(4,3)=4 triples of feature keys, ordered to cover all 6 pairs
        # in the fewest distractors.  The first 3 suffice for n_d=3.
        all_triples = list(_comb(FEATURE_KEYS, 3))   # 4 triples

        for _ in range(cfg.max_rejection_iters):
            target  = self._sample_object("T")
            t_feats = target.features()

            distractors: list[Object] = []
            for i in range(n_d):
                locked_keys = set(all_triples[i % len(all_triples)])
                new_feats: dict[str, str] = {}
                for k in FEATURE_KEYS:
                    if k in locked_keys:
                        new_feats[k] = t_feats[k]
                    else:
                        vocab   = FEATURE_VOCAB[k]
                        choices = [v for v in vocab if v != t_feats[k]]
                        new_feats[k] = self._rng.choice(choices) if choices else t_feats[k]
                distractors.append(Object(id=f"D{i}", **new_feats))

            all_objs = [target] + distractors
            feature_sets = [frozenset(o.features().items()) for o in all_objs]
            if len(feature_sets) != len(set(feature_sets)):
                continue
            if not self._meets_constraints(target, distractors):
                continue
            return self._finalize_scene(scene_id, target, distractors)

        raise RuntimeError(
            f"Could not sample a valid high-ambiguity scene after "
            f"{cfg.max_rejection_iters} attempts. Config: {cfg}"
        )

    def _finalize_scene(
        self, scene_id: str, target: Object, distractors: list[Object]
    ) -> Scene:
        """Shuffle object order, assign clean IDs, compute annotation."""
        import math as _math
        all_objs = [target] + distractors
        self._rng.shuffle(all_objs)
        combined = [
            Object(id=f"obj_{i}", color=o.color, shape=o.shape,
                   size=o.size, location=o.location)
            for i, o in enumerate(all_objs)
        ]
        target_idx = next(
            i for i, o in enumerate(combined)
            if o.color == target.color and o.shape == target.shape
            and o.size == target.size and o.location == target.location
        )
        target_obj      = combined[target_idx]
        distractor_objs = [o for i, o in enumerate(combined) if i != target_idx]
        mdl    = self._compute_min_desc_length(target_obj, distractor_objs)
        max_ov = max(self._feature_overlap(target_obj, d) for d in distractor_objs)
        ann = EntropyAnnotation(
            n_objects=len(combined),
            min_desc_length=mdl,
            max_overlap=max_ov,
            h_uniform=_math.log(len(combined)),
            ambiguity_tier=self._tier_label(mdl),
        )
        return Scene(id=scene_id, objects=combined, target_idx=target_idx,
                     entropy_annotation=ann)

        raise RuntimeError(
            f"Could not sample a valid scene after {cfg.max_rejection_iters} attempts. "
            f"Config: {cfg}"
        )

    @staticmethod
    def _tier_label(mdl: int) -> str:
        """Map min_desc_length to a human-readable tier (lower bound semantics)."""
        if mdl <= 1:
            return "low"
        if mdl == 2:
            return "medium"
        return "high"   # mdl >= 3
