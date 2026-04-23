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
- `overlap_mode` controls *physical* (pixel) overlap between rendered
  objects:
    "none"  — objects must not visually overlap at all (default)
    "allow" — no constraint (any overlap is fine)
    "force" — every object must physically overlap at least one other,
              with each object's center pixel still visible (never covered
              by another object's bounding box)
- The generator rejects and resamples scenes that don't meet the requested
  constraints (rejection sampling with a max-iteration guard).
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .schema import (
    COLORS, SHAPES, SIZES, FEATURE_KEYS, FEATURE_VOCAB,
    Object, Scene, EntropyAnnotation, loc_label,
)
from .renderer import CANVAS_W, CANVAS_H, MARGIN, SIZE_SCALE


# ── Physical-overlap geometry helpers ────────────────────────────────────────

_SIZE_RAW: dict[str, int] = {"small": 8, "medium": 12, "large": 16}


def _to_px(x_loc: int, y_loc: int) -> tuple[int, int]:
    """Data coords [0,100]² → pixel (px, py). Mirrors renderer._to_pixel."""
    draw_w = CANVAS_W - 2 * MARGIN
    draw_h = CANVAS_H - 2 * MARGIN
    px = int(MARGIN + x_loc / 100 * draw_w)
    py = int((CANVAS_H - MARGIN) - y_loc / 100 * draw_h)
    return px, py


def _aabb(x_loc: int, y_loc: int, size_str: str) -> tuple[int, int, int, int]:
    """Pixel-space AABB (x0, y0, x1, y1) for an object with the given size."""
    cx, cy = _to_px(x_loc, y_loc)
    r = _SIZE_RAW[size_str] * SIZE_SCALE
    return cx - r, cy - r, cx + r, cy + r


def _boxes_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _center_in_box(cx: int, cy: int, box: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= cx <= x1 and y0 <= cy <= y1


# Integer bucket bounds aligned with schema.loc_label thresholds
# (<0.33 → left/top, >0.66 → right/bottom, else center; clamped to [10,90]).
_COL_BOUNDS: dict[str, tuple[int, int]] = {
    "left":   (10, 32),
    "center": (33, 66),
    "right":  (67, 90),
}
_ROW_BOUNDS: dict[str, tuple[int, int]] = {
    "top":    (10, 32),
    "center": (33, 66),
    "bottom": (67, 90),
}


def _bucket_bounds(label: str | None) -> tuple[int, int, int, int]:
    """Inclusive (x_lo, x_hi, y_lo, y_hi) sampling bounds for a location label."""
    if label is None:
        return 10, 90, 10, 90
    if label == "center":
        xlo, xhi = _COL_BOUNDS["center"]; ylo, yhi = _ROW_BOUNDS["center"]
    elif "-" in label:
        row, col = label.split("-")
        xlo, xhi = _COL_BOUNDS[col]; ylo, yhi = _ROW_BOUNDS[row]
    elif label in _COL_BOUNDS:
        xlo, xhi = _COL_BOUNDS[label]; ylo, yhi = _ROW_BOUNDS["center"]
    elif label in _ROW_BOUNDS:
        xlo, xhi = _COL_BOUNDS["center"]; ylo, yhi = _ROW_BOUNDS[label]
    else:
        raise ValueError(f"Unknown location label: {label!r}")
    return xlo, xhi, ylo, yhi


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
    overlap_mode       : physical (pixel) overlap policy
                         "none"  — no object bounding boxes may intersect
                         "allow" — no constraint
                         "force" — every object must overlap ≥1 other,
                                   and no object's center may be covered
    seed               : RNG seed (None = non-deterministic)
    max_rejection_iters: rejection sampling cap before raising
    """
    n_objects:           int         = 4
    min_desc_length:     int | None  = None
    max_overlap:         int | None  = None
    ambiguity_tier:      str | None  = None   # "low" | "medium" | "high"
    overlap_mode:        str         = "none"  # "none" | "allow" | "force"
    seed:                int | None  = 42
    max_rejection_iters: int         = 2000

    def __post_init__(self) -> None:
        # Resolve ambiguity_tier → min_desc_length
        _tier_map = {"low": 1, "medium": 2, "high": 3}
        if self.ambiguity_tier is not None:
            if self.ambiguity_tier not in _tier_map:
                raise ValueError(f"ambiguity_tier must be one of {list(_tier_map)}")
            self.min_desc_length = _tier_map[self.ambiguity_tier]
        if self.overlap_mode not in ("none", "allow", "force"):
            raise ValueError(
                f"overlap_mode must be 'none' | 'allow' | 'force', "
                f"got {self.overlap_mode!r}"
            )


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
        """Yield `n` scenes (no images). Use generate_with_images for PNG output."""
        for i in range(n):
            yield self._sample_scene(scene_id=str(i))

    def generate_with_images(
        self,
        n: int,
        out_dir: str | Path,
        prefix: str = "scene",
    ) -> list[Scene]:
        """
        Generate `n` scenes and render each to a PNG file.

        Images are written to out_dir/{prefix}_{i}.png and the scene's
        image_path field is set to that relative path.
        """
        from .renderer import render_scene_from_objects
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes: list[Scene] = []
        for i in range(n):
            scene      = self._sample_scene(scene_id=str(i))
            img_path   = out_dir / f"{prefix}_{i}.png"
            render_scene_from_objects(scene.objects, img_path)
            scene.image_path = str(img_path)
            scenes.append(scene)
        return scenes

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
                overlap_mode=self.config.overlap_mode,
                seed=(self.config.seed or 0) + hash(tier) % 1000,
                max_rejection_iters=self.config.max_rejection_iters,
            )
            sub = SceneGenerator(cfg)
            scenes.extend(sub.generate(n_per_tier))
        return scenes

    # ── Internals ────────────────────────────────────────────────────────────

    def _sample_position(
        self,
        size_str: str,
        placed: list[Object],
        bucket: str | None = None,
        max_tries: int = 300,
    ) -> tuple[int, int] | None:
        """
        Sample (x_loc, y_loc) for a new object of `size_str`, honouring
        self.config.overlap_mode against already-placed objects. Returns
        None if no valid placement is found within `max_tries`.

        `bucket` (a location label) restricts sampling to the spatial
        region implied by that label — used by the constructive path when
        "location" is a locked feature.
        """
        mode = self.config.overlap_mode
        xlo, xhi, ylo, yhi = _bucket_bounds(bucket)
        r_new = _SIZE_RAW[size_str] * SIZE_SCALE

        for _ in range(max_tries):
            # Anchor-biased sampling for "force" mode: pick a random placed
            # object and sample a position whose AABB overlaps it.
            if mode == "force" and placed:
                anchor = self._rng.choice(placed)
                r_anc  = _SIZE_RAW[anchor.size] * SIZE_SCALE
                acx_px, acy_px = _to_px(anchor.x_loc, anchor.y_loc)
                span = r_new + r_anc - 1
                new_cx_px = acx_px + self._rng.randint(-span, span)
                new_cy_px = acy_px + self._rng.randint(-span, span)
                draw_w = CANVAS_W - 2 * MARGIN
                draw_h = CANVAS_H - 2 * MARGIN
                x_loc = int(round((new_cx_px - MARGIN) / draw_w * 100))
                y_loc = int(round(((CANVAS_H - MARGIN) - new_cy_px) / draw_h * 100))
            else:
                x_loc = self._rng.randint(xlo, xhi)
                y_loc = self._rng.randint(ylo, yhi)

            if not (xlo <= x_loc <= xhi and ylo <= y_loc <= yhi):
                continue

            cand_box = _aabb(x_loc, y_loc, size_str)
            cand_cx_px, cand_cy_px = _to_px(x_loc, y_loc)

            if mode == "allow":
                return x_loc, y_loc

            if mode == "none":
                if any(
                    _boxes_intersect(cand_box, _aabb(p.x_loc, p.y_loc, p.size))
                    for p in placed
                ):
                    continue
                return x_loc, y_loc

            # mode == "force"
            if not placed:
                return x_loc, y_loc
            overlaps_any = False
            centers_ok   = True
            for p in placed:
                p_box = _aabb(p.x_loc, p.y_loc, p.size)
                p_cx_px, p_cy_px = _to_px(p.x_loc, p.y_loc)
                if _boxes_intersect(cand_box, p_box):
                    overlaps_any = True
                if (_center_in_box(cand_cx_px, cand_cy_px, p_box)
                        or _center_in_box(p_cx_px, p_cy_px, cand_box)):
                    centers_ok = False
                    break
            if centers_ok and overlaps_any:
                return x_loc, y_loc
        return None

    def _sample_object(
        self,
        obj_id: str,
        placed: list[Object],
        bucket: str | None = None,
    ) -> Object | None:
        """
        Sample a random object whose position respects `overlap_mode` vs
        already-`placed` objects. Returns None if placement fails.
        """
        size  = self._rng.choice(SIZES)
        pos   = self._sample_position(size, placed, bucket=bucket)
        if pos is None:
            return None
        x_loc, y_loc = pos
        return Object(
            id=obj_id,
            color=self._rng.choice(COLORS),
            shape=self._rng.choice(SHAPES),
            size=size,
            location=loc_label(x_loc, y_loc),
            x_loc=x_loc,
            y_loc=y_loc,
        )

    def _feature_overlap(self, a: Object, b: Object) -> int:
        """Number of feature dimensions with the same value."""
        fa, fb = a.features(), b.features()
        return sum(fa[k] == fb[k] for k in FEATURE_KEYS)

    def _compute_min_desc_length(self, target: Object, distractors: list[Object]) -> int:
        """Wrapper around Scene helper; avoids constructing a full Scene."""
        scene = Scene(id="_tmp", objects=[target] + distractors, target_idx=0)
        return scene.min_description_length()

    def _meets_overlap_constraint(self, objs: list[Object]) -> bool:
        """Final audit of pairwise physical overlap per `overlap_mode`."""
        mode = self.config.overlap_mode
        if mode == "allow":
            return True
        n       = len(objs)
        boxes   = [_aabb(o.x_loc, o.y_loc, o.size) for o in objs]
        centers = [_to_px(o.x_loc, o.y_loc)        for o in objs]
        pair_overlap = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if _boxes_intersect(boxes[i], boxes[j]):
                    pair_overlap[i][j] = pair_overlap[j][i] = True
        if mode == "none":
            return not any(
                pair_overlap[i][j] for i in range(n) for j in range(i + 1, n)
            )
        # mode == "force"
        for i in range(n):
            if not any(pair_overlap[i]):
                return False
        for i in range(n):
            cx, cy = centers[i]
            for j in range(n):
                if i == j:
                    continue
                if _center_in_box(cx, cy, boxes[j]):
                    return False
        return True

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

        if not self._meets_overlap_constraint([target] + distractors):
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
            placed: list[Object] = []
            target = self._sample_object("T", placed)
            if target is None:
                continue
            placed.append(target)

            distractors: list[Object] = []
            failed = False
            for i in range(cfg.n_objects - 1):
                d = self._sample_object(f"D{i}", placed)
                if d is None:
                    failed = True
                    break
                distractors.append(d)
                placed.append(d)
            if failed:
                continue

            feature_sets = [frozenset(o.features().items()) for o in placed]
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
            placed: list[Object] = []
            target = self._sample_object("T", placed)
            if target is None:
                continue
            placed.append(target)
            t_feats = target.features()

            distractors: list[Object] = []
            failed = False
            for i in range(n_d):
                locked_keys = set(all_triples[i % len(all_triples)])
                # Non-location features: copy if locked, else resample to differ
                d_feats: dict[str, str] = {}
                for k in ("color", "shape", "size"):
                    if k in locked_keys:
                        d_feats[k] = t_feats[k]
                    else:
                        vocab   = FEATURE_VOCAB[k]
                        choices = [v for v in vocab if v != t_feats[k]]
                        d_feats[k] = self._rng.choice(choices) if choices else t_feats[k]
                # Location: if locked, constrain sampling to target's bucket.
                # Otherwise, sample anywhere and derive the label from the pos.
                bucket = t_feats["location"] if "location" in locked_keys else None
                pos = self._sample_position(d_feats["size"], placed, bucket=bucket)
                if pos is None:
                    failed = True
                    break
                x_loc, y_loc = pos
                d_feats["location"] = loc_label(x_loc, y_loc)
                d = Object(id=f"D{i}", x_loc=x_loc, y_loc=y_loc, **d_feats)
                distractors.append(d)
                placed.append(d)
            if failed:
                continue

            feature_sets = [frozenset(o.features().items()) for o in placed]
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
                   size=o.size, location=o.location,
                   x_loc=o.x_loc, y_loc=o.y_loc)
            for i, o in enumerate(all_objs)
        ]
        target_idx = next(
            i for i, o in enumerate(combined)
            if o.color == target.color and o.shape == target.shape
            and o.size == target.size and o.x_loc == target.x_loc
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

    @staticmethod
    def _tier_label(mdl: int) -> str:
        """Map min_desc_length to a human-readable tier (lower bound semantics)."""
        if mdl <= 1:
            return "low"
        if mdl == 2:
            return "medium"
        return "high"   # mdl >= 3
