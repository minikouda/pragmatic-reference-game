"""
OrdinalSpeaker: describe the target using ordinal / superlative expressions.

Motivation
----------
Ordinal expressions are highly unambiguous:
  "the leftmost circle", "the largest square", "the topmost object"
A VLM listener can execute these as simple visual search operations without
any ambiguity, even in crowded scenes.

Strategy
--------
For each feature dimension, compute whether the target is the extreme member:
  - spatial: leftmost / rightmost / topmost / bottommost (by x_loc / y_loc)
  - size:    smallest / largest (within a same-shape or same-color group)
  - count:   "the only triangle" (if the target's shape/color is unique)

Priority (use first one that applies and is unambiguous):
  1. "the only <shape>"          — if target is the sole object with its shape
  2. "the only <color> object"   — if target is the sole object with its color
  3. "the leftmost/rightmost/topmost/bottommost <shape>"
  4. "the largest/smallest <color> <shape>"
  5. Fallback: plain "the <color> <shape>"

All logic is rule-based — no LLM calls needed.
"""

from __future__ import annotations

from ..data.schema import Object, Scene, Utterance
from .base import BaseSpeaker


# ── Ordinal helpers ───────────────────────────────────────────────────────────

def _group(scene: Scene, target_idx: int, key: str) -> list[Object]:
    """All objects sharing the same value for `key` as the target."""
    target_val = scene.objects[target_idx].features()[key]
    return [o for o in scene.objects if o.features()[key] == target_val]


def _is_unique_in(scene: Scene, target_idx: int, key: str) -> bool:
    return len(_group(scene, target_idx, key)) == 1


def _ordinal_spatial(target: Object, group: list[Object]) -> str | None:
    """
    Return a spatial ordinal if the target is the extreme in the group.
    Uses image coordinates: x increases right, y increases downward.
    """
    if len(group) <= 1:
        return None

    xs = [o.x_loc for o in group]
    ys = [o.y_loc for o in group]

    if target.x_loc == min(xs) and xs.count(min(xs)) == 1:
        return "leftmost"
    if target.x_loc == max(xs) and xs.count(max(xs)) == 1:
        return "rightmost"
    if target.y_loc == min(ys) and ys.count(min(ys)) == 1:
        return "topmost"       # y=0 is top in image coords
    if target.y_loc == max(ys) and ys.count(max(ys)) == 1:
        return "bottommost"
    return None


_SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}

def _ordinal_size(target: Object, group: list[Object]) -> str | None:
    if len(group) <= 1:
        return None
    sizes = [_SIZE_ORDER.get(o.features()["size"], 1) for o in group]
    t_size = _SIZE_ORDER.get(target.features()["size"], 1)
    if t_size == max(sizes) and sizes.count(max(sizes)) == 1:
        return "largest"
    if t_size == min(sizes) and sizes.count(min(sizes)) == 1:
        return "smallest"
    return None


# ── Speaker ───────────────────────────────────────────────────────────────────

class OrdinalSpeaker(BaseSpeaker):
    """
    Rule-based speaker using ordinal / superlative / uniqueness expressions.
    No LLM calls — purely symbolic.
    """

    @property
    def name(self) -> str:
        return "ordinal"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        target = scene.objects[target_idx]
        f      = target.features()
        meta: dict = {}

        # 1. Unique shape?
        if _is_unique_in(scene, target_idx, "shape"):
            text = f"the only {f['shape']}"
            meta["strategy"] = "unique_shape"
            return Utterance(text=text, speaker_type=self.name, speaker_meta=meta)

        # 2. Unique color?
        if _is_unique_in(scene, target_idx, "color"):
            text = f"the {f['color']} one"
            meta["strategy"] = "unique_color"
            return Utterance(text=text, speaker_type=self.name, speaker_meta=meta)

        # 3. Spatial ordinal within same-shape group
        shape_group = _group(scene, target_idx, "shape")
        spatial_ord = _ordinal_spatial(target, shape_group)
        if spatial_ord:
            text = f"the {spatial_ord} {f['shape']}"
            meta["strategy"] = f"spatial_ordinal({spatial_ord})"
            return Utterance(text=text, speaker_type=self.name, speaker_meta=meta)

        # 4. Size ordinal within same-color+shape group
        color_shape_group = [
            o for o in scene.objects
            if o.features()["color"] == f["color"] and o.features()["shape"] == f["shape"]
        ]
        size_ord = _ordinal_size(target, color_shape_group)
        if size_ord:
            text = f"the {size_ord} {f['color']} {f['shape']}"
            meta["strategy"] = f"size_ordinal({size_ord})"
            return Utterance(text=text, speaker_type=self.name, speaker_meta=meta)

        # 5. Spatial ordinal across all objects
        all_spatial = _ordinal_spatial(target, scene.objects)
        if all_spatial:
            text = f"the {all_spatial} {f['color']} {f['shape']}"
            meta["strategy"] = f"global_spatial({all_spatial})"
            return Utterance(text=text, speaker_type=self.name, speaker_meta=meta)

        # 6. Fallback: color + shape (+ location if still ambiguous)
        candidates = [
            o for o in scene.objects
            if o.features()["color"] == f["color"] and o.features()["shape"] == f["shape"]
        ]
        if len(candidates) > 1:
            text = f"the {f['color']} {f['shape']} on the {f['location']}"
            meta["strategy"] = "color_shape_location"
        else:
            text = f"the {f['color']} {f['shape']}"
            meta["strategy"] = "color_shape"

        return Utterance(text=text, speaker_type=self.name, speaker_meta=meta)
