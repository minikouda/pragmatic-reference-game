"""
Core data structures for reference game scenes.

Supports both symbolic (rule-based) and visual (VLLM) pipelines:
- Symbolic attributes (color, shape, size, location) are used by Literal/RSA models.
- Pixel coordinates (x_loc, y_loc) and image_path are used by VLLM models.

Vocabulary is aligned with the reference_game_dataset:
  shapes : circle, square, triangle
  colors : black, blue, green, red, yellow
  sizes  : small (8px), medium (12px), large (16px)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import FrozenSet


# ── Vocabulary (aligned with reference_game_dataset) ───────────────────────

COLORS    = ("black", "blue", "green", "red", "yellow")
SHAPES    = ("circle", "square", "triangle")
SIZES     = ("small", "medium", "large")
LOCATIONS = ("top-left", "top-right", "bottom-left", "bottom-right",
             "top", "bottom", "left", "right", "center")

# Numeric pixel size → categorical label
SIZE_MAP: dict[int, str] = {8: "small", 12: "medium", 16: "large"}

FEATURE_VOCAB: dict[str, tuple[str, ...]] = {
    "color":    COLORS,
    "shape":    SHAPES,
    "size":     SIZES,
    "location": LOCATIONS,
}
FEATURE_KEYS = tuple(FEATURE_VOCAB.keys())


def loc_label(x: float, y: float, canvas: int = 100) -> str:
    """Map pixel (x, y) on a [0, canvas]² grid to a coarse spatial label."""
    xn, yn = x / canvas, y / canvas
    col = "left" if xn < 0.33 else ("right" if xn > 0.66 else "center")
    row = "top"  if yn < 0.33 else ("bottom" if yn > 0.66 else "center")
    if row == col == "center":
        return "center"
    if row == "center":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


# ── Object ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Object:
    """
    A single object in a reference game scene.

    Symbolic attributes (color, shape, size, location) are used by rule-based
    and RSA models.  Pixel coordinates (x_loc, y_loc) are stored for
    reference by VLLM prompts and scene rendering.
    """
    id:       str
    color:    str
    shape:    str        # "circle" | "square" | "triangle"
    size:     str        # "small" | "medium" | "large"
    location: str        # coarse spatial label derived from (x_loc, y_loc)
    x_loc:    int = 0    # pixel x-coordinate in the rendered image
    y_loc:    int = 0    # pixel y-coordinate in the rendered image

    def __post_init__(self) -> None:
        assert self.color    in COLORS,    f"Unknown color: {self.color}"
        assert self.shape    in SHAPES,    f"Unknown shape: {self.shape}"
        assert self.size     in SIZES,     f"Unknown size:  {self.size}"
        assert self.location in LOCATIONS, f"Unknown location: {self.location}"

    def features(self) -> dict[str, str]:
        """Symbolic feature dict used by Literal/RSA models."""
        return {"color": self.color, "shape": self.shape,
                "size": self.size, "location": self.location}

    def feature_set(self) -> FrozenSet[str]:
        """Flat set of all feature values, used for utterance token matching."""
        return frozenset(self.features().values())

    def natural_description(self) -> str:
        return f"the {self.size} {self.color} {self.shape} on the {self.location}"

    def __str__(self) -> str:
        return self.natural_description()


# ── Scene ───────────────────────────────────────────────────────────────────

@dataclass
class Scene:
    """
    A reference game scene: a rendered image + structured object list + target.

    image_path         : path to the PNG rendered by the dataset generator.
                         None for synthetically generated scenes without images.
    entropy_annotation : complexity metadata (set by scene generator).
    """
    id:         str
    objects:    list[Object]
    target_idx: int
    image_path: str | None = field(default=None, repr=False)
    entropy_annotation: EntropyAnnotation | None = field(default=None, repr=False)

    @property
    def target(self) -> Object:
        return self.objects[self.target_idx]

    @property
    def distractors(self) -> list[Object]:
        return [o for i, o in enumerate(self.objects) if i != self.target_idx]

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def min_description_length(self) -> int:
        """
        Minimum number of symbolic features needed to uniquely identify the target.
        Returns len(FEATURE_KEYS)+1 if the target is indistinguishable (degenerate).
        """
        from itertools import combinations
        target_feats = self.target.features()
        for r in range(1, len(FEATURE_KEYS) + 1):
            for combo in combinations(FEATURE_KEYS, r):
                target_vals = {k: target_feats[k] for k in combo}
                if not any(
                    all(d.features()[k] == v for k, v in target_vals.items())
                    for d in self.distractors
                ):
                    return r
        return len(FEATURE_KEYS) + 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target_idx": self.target_idx,
            "image_path": self.image_path,
            "objects": [
                {**o.features(), "id": o.id, "x_loc": o.x_loc, "y_loc": o.y_loc}
                for o in self.objects
            ],
            "entropy_annotation": self.entropy_annotation.to_dict()
            if self.entropy_annotation else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        objects = [
            Object(
                id=o["id"], color=o["color"], shape=o["shape"],
                size=o["size"], location=o["location"],
                x_loc=o.get("x_loc", 0), y_loc=o.get("y_loc", 0),
            )
            for o in d["objects"]
        ]
        ann = EntropyAnnotation.from_dict(d["entropy_annotation"]) \
              if d.get("entropy_annotation") else None
        return cls(
            id=d["id"], objects=objects, target_idx=d["target_idx"],
            image_path=d.get("image_path"), entropy_annotation=ann,
        )


# ── EntropyAnnotation ────────────────────────────────────────────────────────

@dataclass
class EntropyAnnotation:
    n_objects:       int
    min_desc_length: int
    max_overlap:     int
    h_uniform:       float
    ambiguity_tier:  str   # "low" | "medium" | "high"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> EntropyAnnotation:
        return cls(**d)


# ── Utterance ────────────────────────────────────────────────────────────────

@dataclass
class Utterance:
    """The output of a speaker: a referring expression + metadata."""
    text:          str
    speaker_type:  str
    feature_combo: tuple[str, ...] | None = None
    speaker_meta:  dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


# ── ListenerOutput ───────────────────────────────────────────────────────────

@dataclass
class ListenerOutput:
    """
    Listener posterior P(object_i | utterance) over all objects in the scene.

    posterior        : probability distribution, length = n_objects, sums to 1
    predicted_idx    : argmax index
    listener_type    : identifier string
    listener_meta    : model-specific metadata (e.g. raw VLM response, RSA α)
    """
    posterior:     list[float]
    predicted_idx: int
    listener_type: str
    listener_meta: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.posterior[self.predicted_idx]

    def referential_entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.posterior if p > 0)


# ── ClarificationDecision ────────────────────────────────────────────────────

@dataclass
class ClarificationDecision:
    """Expected-utility clarification decision from the Cost-Aware Listener."""
    action:      str       # "commit" | "ask"
    eu_commit:   float
    eu_ask:      float
    cost_c:      float
    question:    str | None
    base_output: ListenerOutput


# ── EvalRecord ───────────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    """One row in the evaluation results table."""
    scene_id:        str
    speaker_type:    str
    listener_type:   str
    cost_c:          float
    utterance:       str
    action:          str
    predicted_idx:   int
    target_idx:      int
    correct:         bool
    eu_commit:       float
    eu_ask:          float
    entropy:         float
    brier_score:     float
    min_desc_length: int | None
    ambiguity_tier:  str | None
    pred_x:          float | None = None   # predicted x in bottom-left origin
    pred_y:          float | None = None   # predicted y in bottom-left origin
