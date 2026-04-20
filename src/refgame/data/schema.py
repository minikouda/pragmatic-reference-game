"""
Core data structures for reference game scenes.

All objects are symbolic: a fixed vocabulary of visual attributes
(color, shape, size, location). This keeps the game tractable for
both rule-based and neural listeners while allowing precise control
over referential entropy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import FrozenSet, Sequence


# ── Vocabulary ─────────────────────────────────────────────────────────────

COLORS    = ("red", "blue", "green", "yellow", "purple", "orange", "pink", "brown")
SHAPES    = ("circle", "square", "triangle", "diamond", "star", "pentagon", "cross")
SIZES     = ("small", "medium", "large")
LOCATIONS = ("top-left", "top-right", "bottom-left", "bottom-right",
             "top", "bottom", "left", "right", "center")

FEATURE_VOCAB: dict[str, tuple[str, ...]] = {
    "color":    COLORS,
    "shape":    SHAPES,
    "size":     SIZES,
    "location": LOCATIONS,
}
FEATURE_KEYS = tuple(FEATURE_VOCAB.keys())


# ── Object ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Object:
    """
    A single object in a reference game scene.

    Attributes are discrete and drawn from fixed vocabularies so that
    feature overlap can be computed exactly (no embedding similarity needed).
    """
    id:       str
    color:    str
    shape:    str
    size:     str
    location: str

    def __post_init__(self) -> None:
        assert self.color    in COLORS,    f"Unknown color: {self.color}"
        assert self.shape    in SHAPES,    f"Unknown shape: {self.shape}"
        assert self.size     in SIZES,     f"Unknown size:  {self.size}"
        assert self.location in LOCATIONS, f"Unknown location: {self.location}"

    # ── Convenience ────────────────────────────────────────────────────────

    def features(self) -> dict[str, str]:
        return {"color": self.color, "shape": self.shape,
                "size": self.size, "location": self.location}

    def feature_set(self) -> FrozenSet[str]:
        """Flat set of all feature values, used for utterance matching."""
        return frozenset(self.features().values())

    def natural_description(self) -> str:
        return f"the {self.size} {self.color} {self.shape} on the {self.location}"

    def __str__(self) -> str:
        return self.natural_description()


# ── Scene ───────────────────────────────────────────────────────────────────

@dataclass
class Scene:
    """
    A reference game scene: a set of objects + a designated target.

    The generator fills `entropy_annotation` with ground-truth complexity
    measures so we can stratify evaluation results by difficulty.
    """
    id:         str
    objects:    list[Object]
    target_idx: int

    # Set by generator; can be None for hand-crafted scenes
    entropy_annotation: EntropyAnnotation | None = field(default=None, repr=False)

    # ── Target access ───────────────────────────────────────────────────────

    @property
    def target(self) -> Object:
        return self.objects[self.target_idx]

    @property
    def distractors(self) -> list[Object]:
        return [o for i, o in enumerate(self.objects) if i != self.target_idx]

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    # ── Ground-truth combinatorics ───────────────────────────────────────────

    def min_description_length(self) -> int:
        """
        Minimum number of features needed to uniquely identify the target.

        Iterates over all subsets of FEATURE_KEYS in size order and returns
        the size of the smallest subset that rules out every distractor.
        Returns len(FEATURE_KEYS)+1 if no subset suffices (degenerate scene).
        """
        from itertools import combinations
        target_feats = self.target.features()
        for r in range(1, len(FEATURE_KEYS) + 1):
            for combo in combinations(FEATURE_KEYS, r):
                target_vals = {k: target_feats[k] for k in combo}
                # Check whether any distractor matches all features in combo
                if not any(
                    all(d.features()[k] == v for k, v in target_vals.items())
                    for d in self.distractors
                ):
                    return r
        return len(FEATURE_KEYS) + 1

    def uniform_referential_entropy(self) -> float:
        """H_max = log(N): entropy if listener has no information."""
        return math.log(self.n_objects)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target_idx": self.target_idx,
            "objects": [o.features() | {"id": o.id, "location": o.location} for o in self.objects],
            "entropy_annotation": self.entropy_annotation.to_dict()
            if self.entropy_annotation else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        objects = [
            Object(id=o["id"], color=o["color"], shape=o["shape"],
                   size=o["size"], location=o["location"])
            for o in d["objects"]
        ]
        ann = EntropyAnnotation.from_dict(d["entropy_annotation"]) \
              if d.get("entropy_annotation") else None
        return cls(id=d["id"], objects=objects, target_idx=d["target_idx"],
                   entropy_annotation=ann)


# ── Entropy annotation (set by generator) ───────────────────────────────────

@dataclass
class EntropyAnnotation:
    """
    Metadata attached by the generator describing the scene's difficulty.

    Fields
    ------
    n_objects          : total objects in scene
    min_desc_length    : min features to uniquely identify target (1–4)
    max_overlap        : max # of features any single distractor shares with target
    h_uniform          : log(N), upper bound on referential entropy
    ambiguity_tier     : bucketed difficulty label ("low" / "medium" / "high")
    """
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
    """
    The output of a speaker: a natural-language referring expression
    plus optional metadata produced by the speaker model.
    """
    text:          str
    speaker_type:  str              # "literal" | "rsa" | "llm"
    feature_combo: tuple[str, ...] | None = None   # which features were used (rule-based)
    speaker_meta:  dict             = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


# ── ListenerOutput ───────────────────────────────────────────────────────────

@dataclass
class ListenerOutput:
    """
    The output of a listener given a scene + utterance.

    posterior        : P(object_i | utterance), sums to 1, length = n_objects
    predicted_idx    : argmax of posterior
    listener_type    : "literal" | "rsa" | "cost_aware"
    listener_meta    : arbitrary extra information (e.g. RSA α used)
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
    """
    The output of the Cost-Aware Listener layer.

    action           : "commit" or "ask"
    eu_commit        : E[U | commit] = max_i P(t_i | u)
    eu_ask           : E[U | ask]   = 1 - c  (assumes question resolves perfectly)
    cost_c           : clarification cost used
    question         : suggested clarification question (if action == "ask")
    base_output      : ListenerOutput from the underlying listener
    """
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
    action:          str        # "commit" | "ask"
    predicted_idx:   int
    target_idx:      int
    correct:         bool
    eu_commit:       float
    eu_ask:          float
    entropy:         float
    brier_score:     float
    min_desc_length: int | None
    ambiguity_tier:  str | None
