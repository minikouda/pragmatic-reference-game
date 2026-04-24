"""
ContrastiveSpeaker: describe the target by contrasting it with its most
confusable distractor.

Motivation
----------
When two objects share color and shape (e.g. two blue squares), a plain
"the blue square" is ambiguous.  A contrastive expression like
"the LARGER blue square" or "the blue square on the LEFT, not the one on the right"
resolves the ambiguity by explicitly calling out the distinguishing dimension.

Strategy
--------
1. Find the "foil" — the distractor most similar to the target (most shared features).
2. Find the first feature dimension where target and foil differ.
3. Generate a contrastive expression that foregrounds that dimension.

Three expression styles (escalating verbosity):
  minimal     : "the large blue square"   (just add the distinguishing feature)
  explicit    : "the large blue square, not the small one"
  relational  : "the blue square above the red circle"  (spatial contrast)

Mode
----
contrastive_rule  : purely rule-based
contrastive_vllm  : LLM refines the expression, given the foil as context
"""

from __future__ import annotations

from ..data.schema import FEATURE_KEYS, Object, Scene, Utterance
from .base import BaseSpeaker


# ── Foil selection ────────────────────────────────────────────────────────────

def _n_shared_features(a: Object, b: Object) -> int:
    af, bf = a.features(), b.features()
    return sum(af[k] == bf[k] for k in FEATURE_KEYS)


def _find_foil(target: Object, scene: Scene, target_idx: int) -> Object | None:
    """Most similar distractor (most shared features with target)."""
    candidates = [(i, o) for i, o in enumerate(scene.objects) if i != target_idx]
    if not candidates:
        return None
    return max(candidates, key=lambda x: _n_shared_features(target, x[1]))[1]


def _first_distinguishing_feature(target: Object, foil: Object) -> str | None:
    """First feature in canonical order where target and foil differ."""
    for key in ("size", "color", "shape", "location"):
        if target.features()[key] != foil.features()[key]:
            return key
    return None


def _spatial_relation(target: Object, foil: Object) -> str:
    """Describe where target is relative to foil."""
    dx = target.x_loc - foil.x_loc
    dy = target.y_loc - foil.y_loc  # positive = target is below

    if abs(dx) > abs(dy):
        return "to the right of" if dx > 0 else "to the left of"
    else:
        return "below" if dy > 0 else "above"


# ── Rule-based expression builders ───────────────────────────────────────────

def _build_expression(target: Object, foil: Object | None) -> str:
    f = target.features()
    base = f"{f['color']} {f['shape']}"

    if foil is None:
        return f"the {base}"

    key = _first_distinguishing_feature(target, foil)

    if key is None:
        # Identical features — use spatial relation as last resort
        rel = _spatial_relation(target, foil)
        foil_desc = f"the other {f['color']} {f['shape']}"
        return f"the {base} {rel} {foil_desc}"

    if key == "location":
        # Explicit spatial contrast
        return f"the {base} on the {f['location']}"

    if key == "size":
        return f"the {f['size']} {base}"

    if key == "color":
        return f"the {f['color']} {f['shape']}"

    if key == "shape":
        return f"the {f['color']} {f['shape']}"

    return f"the {base}"


class ContrastiveSpeaker(BaseSpeaker):
    """
    Rule-based contrastive speaker.
    Finds the most confusable distractor and surfaces the one feature that
    distinguishes the target from it.
    """

    @property
    def name(self) -> str:
        return "contrastive"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        target = scene.objects[target_idx]
        foil   = _find_foil(target, scene, target_idx)
        text   = _build_expression(target, foil)

        return Utterance(
            text=text,
            speaker_type=self.name,
            speaker_meta={"foil_id": foil.id if foil else None},
        )


_SYSTEM_CONTRASTIVE_VLLM = """\
You are a speaker in a visual reference game.

The TARGET object:
  color    : {color}
  shape    : {shape}
  size     : {size}
  location : {location}

All other objects in the scene:
{distractors}

Your task: describe the target using a CONTRASTIVE strategy.
Find the object most likely to be confused with the target, then highlight \
the one feature that sets the target apart from it.

Examples:
  "the LARGE blue square"         (if there is also a small blue square)
  "the blue square on the LEFT"   (if there is another blue square elsewhere)
  "the red circle, not the red triangle"

Rules:
- Identify the most similar distractor first (same color? same shape? same size?).
- Surface the ONE feature that distinguishes target from that distractor.
- Keep the expression SHORT (under 12 words).

Output ONLY the referring expression. No explanation."""


class ContrastiveVLLMSpeaker(BaseSpeaker):
    """
    LLM-based contrastive speaker: the model receives the full scene description
    and decides which distractor is the foil and which feature to contrast —
    no Python pre-computation of foil or distinguishing feature.
    """

    def __init__(self, client) -> None:
        self.client = client

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"contrastive-vllm({model})"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        from ..utils.llm_client import ChatMessage

        target = scene.objects[target_idx]
        feats  = target.features()

        distractors = "\n".join(
            f"  - {o.features()['size']} {o.features()['color']} {o.features()['shape']} at {o.features()['location']}"
            for i, o in enumerate(scene.objects) if i != target_idx
        )

        system = _SYSTEM_CONTRASTIVE_VLLM.format(**feats, distractors=distractors)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user",   content="Describe the target."),
            ],
            image_path=scene.image_path,
        )

        expression = raw.strip().strip('"').strip(".")
        return Utterance(
            text=expression,
            speaker_type=self.name,
            speaker_meta={"raw_response": raw},
        )
