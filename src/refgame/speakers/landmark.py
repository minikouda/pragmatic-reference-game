"""
LandmarkSpeaker: describe the target relative to a visually unique "landmark" object.

Strategy
--------
When the target shares its most salient features (color, shape) with distractors,
it is hard to describe uniquely by its own properties alone.  Instead, find a
landmark — a distractor that is maximally unique (few or no shared features with
others) and spatially close to the target — and express the target's position
relative to it.

Example: target = small black triangle (ambiguous, 2 other black objects)
         landmark = yellow square (unique color in scene)
         → "the small black triangle to the left of the yellow square"

Landmark selection
------------------
Score each distractor by:
  uniqueness = number of features with no other object sharing that value
  proximity  = 1 / (euclidean distance to target + 1)
  score = uniqueness + 0.5 * proximity   (uniqueness weighted higher)

The best landmark is the distractor with the highest score.

Direction computation
---------------------
Compass direction from landmark to target (8-way):
  dx = target.x - landmark.x  (positive = target is to the right)
  dy = target.y - landmark.y  (positive = target is below in image coords)
→ map to "above/below/left/right/upper-left/..." English phrase.

Two modes
---------
landmark_only : purely rule-based, no LLM
landmark_vllm : uses an LLM to rephrase the rule-generated expression more
                naturally, with the image for grounding
"""

from __future__ import annotations

import math

from ..data.schema import FEATURE_KEYS, Object, Scene, Utterance
from .base import BaseSpeaker


# ── Landmark selection ────────────────────────────────────────────────────────

def _uniqueness(obj: Object, all_objects: list[Object]) -> int:
    """Count features of obj whose value no other object shares."""
    others = [o for o in all_objects if o.id != obj.id]
    count = 0
    for key in FEATURE_KEYS:
        val = obj.features()[key]
        if not any(o.features()[key] == val for o in others):
            count += 1
    return count


def _distance(a: Object, b: Object) -> float:
    return math.sqrt((a.x_loc - b.x_loc) ** 2 + (a.y_loc - b.y_loc) ** 2)


def _best_landmark(target: Object, scene: Scene, target_idx: int) -> Object | None:
    candidates = [o for i, o in enumerate(scene.objects) if i != target_idx]
    if not candidates:
        return None

    def score(obj: Object) -> float:
        u = _uniqueness(obj, scene.objects)
        d = _distance(target, obj)
        return u + 0.5 / (d + 1.0)

    return max(candidates, key=score)


def _direction(target: Object, landmark: Object) -> str:
    """Compass direction FROM landmark TO target (image coords: y increases downward)."""
    dx = target.x_loc - landmark.x_loc
    dy = target.y_loc - landmark.y_loc   # positive = target is below landmark

    angle = math.degrees(math.atan2(dy, dx))   # -180..180

    # Map angle to compass label (image y-axis flipped from math convention)
    if   -22.5  <= angle <  22.5:  return "to the right of"
    elif  22.5  <= angle <  67.5:  return "below and to the right of"
    elif  67.5  <= angle < 112.5:  return "below"
    elif 112.5  <= angle < 157.5:  return "below and to the left of"
    elif angle >= 157.5 or angle < -157.5: return "to the left of"
    elif -157.5 <= angle < -112.5: return "above and to the left of"
    elif -112.5 <= angle <  -67.5: return "above"
    else:                           return "above and to the right of"


def _landmark_description(obj: Object) -> str:
    f = obj.features()
    return f"the {f['color']} {f['shape']}"


def _target_description(target: Object, scene: Scene, target_idx: int) -> str:
    """Minimal self-description of the target (color+shape, or add size if needed)."""
    distractors = [o for i, o in enumerate(scene.objects) if i != target_idx]
    f = target.features()
    base = f"{f['color']} {f['shape']}"
    # If base description is ambiguous, add size
    if any(o.features()["color"] == f["color"] and o.features()["shape"] == f["shape"]
           for o in distractors):
        base = f"{f['size']} {base}"
    return f"the {base}"


# ── Speakers ──────────────────────────────────────────────────────────────────

class LandmarkSpeaker(BaseSpeaker):
    """
    Rule-based speaker that uses a unique landmark object for spatial reference.
    Falls back to color+shape self-description if no good landmark exists.
    """

    @property
    def name(self) -> str:
        return "landmark"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        target   = scene.objects[target_idx]
        landmark = _best_landmark(target, scene, target_idx)

        if landmark is not None and _uniqueness(landmark, scene.objects) > 0:
            direction   = _direction(target, landmark)
            target_desc = _target_description(target, scene, target_idx)
            lm_desc     = _landmark_description(landmark)
            text        = f"{target_desc} {direction} {lm_desc}"
        else:
            # Fallback: plain self-description
            text = _target_description(target, scene, target_idx)

        return Utterance(
            text=text,
            speaker_type=self.name,
            speaker_meta={"landmark_id": landmark.id if landmark else None},
        )


_SYSTEM_LANDMARK_VLLM = """\
You are a speaker in a visual reference game.

The TARGET object:
  color    : {color}
  shape    : {shape}
  size     : {size}
  location : {location}

All other objects in the scene:
{distractors}

Your task: describe the target using a LANDMARK strategy.
Look for a visually distinctive nearby object and describe the target's \
position relative to it.
Example: "the blue circle just above the large yellow square"

Rules:
- Pick a landmark that is easy to identify (unique color, shape, or size).
- Use a clear spatial relation: above, below, left of, right of, etc.
- Keep the expression SHORT (under 12 words).

Output ONLY the referring expression. No explanation."""


class LandmarkVLLMSpeaker(BaseSpeaker):
    """
    LLM-based landmark speaker: the model receives the full scene description
    and decides which nearby object to use as a landmark and how to express
    the spatial relation — no Python pre-computation of landmark or direction.
    """

    def __init__(self, client) -> None:
        self.client = client

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"landmark-vllm({model})"

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

        system = _SYSTEM_LANDMARK_VLLM.format(**feats, distractors=distractors)

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
