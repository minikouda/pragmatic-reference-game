"""
FeatureMatchListener: VLM identifies referred object by features, not coordinates.

Instead of predicting (x, y) coordinates (which requires precise spatial
reasoning), the model reads the utterance and produces a feature description
of the referred object (color, shape, size, location).  Each scene object is
then scored by how many of its features match the extracted description.

Why this helps
--------------
Coordinate prediction fails when utterances are spatial ("the one on the left")
or vague ("the small one").  Feature matching directly exploits the discrete
feature space that all objects live in, and is robust to imprecise spatial
language.

Posterior
---------
    overlap_i = number of matching features between extracted description and obj_i
    score_i   = exp(overlap_i * sharpness)    [sharpness controls peakedness]
    posterior = softmax(scores)

With sharpness=2 and 4 features, a perfect 4/4 match scores exp(8) ≈ 3000x
higher than a 0/4 match, giving near-certain posteriors on unambiguous objects.

Two variants
------------
FeatureMatchListener      : VLM reads utterance + scene image
FeatureMatchTextListener  : VLM reads utterance + structured scene description
                            (no image) — useful for ablating visual grounding
"""

from __future__ import annotations

import json
import math
import re

from ..data.schema import FEATURE_KEYS, ListenerOutput, Object, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseListener


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_FEATURE_MATCH = """\
You are a listener in a visual reference game.

You see a scene with several colored shapes.

The speaker said: "{utterance}"

Your task: identify the properties of the object the speaker is referring to.

Output ONLY a JSON object with these keys:
  "color"    : one of black, red, blue, green, yellow, orange, purple, magenta
  "shape"    : one of circle, square, triangle
  "size"     : one of small, medium, large
  "location" : one of top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right

If you are unsure about a property, output null for that key.

Example: {{"color": "red", "shape": "circle", "size": "large", "location": null}}

No explanation."""

_SYSTEM_FEATURE_MATCH_TEXT = """\
You are a listener in a visual reference game.

The scene contains these objects:
{scene_desc}

The speaker said: "{utterance}"

Your task: identify the properties of the object the speaker is referring to.

Output ONLY a JSON object with these keys:
  "color"    : one of black, red, blue, green, yellow, orange, purple, magenta
  "shape"    : one of circle, square, triangle
  "size"     : one of small, medium, large
  "location" : one of top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right

If you are unsure about a property, output null for that key.

No explanation."""


# ── Feature extraction ────────────────────────────────────────────────────────

def _parse_features(raw: str) -> dict[str, str | None]:
    try:
        blob = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if blob:
            d = json.loads(blob.group())
            return {k: d.get(k) for k in FEATURE_KEYS}
    except (json.JSONDecodeError, ValueError):
        pass
    return {k: None for k in FEATURE_KEYS}


def _feature_overlap(obj: Object, desc: dict[str, str | None]) -> int:
    """Number of non-null features in desc that match obj's features."""
    obj_feats = obj.features()
    return sum(
        1 for k in FEATURE_KEYS
        if desc.get(k) is not None and desc[k] == obj_feats[k]
    )


def _features_to_posterior(
    desc: dict[str, str | None],
    objects: list[Object],
    sharpness: float = 2.0,
) -> tuple[list[float], int]:
    """
    Score each object by feature overlap and return a softmax posterior.

    sharpness: multiplier on overlap count before exp().
               Higher → more peaked posteriors.
    """
    overlaps = [_feature_overlap(obj, desc) for obj in objects]
    scores = [math.exp(ov * sharpness) for ov in overlaps]
    total = sum(scores)
    if total < 1e-12:
        posterior = [1.0 / len(objects)] * len(objects)
    else:
        posterior = [s / total for s in scores]
    predicted = posterior.index(max(posterior))
    return posterior, predicted


# ── Listeners ─────────────────────────────────────────────────────────────────

class FeatureMatchListener(BaseListener):
    """
    VLM extracts referred-object features from scene image + utterance.
    Posterior computed by feature-overlap softmax.

    Parameters
    ----------
    client    : vision-capable LLMClient
    sharpness : softmax temperature multiplier (default 2.0)
    """

    def __init__(self, client: LLMClient, sharpness: float = 2.0) -> None:
        self.client = client
        self.sharpness = sharpness

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"feature-match({model})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        prompt = _SYSTEM_FEATURE_MATCH.format(utterance=utterance.text)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="What are the properties of the referred object?"),
            ],
            image_path=scene.image_path,
        )

        desc = _parse_features(raw)
        posterior, predicted = _features_to_posterior(desc, scene.objects, self.sharpness)

        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={
                "raw_response": raw,
                "extracted_features": desc,
                "overlap_scores": [_feature_overlap(o, desc) for o in scene.objects],
            },
        )


class FeatureMatchTextListener(BaseListener):
    """
    Like FeatureMatchListener but uses a structured text scene description
    instead of the image (ablation: tests whether visual grounding matters).

    Parameters
    ----------
    client    : any LLMClient (does not need vision capability)
    sharpness : softmax temperature multiplier
    """

    def __init__(self, client: LLMClient, sharpness: float = 2.0) -> None:
        self.client = client
        self.sharpness = sharpness

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"feature-match-text({model})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        scene_desc = "\n".join(
            f"  [{i}] {o.features()['size']} {o.features()['color']} "
            f"{o.features()['shape']} at {o.features()['location']}"
            for i, o in enumerate(scene.objects)
        )
        prompt = _SYSTEM_FEATURE_MATCH_TEXT.format(
            scene_desc=scene_desc,
            utterance=utterance.text,
        )

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="What are the properties of the referred object?"),
            ],
        )

        desc = _parse_features(raw)
        posterior, predicted = _features_to_posterior(desc, scene.objects, self.sharpness)

        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={
                "raw_response": raw,
                "extracted_features": desc,
                "overlap_scores": [_feature_overlap(o, desc) for o in scene.objects],
            },
        )
