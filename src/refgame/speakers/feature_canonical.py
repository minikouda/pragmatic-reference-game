"""
FeatureCanonicalSpeaker: always outputs full canonical feature tuple.

Motivation
----------
FeatureMatchListener works best when the utterance contains explicit feature
names (color, shape, size, location) that it can directly parse.  Vague
utterances ("the small one") or spatial-only descriptions ("the one on the
left") only partially match object features, giving flat posteriors.

This speaker always outputs the full tuple — using only discriminating features
for brevity, but in a standardized format that feature-based listeners can parse.

Two variants
------------
feature_canonical    : rule-based, outputs minimal canonical description
feature_canonical_vllm : LLM picks minimally discriminating features but
                         outputs them in the same canonical format
"""

from __future__ import annotations

from ..data.schema import FEATURE_KEYS, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseSpeaker


def _discriminating_features(scene: Scene, target_idx: int) -> list[str]:
    """
    Return the subset of FEATURE_KEYS that uniquely identify the target
    among all objects in the scene.  Falls back to all 4 if needed.
    """
    target = scene.objects[target_idx]
    distractors = [o for i, o in enumerate(scene.objects) if i != target_idx]
    tf = target.features()

    # Greedily add features until the description is unique
    chosen: list[str] = []
    for key in FEATURE_KEYS:
        # Remaining distractors after filtering by all chosen + this key
        candidates = [
            d for d in distractors
            if all(d.features()[k] == tf[k] for k in chosen + [key])
        ]
        chosen.append(key)
        if not candidates:
            break
    return chosen


def _build_canonical(scene: Scene, target_idx: int) -> str:
    """Build 'the SIZE COLOR SHAPE at LOCATION' using only discriminating features."""
    target = scene.objects[target_idx]
    tf = target.features()
    chosen = _discriminating_features(scene, target_idx)

    parts = []
    if "size"  in chosen: parts.append(tf["size"])
    if "color" in chosen: parts.append(tf["color"])
    if "shape" in chosen: parts.append(tf["shape"])

    base = " ".join(parts) if parts else f"{tf['color']} {tf['shape']}"
    if "location" in chosen:
        return f"the {base} at {tf['location']}"
    return f"the {base}"


class FeatureCanonicalSpeaker(BaseSpeaker):
    """
    Rule-based speaker using a minimal canonical feature description.
    Designed to pair well with FeatureMatchListener.
    """

    @property
    def name(self) -> str:
        return "feature-canonical"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        text = _build_canonical(scene, target_idx)
        chosen = _discriminating_features(scene, target_idx)
        return Utterance(
            text=text,
            speaker_type=self.name,
            speaker_meta={"discriminating_features": chosen},
        )


_SYSTEM_FEATURE_CANONICAL_VLLM = """\
You are a speaker in a visual reference game.

The TARGET object:
  color    : {color}
  shape    : {shape}
  size     : {size}
  location : {location}

The other objects in the scene:
{distractors}

Your task: produce a MINIMAL canonical description using ONLY the features that
distinguish the target from all other objects.

Output format: "the [SIZE] [COLOR] SHAPE [at LOCATION]"
  - Include SIZE only if needed.
  - Include LOCATION only if needed.
  - Always include COLOR and SHAPE.

Examples:
  "the red circle"                    (color+shape is sufficient)
  "the large blue square"             (need size to disambiguate)
  "the red circle at bottom-left"     (need location to disambiguate)

Output ONLY the canonical description. No explanation."""


class FeatureCanonicalVLLMSpeaker(BaseSpeaker):
    """
    LLM-based speaker that picks minimal discriminating features and outputs
    them in the canonical format that FeatureMatchListener can reliably parse.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"feature-canonical-vllm({model})"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        target = scene.objects[target_idx]
        feats = target.features()

        distractors = "\n".join(
            f"  - {o.features()['size']} {o.features()['color']} {o.features()['shape']} at {o.features()['location']}"
            for i, o in enumerate(scene.objects) if i != target_idx
        )

        system = _SYSTEM_FEATURE_CANONICAL_VLLM.format(**feats, distractors=distractors)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content="Describe the target canonically."),
            ],
            image_path=scene.image_path,
        )

        expression = raw.strip().strip('"').strip(".")
        return Utterance(
            text=expression,
            speaker_type=self.name,
            speaker_meta={"raw_response": raw},
        )
