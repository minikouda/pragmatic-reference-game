"""
SceneAwareSpeaker: gives the VLM the full scene context (all objects) alongside
the target's properties, so it can reason about what's distinctive.

This is the simplest upgrade over VLLMSpeaker (naive): the naive speaker only
sees the target's own properties. SceneAwareSpeaker also tells the model what
the distractors look like, enabling it to pick only the features that actually
discriminate the target.

Two modes
---------
scene_aware       : full distractor list in prompt, model picks minimal features
scene_aware_rank  : additionally asks the model to rank features by uniqueness
                    before generating the expression (structured CoT)
"""

from __future__ import annotations

from ..data.schema import Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseSpeaker


def _format_distractors(scene: Scene, target_idx: int) -> str:
    lines = []
    for i, obj in enumerate(scene.objects):
        if i == target_idx:
            continue
        f = obj.features()
        lines.append(f"  - {f['size']} {f['color']} {f['shape']} at {f['location']}")
    return "\n".join(lines)


_SYSTEM_SCENE_AWARE = """\
You are a speaker in a visual reference game.

The TARGET object has these properties:
  color    : {color}
  shape    : {shape}
  size     : {size}
  location : {location}

The other objects in the scene (distractors) are:
{distractors}

Your task: produce the SHORTEST referring expression that uniquely identifies \
the target so a listener can locate it in the image.

Rules:
- Only mention properties that help distinguish the target from the distractors.
- Omit any property that is shared by all other objects (it adds no information).
- Output ONLY the referring expression. No explanation."""

_SYSTEM_SCENE_AWARE_RANK = """\
You are a speaker in a visual reference game.

The TARGET object has these properties:
  color    : {color}
  shape    : {shape}
  size     : {size}
  location : {location}

The other objects in the scene (distractors) are:
{distractors}

Think step by step:
1. For each of the target's properties (color, shape, size, location), count \
how many distractors share that value. Fewer sharers = more distinctive.
2. Rank properties from most to least distinctive.
3. Use the top-ranked property (or two if one is not enough) to form the expression.

Output format (follow exactly):
RANKING: <property: n_sharers, ...>
EXPRESSION: <the referring expression>"""


class SceneAwareSpeaker(BaseSpeaker):
    """
    Gives the VLM both target properties and the full distractor list.
    The model picks only the features that actually discriminate.

    Parameters
    ----------
    client : LLMClient for a vision-capable model.
    ranked : if True, ask the model to explicitly rank features by uniqueness first.
    """

    def __init__(self, client: LLMClient, ranked: bool = False) -> None:
        self.client = client
        self.ranked = ranked

    @property
    def name(self) -> str:
        mode  = "ranked" if self.ranked else "aware"
        model = self.client.model.split("/")[-1]
        return f"scene-{mode}({model})"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        target     = scene.objects[target_idx]
        feats      = target.features()
        distractors = _format_distractors(scene, target_idx)
        template   = _SYSTEM_SCENE_AWARE_RANK if self.ranked else _SYSTEM_SCENE_AWARE
        system     = template.format(**feats, distractors=distractors)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user",   content="Describe the target."),
            ],
            image_path=scene.image_path,
        )

        if self.ranked:
            expression = _extract_expression(raw)
            ranking    = _extract_ranking(raw)
        else:
            expression = raw.strip().strip('"').strip(".")
            ranking    = None

        return Utterance(
            text=expression,
            speaker_type=self.name,
            speaker_meta={"raw_response": raw, "ranking": ranking},
        )


def _extract_expression(text: str) -> str:
    import re
    m = re.search(r"EXPRESSION:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"').strip(".")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip('"') if lines else text.strip()


def _extract_ranking(text: str) -> str | None:
    import re
    m = re.search(r"RANKING:\s*(.+?)(?=EXPRESSION:|$)", text,
                  re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None
