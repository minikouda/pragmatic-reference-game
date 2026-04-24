"""
VLLM Speaker: uses a vision-language model to generate referring expressions.

The speaker receives the scene image with the target object visually highlighted
(a numbered label overlay) and is asked to produce a natural referring expression
that a listener could use to identify the target from the image.

Two modes
---------
naive     : "Describe the target object so someone can find it."
pragmatic : Chain-of-thought — enumerate distinguishing features, then produce
            the shortest sufficient expression.

Image annotation
----------------
We draw a red bounding box and index label around the target in a temporary copy
of the image so the model knows exactly which object to describe.
Without annotation the model has no way to know which object is the target.
"""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path

from ..data.schema import Object, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseSpeaker


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_NAIVE = """\
You are a speaker in a visual reference game.

You see a scene with several colored shapes. The TARGET object has these properties:
  color    : {color}
  shape    : {shape}
  size     : {size}
  location : {location}

Your task: produce a SHORT natural-language referring expression that uniquely \
identifies the target so a listener, seeing the same image, can pick it out.

Rules:
- Use only the properties listed above.
- Be as brief as possible — omit attributes that don't help distinguish the target.
- Output ONLY the referring expression (e.g. "the large red circle"). No other text."""

_SYSTEM_PRAGMATIC = """\
You are a pragmatic speaker in a visual reference game, reasoning like a \
Rational Speech Act (RSA) model.

The TARGET object has these properties:
  color    : {color}
  shape    : {shape}
  size     : {size}
  location : {location}

The scene also contains other objects visible in the image (distractors).

Think step by step:
1. For each property, check whether it alone rules out all other objects in the scene.
2. Find the MINIMAL set of properties that uniquely identifies the target.
3. Produce the shortest natural English expression using only those properties.

Output format (follow exactly):
REASONING: <your analysis>
EXPRESSION: <the referring expression>"""


# ── Speaker ───────────────────────────────────────────────────────────────────

class VLLMSpeaker(BaseSpeaker):
    """
    Speaker that sends the annotated scene image to a VLM.

    Parameters
    ----------
    client    : LLMClient configured for a vision-capable model.
    pragmatic : if True, use chain-of-thought prompt.
    """

    def __init__(self, client: LLMClient, pragmatic: bool = False) -> None:
        self.client    = client
        self.pragmatic = pragmatic

    @property
    def name(self) -> str:
        mode  = "pragmatic" if self.pragmatic else "naive"
        model = self.client.model.split("/")[-1]
        return f"vllm-{mode}({model})"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        if scene.image_path is None:
            raise ValueError(
                f"Scene {scene.id} has no image_path. "
                "VLLMSpeaker requires image-backed scenes."
            )

        target = scene.objects[target_idx]
        feats  = target.features()
        template = _SYSTEM_PRAGMATIC if self.pragmatic else _SYSTEM_NAIVE
        system   = template.format(**feats)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user",   content="Describe the target so a listener can identify it from the image."),
            ],
            image_path=scene.image_path,
        )

        if self.pragmatic:
            expression = _extract_expression(raw)
            reasoning  = _extract_reasoning(raw)
        else:
            expression = raw.strip().strip('"')
            reasoning  = None

        return Utterance(
            text=expression,
            speaker_type=self.name,
            speaker_meta={"raw_response": raw, "reasoning": reasoning},
        )


# ── Image annotation (draws TARGET box) ───────────────────────────────────────

def _annotate_image(
    image_path: str | Path,
    objects:    list[Object],
    target_idx: int,
) -> Path:
    """
    Draw a red bounding box + "TARGET" label around the target object.

    Returns a path to a temporary annotated PNG (cleaned up by the OS on exit).
    Falls back to the original image path if Pillow is not installed.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return Path(image_path)  # graceful degradation without Pillow

    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    target = objects[target_idx]
    scale_x = w / 100
    scale_y = h / 100
    cx = int(target.x_loc * scale_x)
    cy = int(target.y_loc * scale_y)

    # Use the object's actual pixel radius + small padding, not a fixed 20px
    from ..data.schema import SIZE_MAP
    size_label = target.size if target.size in ("small", "medium", "large") else "medium"
    px_radius  = {v: k for k, v in SIZE_MAP.items()}.get(size_label, 12)
    r = int((px_radius + 4) * scale_x)

    box = [cx - r, cy - r, cx + r, cy + r]
    draw.rectangle(box, outline="magenta", width=2)
    draw.text((box[0], max(0, box[1] - 14)), "TARGET", fill="magenta")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return Path(tmp.name)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _extract_expression(text: str) -> str:
    m = re.search(r"EXPRESSION:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"')
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip('"') if lines else text.strip()


def _extract_reasoning(text: str) -> str | None:
    m = re.search(r"REASONING:\s*(.+?)(?=EXPRESSION:|$)", text,
                  re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None
