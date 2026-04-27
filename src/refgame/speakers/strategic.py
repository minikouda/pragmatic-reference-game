"""
StrategicVLLMSpeaker: visually-grounded speakers using prompt-engineered strategies.

Design principle
----------------
Earlier VLLM speakers (VLLMSpeaker, SceneAwareSpeaker, LandmarkVLLMSpeaker,
ContrastiveVLLMSpeaker) all pre-parsed the scene into structured feature lists
and injected them into the prompt.  The model was effectively a text formatter —
it never needed to look at the image or reason about what it saw.

This module takes the opposite approach:
  1. Annotate the image to mark the target (magenta box + "TARGET" label).
  2. Give the model ONLY a strategy instruction — no pre-computed features.
  3. Let the model's own visual reasoning produce natural language.

The result is descriptions generated from genuine visual understanding:
  "the round shape near the triangle in the corner"
  "unlike the other blue shape, this one is the large one on the right"
  "the circle directly above the yellow square"

Strategies
----------
  natural      : "Describe the target object so a listener can identify it."
                 Unconstrained — tests the model's default referring strategy.

  contrastive  : "Find the object most similar to the target and describe what
                 makes the target different."
                 Forces the model to reason about near-confusable distractors.

  landmark     : "Describe where the target is relative to a distinctive nearby
                 object."
                 Forces spatial relational language.

  superlative  : "Describe the target using a superlative or uniqueness property
                 (e.g., 'the only', 'the largest', 'the leftmost')."
                 Best for scenes where the target has an extreme value.

  pragmatic    : Chain-of-thought: enumerate all objects, identify distractors,
                 then produce the minimal discriminating expression.
                 Most expensive but should produce the most precise utterances.

All strategies produce a single short natural-language referring expression.
The annotated image is sent alongside the prompt so the model can ground its
description visually.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..data.schema import Object, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseSpeaker


# ── Strategy prompts ──────────────────────────────────────────────────────────
# Each prompt ends with a request for a SHORT referring expression only.
# The target is visually marked in the image — the prompt does NOT describe its
# features; the model must use its own visual perception.

_PROMPTS: dict[str, str] = {

    "scene_first": """\
You are a speaker in a visual reference game.

The TARGET object is marked with a MAGENTA bounding box in the image.

Think step by step:
1. INVENTORY — list every object you see in the image (color, shape, approximate size).
2. IDENTIFY — which one has the magenta box? Confirm its properties.
3. DISCRIMINATE — what single property or combination makes it uniquely identifiable
   compared to every other object you listed?
4. EXPRESSION — write the shortest natural phrase using only that property.

Output format (follow exactly):
INVENTORY: <comma-separated list of objects>
TARGET: <properties of the marked object>
DISCRIMINATING: <what makes it unique>
EXPRESSION: <the referring expression>""",

    "listener_aware": """\
You are a speaker in a visual reference game.

The TARGET object is marked with a MAGENTA bounding box in the image.

Your listener will see the SAME image, read your description, and assign a
probability to each object being the one you mean.  Your goal is to write
a description that gives the listener maximum confidence in the correct object.

Think: what description would make every OTHER object clearly wrong, and this
object clearly right?  Prefer descriptions using visually obvious properties
(color, shape, relative position) over subtle ones.

Output ONLY the referring expression (short, under 12 words). No other text.""",

    "natural": """\
You are a speaker in a visual reference game.

The TARGET object is marked with a MAGENTA bounding box in the image.

Your task: produce a SHORT natural referring expression that uniquely identifies
the target so a listener, seeing the same image, can pick it out.

Rules:
- Use whatever description comes naturally from looking at the image.
- Be as brief as possible — omit anything a listener doesn't need.
- Output ONLY the referring expression (e.g. "the large red circle"). No other text.""",

    "contrastive": """\
You are a speaker in a visual reference game.

The TARGET object is marked with a MAGENTA bounding box in the image.

Your task: produce a CONTRASTIVE referring expression.
Look at the other objects in the scene and find the one most likely to be
confused with the target.  Then highlight the ONE feature that distinguishes
the target from that confusable object.

Examples:
  "the LARGE blue square"          (if there is also a small blue square)
  "the blue square on the LEFT"    (if there is another blue square elsewhere)
  "the red circle, not the red triangle"

Rules:
- Identify the most similar distractor by visual inspection.
- Surface exactly one distinguishing feature.
- Keep the expression under 12 words.

Output ONLY the referring expression. No other text.""",

    "landmark": """\
You are a speaker in a visual reference game.

The TARGET object is marked with a MAGENTA bounding box in the image.

Your task: produce a LANDMARK referring expression.
Look for a visually distinctive nearby object (unique color, shape, or size)
and describe where the target is relative to it.

Example: "the blue circle just above the large yellow square"

Rules:
- Pick a landmark that is easy to identify visually (stands out in the scene).
- Use a clear spatial relation: above, below, left of, right of, etc.
- Keep the expression under 12 words.

Output ONLY the referring expression. No other text.""",

    "superlative": """\
You are a speaker in a visual reference game.

The TARGET object is marked with a MAGENTA bounding box in the image.

Your task: produce a SUPERLATIVE or UNIQUENESS referring expression.
Describe the target using a property that makes it extreme or unique in the scene.

Examples:
  "the only triangle"
  "the largest circle"
  "the leftmost blue shape"
  "the shape in the top-right corner"
  "the smallest object"

Rules:
- Look for what makes the target stand out: is it the only one of its kind?
  The biggest? The leftmost? In a corner?
- Use whichever extreme property is most obvious from the image.
- Keep the expression under 10 words.

Output ONLY the referring expression. No other text.""",

    "pragmatic": """\
You are a pragmatic speaker in a visual reference game, reasoning like an RSA model.

The TARGET object is marked with a MAGENTA bounding box in the image.

Think step by step:
1. Look at all objects in the scene and list their visible properties.
2. List any objects that could be confused with the TARGET.
3. For each confusable object, identify the one property where it differs from the TARGET.
4. Use that property to produce the shortest possible unambiguous expression.

Output format (follow exactly):
OBJECTS: <brief list of what you see>
CONFUSABLE: <which objects could be confused with the target>
DISTINGUISHING: <which property separates the target from confusable objects>
EXPRESSION: <the referring expression>""",
}


# ── Image annotation ──────────────────────────────────────────────────────────

def _annotate_target(
    image_path: str | Path,
    objects:    list[Object],
    target_idx: int,
) -> Path:
    """
    Draw a magenta bounding box + 'TARGET' label on the target object.
    Returns path to a temporary annotated PNG.
    Falls back to the original image if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return Path(image_path)

    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    target = objects[target_idx]

    # Mirror renderer.py coordinate formula exactly:
    #   px = MARGIN + x_loc/100 * (W - 2*MARGIN)
    #   py = (H - MARGIN) - y_loc/100 * (H - 2*MARGIN)
    # Scale MARGIN proportionally if the image has been resized.
    CANVAS_W, CANVAS_H, MARGIN_PX = 330, 328, 5
    scale_x = w / CANVAS_W
    scale_y = h / CANVAS_H
    margin_x = MARGIN_PX * scale_x
    margin_y = MARGIN_PX * scale_y
    draw_w   = w - 2 * margin_x
    draw_h   = h - 2 * margin_y

    px = int(margin_x + target.x_loc / 100 * draw_w)
    py = int((h - margin_y) - target.y_loc / 100 * draw_h)

    # Object radius from SIZE_MAP, scaled to actual image dimensions.
    from ..data.schema import SIZE_MAP
    size_label = target.size if hasattr(target, "size") else "medium"
    px_radius = next((k for k, v in SIZE_MAP.items() if v == size_label), 12)
    r = int(px_radius * scale_x + 6)

    box = [px - r, py - r, px + r, py + r]
    color = "magenta"
    draw.rectangle(box, outline=color, width=3)
    draw.text((box[0], max(0, box[1] - 16)), "TARGET", fill=color)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return Path(tmp.name)


# ── Expression extraction ─────────────────────────────────────────────────────

def _extract_expression(raw: str) -> str:
    import re
    m = re.search(r"EXPRESSION:\s*(.+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"').strip(".")
    # Fallback: last non-empty line
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    return lines[-1].strip('"').strip(".") if lines else raw.strip()


# ── Speaker ───────────────────────────────────────────────────────────────────

class StrategicVLLMSpeaker(BaseSpeaker):
    """
    A visually-grounded VLM speaker that uses annotated images and
    strategy-only prompts — no pre-parsed feature lists.

    The target is marked in the image with a magenta bounding box.
    The model must use its own visual perception to produce a description
    consistent with the requested strategy.

    Parameters
    ----------
    client   : vision-capable LLMClient
    strategy : one of "natural", "contrastive", "landmark", "superlative",
               "pragmatic"
    """

    STRATEGIES = list(_PROMPTS.keys())

    def __init__(self, client: LLMClient, strategy: str = "natural") -> None:
        if strategy not in _PROMPTS:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(_PROMPTS)}")
        self.client   = client
        self.strategy = strategy

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"strategic-{self.strategy}({model})"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        annotated_path = _annotate_target(scene.image_path, scene.objects, target_idx)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=_PROMPTS[self.strategy]),
                ChatMessage(role="user",   content="Describe the TARGET object."),
            ],
            image_path=annotated_path,
        )

        expression = _extract_expression(raw)

        return Utterance(
            text=expression,
            speaker_type=self.name,
            speaker_meta={
                "strategy": self.strategy,
                "raw_response": raw,
            },
        )
