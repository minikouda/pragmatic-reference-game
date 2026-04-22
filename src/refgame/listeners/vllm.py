"""
VLLM Listener: uses a vision-language model to infer the intended object.

The listener sees the scene image + the speaker's utterance and must identify
which object the speaker is referring to.

Posterior construction
----------------------
Most VLMs don't expose log-probabilities, so we use a forced-choice strategy:

  1. Show the image with all objects labeled 0..N-1 (numbered overlays).
  2. Ask the model to output a JSON dict {index: confidence} scoring each object.
  3. Softmax the confidences to get a proper posterior.

Fallback: if the model outputs only a single index (integer), convert to a
one-hot-style posterior with high confidence on the chosen object and uniform
small mass on the rest.
"""

from __future__ import annotations

import json
import math
import re

from ..data.schema import ListenerOutput, Object, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseListener
from ..speakers.vllm import _annotate_image   # reuse annotation helper


# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_LISTENER = """\
You are a listener in a visual reference game.

You see a scene with several colored shapes. Each object is labeled with a \
number (0, 1, 2, ...) drawn directly on the image.

The speaker said: "{utterance}"

Your task: decide which object the speaker most likely intended.

Output a JSON object with one key per object index, mapping to a confidence \
score (0–10) for how well the utterance matches that object:
  {{"0": <score>, "1": <score>, ...}}

Be precise: 10 = perfect match, 0 = no match at all.
Output ONLY valid JSON. No explanation."""


# ── Listener ──────────────────────────────────────────────────────────────────

class VLLMListener(BaseListener):
    """
    Listener that sends the labeled scene image + utterance to a VLM.

    Parameters
    ----------
    client : LLMClient configured for a vision-capable model.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"vllm-listener({model})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(
                f"Scene {scene.id} has no image_path. "
                "VLLMListener requires image-backed scenes."
            )

        annotated  = _label_all_objects(scene.image_path, scene.objects)
        prompt     = _SYSTEM_LISTENER.format(utterance=utterance.text)
        n          = len(scene.objects)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user",   content=f"Which object did the speaker mean? ({n} objects labeled 0–{n-1})"),
            ],
            image_path=annotated,
        )

        scores     = _parse_scores(raw, n)
        posterior  = _softmax(scores)
        predicted  = posterior.index(max(posterior))

        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "raw_scores": scores},
        )


# ── Image annotation (draws index labels on all objects) ──────────────────────

def _label_all_objects(
    image_path: str,
    objects:    list[Object],
) -> "Path":
    """Draw numbered labels on all objects so the VLM can reference them by index."""
    from pathlib import Path
    import tempfile

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return Path(image_path)

    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    scale_x, scale_y = w / 100, h / 100

    for i, obj in enumerate(objects):
        cx  = int(obj.x_loc * scale_x)
        cy  = int(obj.y_loc * scale_y)
        r   = 10
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.rectangle(box, outline="white", width=2)
        draw.text((cx - 4, cy - 7), str(i), fill="white")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    from pathlib import Path
    return Path(tmp.name)


# ── Score parsing ─────────────────────────────────────────────────────────────

def _parse_scores(raw: str, n: int) -> list[float]:
    """
    Parse the VLM's JSON confidence response into a list of n floats.

    Handles:
    - {"0": 8, "1": 2, "2": 9}       — preferred format
    - {"0": 8}                         — partial (missing indices get 1.0)
    - A bare integer "2"               — one-hot at that index
    """
    # Try JSON parse
    try:
        blob = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if blob:
            d = json.loads(blob.group())
            scores = [float(d.get(str(i), 1.0)) for i in range(n)]
            return scores
    except (json.JSONDecodeError, ValueError):
        pass

    # Try bare integer
    m = re.search(r"\b(\d+)\b", raw)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n:
            return [9.0 if i == idx else 1.0 for i in range(n)]

    # Fallback: uniform
    return [1.0] * n


def _softmax(scores: list[float]) -> list[float]:
    max_s = max(scores)
    exps  = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
