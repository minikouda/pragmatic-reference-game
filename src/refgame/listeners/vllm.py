"""
VLLM Listener: uses a vision-language model to infer the intended object.

The listener sees the scene image (no numbered overlays) + the speaker's
utterance and must predict the (x, y) location of the referred object.

Coordinate convention
---------------------
The model is asked to predict in a bottom-left=(0,0), top-right=(100,100)
coordinate system.  Internally, scene objects store y increasing downward
(image convention), so we convert: y_scene → 100 - y_scene before showing
the coordinate system to the model, and invert back when matching.

Accuracy
--------
The predicted (x, y) is snapped to the nearest object by Euclidean distance
(in image-convention coordinates).  That object's index becomes predicted_idx,
and correct = (predicted_idx == target_idx).
"""

from __future__ import annotations

import json
import math
import re

from ..data.schema import ListenerOutput, Object, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseListener


# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_LISTENER = """\
You are a listener in a visual reference game.

You see a scene with several colored shapes on a white canvas.

The speaker said: "{utterance}"

Your task: predict the (x, y) position of the object the speaker is referring to.

Coordinate system: bottom-left is (0, 0), top-right is (100, 100).
x increases to the right, y increases upward.

Output ONLY a JSON object with keys "x" and "y" (floats between 0 and 100):
  {{"x": <value>, "y": <value>}}

No explanation."""


# ── Listener ──────────────────────────────────────────────────────────────────

class VLLMListener(BaseListener):
    """
    Listener that sends the scene image + utterance to a VLM.

    The model predicts (x, y) coordinates (bottom-left origin) of the
    referred object.  The prediction is snapped to the nearest object by
    Euclidean distance to determine predicted_idx and the posterior.

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

        prompt = _SYSTEM_LISTENER.format(utterance=utterance.text)

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user",   content="Where is the object the speaker referred to?"),
            ],
            image_path=scene.image_path,
        )

        pred_x, pred_y = _parse_coords(raw)
        posterior, predicted = _coords_to_posterior(pred_x, pred_y, scene.objects)

        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={
                "raw_response": raw,
                "pred_x": pred_x,
                "pred_y": pred_y,
            },
        )


# ── Coordinate parsing ────────────────────────────────────────────────────────

def _parse_coords(raw: str) -> tuple[float, float]:
    """
    Parse the VLM's JSON coordinate response into (x, y) in bottom-left origin.

    Handles:
    - {"x": 42.0, "y": 67.5}   — preferred format
    - Fallback: (50, 50) center
    """
    try:
        blob = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if blob:
            d = json.loads(blob.group())
            x = float(d.get("x", 50.0))
            y = float(d.get("y", 50.0))
            return x, y
    except (json.JSONDecodeError, ValueError):
        pass

    # Try bare "x=... y=..." pattern
    mx = re.search(r'"?x"?\s*[:=]\s*([\d.]+)', raw)
    my = re.search(r'"?y"?\s*[:=]\s*([\d.]+)', raw)
    if mx and my:
        return float(mx.group(1)), float(my.group(1))

    return 50.0, 50.0


def _coords_to_posterior(
    pred_x: float,
    pred_y: float,
    objects: list[Object],
) -> tuple[list[float], int]:
    """
    Convert a predicted (x, y) in bottom-left origin to a posterior over objects.

    Objects store y increasing downward (image convention), so we flip:
        y_image = 100 - y_bottomleft

    Posterior is a distance-based softmax (closer → higher probability).
    predicted_idx = nearest object.
    """
    y_img = 100.0 - pred_y  # convert to image-convention y

    dists = [
        math.sqrt((obj.x_loc - pred_x) ** 2 + (obj.y_loc - y_img) ** 2)
        for obj in objects
    ]

    # Invert distances to scores (avoid div-by-zero)
    scores = [1.0 / (d + 1e-3) for d in dists]
    total  = sum(scores)
    posterior = [s / total for s in scores]
    predicted = posterior.index(max(posterior))

    return posterior, predicted
