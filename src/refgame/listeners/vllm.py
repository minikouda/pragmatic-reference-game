"""
VLLM Listener: uses a vision-language model to infer the intended object.

The listener sees the scene image (no numbered overlays) + the speaker's
utterance and must predict the (x, y) location of the referred object.

Coordinate convention
---------------------
Both the model prompt and scene object coordinates use bottom-left origin:
bottom-left=(0,0), top-right=(100,100), y increases upward.
No coordinate flip is applied; predicted (x,y) is matched directly against
object (x_loc, y_loc) by Euclidean distance.

Accuracy
--------
The predicted (x, y) is snapped to the nearest object by Euclidean distance.
That object's index becomes predicted_idx, and correct = (predicted_idx == target_idx).

Posterior kernel
----------------
Two kernels are available (controlled by `sigma` parameter):

  sigma=None  (default): inverse-distance  score_i = 1 / (d_i + 1e-3)
              Very flat — max(posterior) rarely exceeds 0.75, causing 100% ask
              rate at clarification cost c ≤ 0.25.

  sigma=float: Gaussian kernel  score_i = exp(−d_i² / (2σ²))
              Falls off much faster.  σ=10 (canvas units) gives mean max≈0.83,
              enabling meaningful commit/ask decisions at c=0.25.
              Recommended default: sigma=10.
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
    sigma  : Gaussian kernel width (canvas units, 0–100 scale).
             None → legacy inverse-distance kernel (flat, not recommended).
             10   → recommended; gives sharp posteriors enabling commit at c=0.25.
    """

    def __init__(self, client: LLMClient, sigma: float | None = 10.0) -> None:
        self.client = client
        self.sigma = sigma

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        kernel = f",σ={self.sigma}" if self.sigma is not None else ""
        return f"vllm-listener({model}{kernel})"

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
        posterior, predicted = _coords_to_posterior(pred_x, pred_y, scene.objects, self.sigma)

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
    sigma: float | None = 10.0,
) -> tuple[list[float], int]:
    """
    Convert a predicted (x, y) in bottom-left origin to a posterior over objects.

    Both predicted coords and scene objects use bottom-left origin
    (y=0 at bottom, y increases upward), so no coordinate flip is needed.

    sigma=None  → inverse-distance kernel: score = 1 / (d + 1e-3)
                  Flat posterior; max rarely exceeds 0.75 for 6+ objects.
    sigma=float → Gaussian kernel: score = exp(−d² / (2σ²))
                  Much sharper; σ=10 gives mean max≈0.83, enabling commits at c=0.25.

    predicted_idx = argmax(posterior) = nearest object regardless of kernel.
    """
    dists = [
        math.sqrt((obj.x_loc - pred_x) ** 2 + (obj.y_loc - pred_y) ** 2)
        for obj in objects
    ]

    if sigma is None:
        scores = [1.0 / (d + 1e-3) for d in dists]
    else:
        scores = [math.exp(-d * d / (2.0 * sigma * sigma)) for d in dists]

    total = sum(scores)
    if total < 1e-12:
        posterior = [1.0 / len(objects)] * len(objects)
    else:
        posterior = [s / total for s in scores]

    predicted = posterior.index(max(posterior))
    return posterior, predicted
