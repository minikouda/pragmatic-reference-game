"""
DirectRankListener: VLM directly scores each object in the scene.

Instead of predicting coordinates or extracting features, the model is
shown all objects (described textually) and asked to assign a probability
to each one.  This gives a direct posterior without any intermediate step.

Why this is better than coordinate prediction
---------------------------------------------
- No noisy (x, y) → nearest-object snapping
- The model reasons about the full set of candidates simultaneously
- Handles both spatial and feature-based utterances naturally
- Works even when utterances are ambiguous — the model spreads probability mass

Prompt design
-------------
The model sees:
  1. The scene image (for visual grounding)
  2. A numbered list of all objects with their text descriptions
  3. The speaker's utterance
  4. An instruction to output one probability per object as a JSON array

The raw JSON probabilities are softmax-normalized to ensure they form a
valid distribution, so the model doesn't need to output exact probabilities.
"""

from __future__ import annotations

import json
import math
import re

from ..data.schema import ListenerOutput, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseListener


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_DIRECT_RANK = """\
You are a listener in a visual reference game.

The scene contains {n} objects:
{object_list}

The speaker said: "{utterance}"

Your task: assign a probability to each object being the one the speaker refers to.
The probabilities must sum to 1.0.

Output ONLY a JSON array of {n} numbers, one per object in order:
  [p0, p1, p2, ...]

Example for 3 objects: [0.7, 0.2, 0.1]

No explanation."""

_SYSTEM_COT_RANK = """\
You are a listener in a visual reference game.

The scene contains {n} objects:
{object_list}

The speaker said: "{utterance}"

Think step by step before assigning probabilities:

1. MATCH — For each object [0]...[{n1}], score how well the utterance describes it
   (0 = clearly wrong, 10 = perfect match).  Consider color, shape, size, location,
   and any spatial relations mentioned.

2. RULE OUT — Identify any objects the utterance explicitly contradicts.

3. ASSIGN — Convert your scores to probabilities summing to 1.0.

Output format (follow exactly):
SCORES: [s0, s1, ...]   ← integers 0–10, one per object
PROBS:  [p0, p1, ...]   ← floats summing to 1.0"""

_SYSTEM_ELIMINATION = """\
You are a listener in a visual reference game.

The scene contains {n} objects:
{object_list}

The speaker said: "{utterance}"

Use an ELIMINATION strategy:

1. RULED OUT — List the indices of objects that clearly do NOT match the utterance
   (wrong color, wrong shape, wrong side of the scene, etc.).
   Be aggressive: rule out anything the utterance contradicts.

2. CANDIDATES — The remaining objects after elimination.

3. RANK — Among the candidates, assign probability proportional to how well
   each matches.  Objects that were ruled out get probability 0.

Output format (follow exactly):
RULED_OUT: [i, j, ...]   ← indices of eliminated objects (empty list [] if none)
CANDIDATES: [i, j, ...]  ← remaining indices
PROBS: [p0, p1, ...]     ← {n} floats summing to 1.0, 0 for ruled-out objects"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_objects(scene: Scene) -> str:
    lines = []
    for i, obj in enumerate(scene.objects):
        f = obj.features()
        lines.append(f"  [{i}] {f['size']} {f['color']} {f['shape']} at {f['location']}")
    return "\n".join(lines)


def _parse_probs(raw: str, n: int) -> list[float] | None:
    """Extract a JSON array of n floats; return None on failure."""
    try:
        blob = re.search(r"\[[\d.,\s]+\]", raw)
        if blob:
            arr = json.loads(blob.group())
            if len(arr) == n and all(isinstance(v, (int, float)) for v in arr):
                return [float(v) for v in arr]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_tagged_probs(raw: str, tag: str, n: int) -> list[float] | None:
    """Extract the array on the line starting with TAG: in CoT/elimination output."""
    m = re.search(rf"{tag}\s*:\s*(\[[\d.,\s]+\])", raw, re.IGNORECASE)
    if m:
        try:
            arr = json.loads(m.group(1))
            if len(arr) == n and all(isinstance(v, (int, float)) for v in arr):
                return [float(v) for v in arr]
        except (json.JSONDecodeError, ValueError):
            pass
    # Fall back to any array of the right length
    return _parse_probs(raw, n)


def _softmax(values: list[float]) -> list[float]:
    """Stable softmax."""
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _normalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total < 1e-12:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


# ── Listener ──────────────────────────────────────────────────────────────────

class DirectRankListener(BaseListener):
    """
    Asks the VLM to directly assign probabilities to each scene object.

    The model sees the scene image + a numbered list of all objects + the
    utterance, and outputs one probability per object as a JSON array.

    Parameters
    ----------
    client : vision-capable LLMClient
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"direct-rank({model})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        n = len(scene.objects)
        object_list = _format_objects(scene)
        prompt = _SYSTEM_DIRECT_RANK.format(
            n=n,
            object_list=object_list,
            utterance=utterance.text,
        )

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="What probability does each object have?"),
            ],
            image_path=scene.image_path,
        )

        raw_probs = _parse_probs(raw, n)

        if raw_probs is not None:
            posterior = _normalize(raw_probs)
            parse_ok = True
        else:
            # Fallback: uniform
            posterior = [1.0 / n] * n
            parse_ok = False

        predicted = posterior.index(max(posterior))

        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={
                "raw_response": raw,
                "raw_probs": raw_probs,
                "parse_ok": parse_ok,
            },
        )


class CoTRankListener(BaseListener):
    """
    Like DirectRankListener but uses chain-of-thought reasoning.

    The model first scores each object 0–10 for how well the utterance matches,
    identifies ruled-out objects, then converts scores to probabilities.
    This forces explicit per-object reasoning instead of direct probability guessing,
    which reduces hallucination and improves calibration.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"cot-rank({model})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        n = len(scene.objects)
        prompt = _SYSTEM_COT_RANK.format(
            n=n,
            n1=n - 1,
            object_list=_format_objects(scene),
            utterance=utterance.text,
        )

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="Score and rank each object."),
            ],
            image_path=scene.image_path,
        )

        raw_probs = _parse_tagged_probs(raw, "PROBS", n)
        if raw_probs is None:
            # Try scores line and normalize
            raw_probs = _parse_tagged_probs(raw, "SCORES", n)

        if raw_probs is not None:
            posterior = _normalize(raw_probs)
            parse_ok = True
        else:
            posterior = [1.0 / n] * n
            parse_ok = False

        predicted = posterior.index(max(posterior))
        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )


class EliminationListener(BaseListener):
    """
    Listener that explicitly rules out objects before assigning probabilities.

    The model identifies which objects the utterance contradicts, removes them,
    then distributes probability over the remaining candidates.  This mirrors
    how humans narrow down referents: rule out the impossible, then pick from
    what remains.

    Works especially well for negative/contrastive utterances and utterances
    that mention specific properties (color, shape) that cleanly exclude objects.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        model = self.client.model.split("/")[-1]
        return f"elimination({model})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        n = len(scene.objects)
        prompt = _SYSTEM_ELIMINATION.format(
            n=n,
            object_list=_format_objects(scene),
            utterance=utterance.text,
        )

        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="Eliminate and rank."),
            ],
            image_path=scene.image_path,
        )

        raw_probs = _parse_tagged_probs(raw, "PROBS", n)

        if raw_probs is not None:
            posterior = _normalize(raw_probs)
            parse_ok = True
        else:
            posterior = [1.0 / n] * n
            parse_ok = False

        predicted = posterior.index(max(posterior))
        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )
