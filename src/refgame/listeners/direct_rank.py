"""
Image-only listener variants for the reference game.

All listeners receive the scene image annotated with index numbers (0, 1, 2, ...)
placed at each object's center. No object feature text is provided — the model
must reason from visual perception alone.

  DirectRankListener    : direct probability array from indexed image
  CoTRankListener       : observe → score 0-10 → assign from indexed image
  EliminationListener   : visually rule out → assign from indexed image
  IndexListener         : hard commit to a single index (CPA == Accuracy)
"""

from __future__ import annotations

import json
import re

from ..data.schema import ListenerOutput, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseListener, annotate_indices


# ── Shared helpers ────────────────────────────────────────────────────────────

def _parse_probs(raw: str, n: int) -> list[float] | None:
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
    m = re.search(rf"{tag}\s*:\s*(\[[\d.,\s]+\])", raw, re.IGNORECASE)
    if m:
        try:
            arr = json.loads(m.group(1))
            if len(arr) == n and all(isinstance(v, (int, float)) for v in arr):
                return [float(v) for v in arr]
        except (json.JSONDecodeError, ValueError):
            pass
    return _parse_probs(raw, n)


def _normalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total < 1e-12:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


def _listen(client, scene, utterance, prompt_tmpl, parse_fn, user_msg, cost_c: float = 0.25):
    """Annotate image with indices, call VLM, parse posterior."""
    n = len(scene.objects)
    annotated = annotate_indices(scene.image_path, scene.objects)
    prompt = prompt_tmpl.format(
        n=n, n1=n - 1, utterance=utterance.text,
        cost_c=cost_c, threshold=1.0 - cost_c,
    )
    raw = client.complete(
        messages=[
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=user_msg),
        ],
        image_path=annotated,
    )
    raw_probs = parse_fn(raw, n)
    posterior = _normalize(raw_probs) if raw_probs is not None else [1.0 / n] * n
    predicted = posterior.index(max(posterior))
    return raw, posterior, predicted, raw_probs is not None


# ── Prompt templates ──────────────────────────────────────────────────────────

_DIRECT = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and position.

The speaker said: "{utterance}"

Clarification cost: c = {cost_c:.2f}. You should only place high confidence on
an object if you are at least {threshold:.0%} sure — otherwise spread probability
to reflect genuine uncertainty.

Assign a probability to each numbered object being the one the speaker refers to.
Use ONLY what you see in the image. Probabilities must sum to 1.0.

Output ONLY a JSON array of {n} floats:  [p0, p1, ..., p{n1}]
No explanation."""

_COT = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and size.

The speaker said: "{utterance}"

Clarification cost: c = {cost_c:.2f}. You should commit with confidence ≥ {threshold:.0%}
or spread uncertainty honestly across candidates.

Reason step by step using only what you see:
1. OBSERVE — for each label 0..{n1}, describe what you see (color, shape, size).
2. MATCH — score how well the utterance describes each object (0 = no match, 10 = perfect).
3. ASSIGN — convert scores to probabilities summing to 1.0.

Output format (follow exactly):
SCORES: [s0, s1, ...]   ← integers 0–10
PROBS:  [p0, p1, ...]   ← floats summing to 1.0"""

_ELIMINATION = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and position.

The speaker said: "{utterance}"

Clarification cost: c = {cost_c:.2f}. Commit confidently (≥ {threshold:.0%} on one object)
only if the utterance clearly identifies it — otherwise distribute honestly.

Use an elimination strategy based only on what you see:
1. RULED_OUT — list indices of objects that clearly do NOT match the utterance
   (wrong color, wrong shape, wrong position). Be aggressive.
2. CANDIDATES — the remaining indices after elimination.
3. PROBS — probability 0 for ruled-out objects; distribute the rest among
   candidates proportional to how well each matches visually.

Output format (follow exactly):
RULED_OUT: [i, j, ...]
CANDIDATES: [i, j, ...]
PROBS: [p0, p1, ...]   ← {n} floats summing to 1.0"""

_INDEX = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and position.

The speaker said: "{utterance}"

Clarification cost: c = {cost_c:.2f}. Only commit to an index if you are at least
{threshold:.0%} confident it is the correct object.

Your task: identify which single numbered object the speaker is referring to.
Use ONLY what you see in the image.

Output ONLY a single integer (the label of the object). No explanation."""


# ── Listeners ─────────────────────────────────────────────────────────────────

class DirectRankListener(BaseListener):
    """Direct probability array from indexed image + utterance only."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"direct({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance, cost_c: float = 0.25) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        raw, posterior, predicted, parse_ok = _listen(
            self.client, scene, utterance, _DIRECT,
            _parse_probs, "Assign probabilities based on what you see.", cost_c,
        )
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )


class CoTRankListener(BaseListener):
    """Chain-of-thought scoring from indexed image + utterance only."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"cot({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance, cost_c: float = 0.25) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        def _parse(raw, n):
            return _parse_tagged_probs(raw, "PROBS", n) or _parse_tagged_probs(raw, "SCORES", n)

        raw, posterior, predicted, parse_ok = _listen(
            self.client, scene, utterance, _COT,
            _parse, "Observe, score, then assign probabilities.", cost_c,
        )
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )


class EliminationListener(BaseListener):
    """Elimination strategy from indexed image + utterance only."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"elimination({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance, cost_c: float = 0.25) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        raw, posterior, predicted, parse_ok = _listen(
            self.client, scene, utterance, _ELIMINATION,
            lambda r, n: _parse_tagged_probs(r, "PROBS", n),
            "Eliminate visually, then assign probabilities.", cost_c,
        )
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )


class IndexListener(BaseListener):
    """
    Hard-commit listener: outputs a single integer index, no distribution.

    Posterior is always 1.0 on the chosen object, so the EU rule never
    triggers clarification and CPA == Accuracy. Serves as an upper-bound
    reference for ask-rate-free performance.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"index({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance, cost_c: float = 0.25) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        n = len(scene.objects)
        annotated = annotate_indices(scene.image_path, scene.objects)
        prompt = _INDEX.format(n=n, n1=n - 1, utterance=utterance.text, cost_c=cost_c, threshold=1.0 - cost_c)
        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="Which object index is the speaker referring to?"),
            ],
            image_path=annotated,
        )
        m = re.search(r"\b(\d+)\b", raw.strip())
        if m:
            idx = int(m.group(1))
            parse_ok = 0 <= idx < n
            predicted = idx if parse_ok else 0
        else:
            predicted = 0
            parse_ok = False
        posterior = [0.0] * n
        posterior[predicted] = 1.0
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )
