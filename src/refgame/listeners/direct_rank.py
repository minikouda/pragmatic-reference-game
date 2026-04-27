"""
Listener variants for the reference game.

Text-assisted (leaky — receive full object feature list as text):
  DirectRankListener    : direct probability array output
  CoTRankListener       : score 0-10 per object, then convert
  EliminationListener   : rule out objects, distribute over remainder

Honest image-only (no feature text — visual reasoning only):
  ImageOnlyDirectRankListener   : direct probability array from image
  ImageOnlyCoTRankListener      : observe → score → assign from image
  ImageOnlyEliminationListener  : eliminate visually → assign from image

All honest listeners receive the scene image annotated with index numbers
(0, 1, 2, ...) placed at each object's center. No color/shape/size text
is provided; the model must identify objects from visual perception alone.
"""

from __future__ import annotations

import json
import math
import re

from ..data.schema import ListenerOutput, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from .base import BaseListener


# ── Text-assisted prompts (leaky) ─────────────────────────────────────────────

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


# ── Shared helpers ────────────────────────────────────────────────────────────

def _format_objects(scene: Scene) -> str:
    lines = []
    for i, obj in enumerate(scene.objects):
        f = obj.features()
        lines.append(f"  [{i}] {f['size']} {f['color']} {f['shape']} at {f['location']}")
    return "\n".join(lines)


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


# ── Text-assisted listeners (leaky) ───────────────────────────────────────────

class DirectRankListener(BaseListener):
    """VLM assigns probabilities given full object feature text + image."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"direct-rank({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        n = len(scene.objects)
        prompt = _SYSTEM_DIRECT_RANK.format(
            n=n, object_list=_format_objects(scene), utterance=utterance.text,
        )
        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="What probability does each object have?"),
            ],
            image_path=scene.image_path,
        )
        raw_probs = _parse_probs(raw, n)
        posterior = _normalize(raw_probs) if raw_probs is not None else [1.0 / n] * n
        predicted = posterior.index(max(posterior))
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": raw_probs is not None},
        )


class CoTRankListener(BaseListener):
    """Chain-of-thought scoring given full object feature text + image."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"cot-rank({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        n = len(scene.objects)
        prompt = _SYSTEM_COT_RANK.format(
            n=n, n1=n - 1, object_list=_format_objects(scene), utterance=utterance.text,
        )
        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="Score and rank each object."),
            ],
            image_path=scene.image_path,
        )
        raw_probs = _parse_tagged_probs(raw, "PROBS", n) or _parse_tagged_probs(raw, "SCORES", n)
        posterior = _normalize(raw_probs) if raw_probs is not None else [1.0 / n] * n
        predicted = posterior.index(max(posterior))
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": raw_probs is not None},
        )


class EliminationListener(BaseListener):
    """Elimination strategy given full object feature text + image."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"elimination({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        n = len(scene.objects)
        prompt = _SYSTEM_ELIMINATION.format(
            n=n, object_list=_format_objects(scene), utterance=utterance.text,
        )
        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="Eliminate and rank."),
            ],
            image_path=scene.image_path,
        )
        raw_probs = _parse_tagged_probs(raw, "PROBS", n)
        posterior = _normalize(raw_probs) if raw_probs is not None else [1.0 / n] * n
        predicted = posterior.index(max(posterior))
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": raw_probs is not None},
        )


# ── Image-only helpers ────────────────────────────────────────────────────────

def _annotate_indices(image_path: str, objects, canvas_w=330, canvas_h=328, margin=5) -> str:
    """Overlay index numbers at each object's center. Returns path to temp PNG."""
    import tempfile
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    sx = w / canvas_w
    mx = margin * sx
    my = margin * (h / canvas_h)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(12, int(14 * sx)))
    except Exception:
        font = ImageFont.load_default()

    for i, obj in enumerate(objects):
        px = int(mx + obj.x_loc / 100 * (w - 2 * mx))
        py = int((h - my) - obj.y_loc / 100 * (h - 2 * my))
        label = str(i)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((px + dx - 5, py + dy - 8), label, fill="white", font=font)
        draw.text((px - 5, py - 8), label, fill="black", font=font)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


def _io_listen(client, scene, utterance, prompt_tmpl, parse_fn, user_msg):
    """Shared: annotate image with indices, call VLM, parse posterior."""
    n = len(scene.objects)
    annotated = _annotate_indices(scene.image_path, scene.objects)
    prompt = prompt_tmpl.format(n=n, n1=n - 1, utterance=utterance.text)
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


# ── Image-only prompt templates ───────────────────────────────────────────────

_IO_INDEX = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and position.

The speaker said: "{utterance}"

Your task: identify which single numbered object the speaker is referring to.
Use ONLY what you see in the image.

Output ONLY a single integer (the label of the object). No explanation."""

_IO_DIRECT = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and position.

The speaker said: "{utterance}"

Assign a probability to each numbered object being the one the speaker refers to.
Use ONLY what you see in the image. Probabilities must sum to 1.0.

Output ONLY a JSON array of {n} floats:  [p0, p1, ..., p{n1}]
No explanation."""

_IO_COT = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and size.

The speaker said: "{utterance}"

Reason step by step using only what you see:
1. OBSERVE — for each label 0..{n1}, describe what you see (color, shape, size).
2. MATCH — score how well the utterance describes each object (0 = no match, 10 = perfect).
3. ASSIGN — convert scores to probabilities summing to 1.0.

Output format (follow exactly):
SCORES: [s0, s1, ...]   ← integers 0–10
PROBS:  [p0, p1, ...]   ← floats summing to 1.0"""

_IO_ELIMINATION = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and position.

The speaker said: "{utterance}"

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


# ── Honest image-only listeners ───────────────────────────────────────────────

class ImageOnlyDirectRankListener(BaseListener):
    """
    Direct probability assignment from indexed image + utterance only.
    No object feature text is provided — purely visual reasoning.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"io-direct({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        raw, posterior, predicted, parse_ok = _io_listen(
            self.client, scene, utterance, _IO_DIRECT,
            _parse_probs, "Assign probabilities based on what you see.",
        )
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )


class ImageOnlyCoTRankListener(BaseListener):
    """
    Chain-of-thought scoring from indexed image + utterance only.
    Observe each object visually, score 0–10, then convert to probabilities.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"io-cot({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        def _parse(raw, n):
            return _parse_tagged_probs(raw, "PROBS", n) or _parse_tagged_probs(raw, "SCORES", n)

        raw, posterior, predicted, parse_ok = _io_listen(
            self.client, scene, utterance, _IO_COT,
            _parse, "Observe, score, then assign probabilities.",
        )
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )


class ImageOnlyEliminationListener(BaseListener):
    """
    Elimination strategy from indexed image + utterance only.
    Visually rule out non-matching objects, then distribute probability.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"io-elimination({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        raw, posterior, predicted, parse_ok = _io_listen(
            self.client, scene, utterance, _IO_ELIMINATION,
            lambda r, n: _parse_tagged_probs(r, "PROBS", n),
            "Eliminate visually, then assign probabilities.",
        )
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )


class ImageOnlyIndexListener(BaseListener):
    """
    Hardest-commit IO listener: output a single integer index, no distribution.

    The model picks one object and commits absolutely — max posterior is always
    1.0, so the EU rule never triggers clarification and CPA == Accuracy.
    This is the simplest possible IO listener and serves as an upper-bound
    reference for ask-rate-free performance.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"io-index({self.client.model.split('/')[-1]})"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")
        n = len(scene.objects)
        annotated = _annotate_indices(scene.image_path, scene.objects)
        prompt = _IO_INDEX.format(n=n, n1=n - 1, utterance=utterance.text)
        raw = self.client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content="Which object index is the speaker referring to?"),
            ],
            image_path=annotated,
        )
        # Parse: first integer in response, clamped to [0, n-1]
        m = re.search(r"\b(\d+)\b", raw.strip())
        if m:
            idx = int(m.group(1))
            parse_ok = 0 <= idx < n
            predicted = idx if parse_ok else 0
        else:
            predicted = 0
            parse_ok = False
        # Hard posterior: probability 1 on the chosen index
        posterior = [0.0] * n
        posterior[predicted] = 1.0
        return ListenerOutput(
            posterior=posterior, predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={"raw_response": raw, "parse_ok": parse_ok},
        )
