"""
LLM Speaker: uses a language model (via OpenRouter) to generate utterances.

Two modes
---------
naive     : single-turn prompt, no explicit reasoning.
pragmatic : chain-of-thought prompt instructing the model to reason about
            which features distinguish the target before producing its expression.

The LLM speaker is the most expensive but provides an upper bound on
utterance quality and tests whether LLMs can reason pragmatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..data.schema import Scene, Utterance
from ..utils.llm_client import LLMClient, ChatMessage
from .base import BaseSpeaker


# ── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_NAIVE = """\
You are a speaker in a reference game. Given a scene with several objects, \
produce a SHORT referring expression that uniquely identifies the TARGET object \
so that a listener can pick it out.

Rules:
- Use only visual attributes (color, shape, size, spatial location).
- Be as brief as possible — omit attributes that don't help.
- Output ONLY the referring expression (e.g. "the red circle"). No other text."""

_SYSTEM_PRAGMATIC = """\
You are a pragmatic speaker in a reference game modeled on Rational Speech \
Acts (RSA) theory.

Think step by step:
1. List all features of the TARGET object.
2. For each feature, check whether it rules out ALL distractors.
3. Find the minimal set of features that uniquely identifies the target.
4. Produce the shortest natural English expression using only those features.

Output format (follow exactly):
REASONING: <your analysis>
EXPRESSION: <the referring expression>"""


def _format_scene(scene: Scene, target_idx: int) -> str:
    lines = ["SCENE:"]
    for i, obj in enumerate(scene.objects):
        marker = "  ← TARGET" if i == target_idx else ""
        lines.append(
            f"  [{obj.id}] {obj.size} {obj.color} {obj.shape} "
            f"at {obj.location}{marker}"
        )
    return "\n".join(lines)


# ── Speaker ──────────────────────────────────────────────────────────────────

class LLMSpeaker(BaseSpeaker):
    """
    Speaker that calls an LLM to produce a referring expression.

    Parameters
    ----------
    client    : LLMClient wrapping the API (OpenRouter or Anthropic).
    pragmatic : if True, use chain-of-thought prompt.
    """

    def __init__(self, client: LLMClient, pragmatic: bool = False) -> None:
        self.client    = client
        self.pragmatic = pragmatic

    @property
    def name(self) -> str:
        mode  = "pragmatic" if self.pragmatic else "naive"
        model = self.client.model.split("/")[-1]   # e.g. "gpt-4o"
        return f"llm-{mode}({model})"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        system  = _SYSTEM_PRAGMATIC if self.pragmatic else _SYSTEM_NAIVE
        user    = _format_scene(scene, target_idx)
        raw     = self.client.complete([
            ChatMessage(role="system", content=system),
            ChatMessage(role="user",   content=user),
        ])

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


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _extract_expression(text: str) -> str:
    m = re.search(r"EXPRESSION:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"')
    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip('"') if lines else text.strip()


def _extract_reasoning(text: str) -> str | None:
    m = re.search(r"REASONING:\s*(.+?)(?=EXPRESSION:|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None
