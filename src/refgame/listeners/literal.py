"""
Literal Listener (L0): rule-based baseline.

L0 assigns a uniform distribution over all objects whose feature set
contains every content word in the utterance.  This is the standard
RSA L0 — no pragmatic reasoning, no speaker model.

Token matching is intentionally simple (whitespace-split, stopword removal)
so that it mirrors the linguistic content of the utterance without any
embedding-level similarity.
"""

from __future__ import annotations

from ..data.schema import FEATURE_VOCAB, ListenerOutput, Scene, Utterance
from .base import BaseListener

# All valid feature value strings (flat set for fast O(1) lookup)
_ALL_FEATURE_VALUES: frozenset[str] = frozenset(
    v for vals in FEATURE_VOCAB.values() for v in vals
)
_STOP_WORDS = frozenset({"the", "a", "an", "on", "at", "in", "of", "one"})


def _content_tokens(text: str) -> frozenset[str]:
    """Return feature-vocabulary tokens present in the utterance."""
    tokens = frozenset(text.lower().split()) - _STOP_WORDS
    return tokens & _ALL_FEATURE_VALUES


class LiteralListener(BaseListener):
    """
    L0: uniform posterior over compatible objects.

    An object is *compatible* if every content token in the utterance
    matches one of its feature values.
    """

    @property
    def name(self) -> str:
        return "literal"

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        query_tokens = _content_tokens(utterance.text)

        compatible: list[int] = []
        for i, obj in enumerate(scene.objects):
            if query_tokens <= obj.feature_set():
                compatible.append(i)

        n = len(compatible) if compatible else len(scene.objects)
        if not compatible:
            # Utterance matched nothing — fall back to uniform (graceful degradation)
            compatible = list(range(len(scene.objects)))

        posterior = [
            1.0 / n if i in compatible else 0.0
            for i in range(len(scene.objects))
        ]
        predicted_idx = compatible[0]   # first compatible object (arbitrary tie-break)
        # If multiple, pick the one that appears first — evaluation harness handles ties

        return ListenerOutput(
            posterior=posterior,
            predicted_idx=posterior.index(max(posterior)),
            listener_type=self.name,
            listener_meta={"n_compatible": len(compatible), "query_tokens": list(query_tokens)},
        )
