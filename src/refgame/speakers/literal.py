"""
Literal Speaker (rule-based baseline).

Strategy: find the *minimal* subset of features that uniquely identifies
the target among distractors, then emit those features as a natural phrase.
Tie-break: prefer features in canonical order (color > shape > size > location).

This speaker is deterministic and requires no API calls.
"""

from __future__ import annotations

from itertools import combinations

from ..data.schema import FEATURE_KEYS, Object, Scene, Utterance
from .base import BaseSpeaker


class LiteralSpeaker(BaseSpeaker):
    """
    Emits the shortest feature subset that uniquely identifies the target.

    If no single feature distinguishes the target, adds features one by one
    (in canonical order) until the description is unambiguous.
    """

    @property
    def name(self) -> str:
        return "literal"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        target     = scene.objects[target_idx]
        distractors = [o for i, o in enumerate(scene.objects) if i != target_idx]

        combo = self._minimal_distinguishing_combo(target, distractors)
        text  = self._render(target, combo)

        return Utterance(
            text=text,
            speaker_type=self.name,
            feature_combo=combo,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _minimal_distinguishing_combo(
        self,
        target: Object,
        distractors: list[Object],
    ) -> tuple[str, ...]:
        """Return the smallest feature subset that rules out all distractors."""
        target_feats = target.features()
        for r in range(1, len(FEATURE_KEYS) + 1):
            for combo in combinations(FEATURE_KEYS, r):
                vals = {k: target_feats[k] for k in combo}
                if not any(
                    all(d.features()[k] == v for k, v in vals.items())
                    for d in distractors
                ):
                    return combo
        # Fallback: full description (should always be reachable for valid scenes)
        return FEATURE_KEYS

    @staticmethod
    def _render(obj: Object, combo: tuple[str, ...]) -> str:
        """Convert a feature combo into a fluent English phrase."""
        feats = obj.features()
        parts: list[str] = []
        loc_part = ""
        for key in ("size", "color", "shape"):
            if key in combo:
                parts.append(feats[key])
        noun = feats["shape"] if "shape" in combo else "one"
        if "location" in combo:
            loc_part = f" on the {feats['location']}"
        inner = " ".join(parts[: -1 if "shape" in combo else len(parts)])
        if "shape" in combo:
            desc = " ".join([p for p in parts if p != feats["shape"]] + [noun])
        else:
            desc = " ".join(parts) + f" {noun}"
        return f"the {desc.strip()}{loc_part}"
