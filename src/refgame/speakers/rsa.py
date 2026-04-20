"""
RSA Speaker (S1): pragmatic speaker from Rational Speech Acts theory.

Frank & Goodman (2012): the S1 speaker chooses utterances that maximize
    U(u; t) = α · log L0(t | u) − cost(u)
then normalizes via softmax over all candidate utterances.

The speaker generates all feature-subset utterances for the target
(same enumeration as the literal speaker) and ranks them by S1 score.
By default it returns the highest-probability utterance deterministically
(mode); set `sample=True` for stochastic sampling.
"""

from __future__ import annotations

import math
import random
from itertools import combinations

from ..data.schema import FEATURE_KEYS, Object, Scene, Utterance
from .base import BaseSpeaker
from ..listeners.literal import LiteralListener   # forward use — see note below


class RSASpeaker(BaseSpeaker):
    """
    Pragmatic speaker (S1) using RSA.

    Parameters
    ----------
    alpha       : rationality / temperature (higher → more peaked)
    cost_weight : cost per word in the utterance (encourages brevity)
    sample      : if True, sample from S1 distribution; otherwise take argmax
    seed        : RNG seed for sampling mode
    """

    def __init__(
        self,
        alpha:       float = 4.0,
        cost_weight: float = 0.1,
        sample:      bool  = False,
        seed:        int | None = None,
    ) -> None:
        self.alpha       = alpha
        self.cost_weight = cost_weight
        self.sample      = sample
        self._rng        = random.Random(seed)
        self._l0         = LiteralListener()

    @property
    def name(self) -> str:
        return f"rsa(α={self.alpha},c={self.cost_weight})"

    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        target     = scene.objects[target_idx]
        candidates = self._enumerate_utterances(target)

        # S1 scores
        scores: dict[str, float] = {}
        for utt_text in candidates:
            l0_out  = self._l0.listen(scene, _stub_utterance(utt_text))
            p_target = l0_out.posterior[target_idx]
            informativeness = math.log(max(p_target, 1e-12))
            cost            = self.cost_weight * len(utt_text.split())
            scores[utt_text] = self.alpha * informativeness - cost

        # Softmax
        max_s = max(scores.values())
        exp_s = {u: math.exp(s - max_s) for u, s in scores.items()}
        total = sum(exp_s.values())
        probs = {u: e / total for u, e in exp_s.items()}

        if self.sample:
            texts  = list(probs.keys())
            weights = list(probs.values())
            chosen = self._rng.choices(texts, weights=weights, k=1)[0]
        else:
            chosen = max(probs, key=probs.__getitem__)

        # Find which combo produced this text
        combo = self._text_to_combo(target, chosen)

        return Utterance(
            text=chosen,
            speaker_type=self.name,
            feature_combo=combo,
            speaker_meta={"s1_prob": probs[chosen], "n_candidates": len(candidates)},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enumerate_utterances(self, obj: Object) -> list[str]:
        feats = obj.features()
        utts: list[str] = []
        for r in range(1, len(FEATURE_KEYS) + 1):
            for combo in combinations(FEATURE_KEYS, r):
                utts.append(_render(feats, combo))
        return utts

    def _text_to_combo(self, obj: Object, text: str) -> tuple[str, ...] | None:
        feats = obj.features()
        for r in range(1, len(FEATURE_KEYS) + 1):
            for combo in combinations(FEATURE_KEYS, r):
                if _render(feats, combo) == text:
                    return combo
        return None


# ── Shared render logic (mirrors LiteralSpeaker) ──────────────────────────────

def _render(feats: dict[str, str], combo: tuple[str, ...]) -> str:
    parts: list[str] = []
    loc_part = ""
    for key in ("size", "color", "shape"):
        if key in combo:
            parts.append(feats[key])
    noun = feats["shape"] if "shape" in combo else "one"
    if "location" in combo:
        loc_part = f" on the {feats['location']}"
    if "shape" in combo:
        desc = " ".join([p for p in parts if p != feats["shape"]] + [noun])
    else:
        desc = " ".join(parts) + f" {noun}"
    return f"the {desc.strip()}{loc_part}"


def _stub_utterance(text: str) -> "Utterance":
    """Create a minimal Utterance for passing to L0 without circular import issues."""
    from ..data.schema import Utterance
    return Utterance(text=text, speaker_type="_rsa_internal")
