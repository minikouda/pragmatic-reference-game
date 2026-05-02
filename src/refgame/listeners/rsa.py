"""
RSA Listener (L1): pragmatic listener from Rational Speech Acts theory.

L1 reasons about the speaker:
    L1(t | u) ∝ S1(u | t) · P(t)
    S1(u | t) ∝ exp(α · log L0(t | u) − cost(u))

For each candidate object t_i, we:
  1. Enumerate all utterances for t_i.
  2. Compute S1(u_obs | t_i) — the pragmatic speaker's probability of
     producing the observed utterance if t_i were the target.
  3. Normalize to get L1(t_i | u_obs).

This is exact (not approximate) RSA using enumerated utterances,
feasible because our vocabulary is small and finite.
"""

from __future__ import annotations

import math

from ..data.schema import FEATURE_KEYS, ListenerOutput, Scene, Utterance
from .base import BaseListener
from .literal import LiteralListener, _content_tokens


class RSAListener(BaseListener):
    """
    Pragmatic listener (L1).

    Parameters
    ----------
    alpha       : rationality parameter (higher → more peaked S1)
    cost_weight : per-word cost applied to S1 speaker model
    """

    def __init__(self, alpha: float = 4.0, cost_weight: float = 0.1) -> None:
        self.alpha       = alpha
        self.cost_weight = cost_weight
        self._l0         = LiteralListener()

    @property
    def name(self) -> str:
        return f"rsa(α={self.alpha},c={self.cost_weight})"

    def listen(self, scene: Scene, utterance: Utterance, cost_c: float = 0.25) -> ListenerOutput:
        utt_text = utterance.text
        obj_scores: list[float] = []

        for i, obj in enumerate(scene.objects):
            # Build a hypothetical scene where obj is the target
            # then compute S1(utt_text | obj as target)
            s1_prob = self._s1_prob(scene, target_idx=i, utt_text=utt_text)
            obj_scores.append(s1_prob)

        total = sum(obj_scores) or 1e-30
        posterior = [s / total for s in obj_scores]
        predicted_idx = posterior.index(max(posterior))

        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted_idx,
            listener_type=self.name,
            listener_meta={"alpha": self.alpha, "cost_weight": self.cost_weight},
        )

    # ── S1 computation ────────────────────────────────────────────────────────

    def _s1_prob(self, scene: Scene, target_idx: int, utt_text: str) -> float:
        """P_{S1}(utt_text | target=scene.objects[target_idx])."""
        from itertools import combinations
        from ..data.schema import FEATURE_KEYS
        from ..speakers.rsa import _render    # shared render fn

        target = scene.objects[target_idx]
        feats  = target.features()

        # Enumerate all candidate utterances for this object
        candidates: list[str] = []
        for r in range(1, len(FEATURE_KEYS) + 1):
            for combo in combinations(FEATURE_KEYS, r):
                candidates.append(_render(feats, combo))

        # Score each candidate via L0
        scores: dict[str, float] = {}
        for cand in candidates:
            stub = Utterance(text=cand, speaker_type="_rsa_internal")
            l0   = self._l0.listen(scene, stub)
            p_t  = l0.posterior[target_idx]
            info = math.log(max(p_t, 1e-12))
            cost = self.cost_weight * len(cand.split())
            scores[cand] = self.alpha * info - cost

        # Softmax
        max_s = max(scores.values())
        exp_s = {u: math.exp(s - max_s) for u, s in scores.items()}
        total = sum(exp_s.values())

        return exp_s.get(utt_text, 0.0) / total
