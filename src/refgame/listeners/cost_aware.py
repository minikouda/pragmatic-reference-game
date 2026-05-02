"""
Cost-Aware Listener: wraps any BaseListener and adds a clarification decision.

Decision rule (Expected Utility)
---------------------------------
Given posterior P(t_i | u):

    E[U | commit] = max_i P(t_i | u)       ← expected accuracy if we guess argmax
    E[U | ask]    = 1 − c                  ← expected accuracy after asking (assumed perfect resolution)
                                             minus the clarification cost c

    action = "ask"    if E[U | ask]    > E[U | commit]
           = "commit" otherwise

This is equivalent to:
    ask iff max_i P(t_i | u) < 1 − c
    i.e., the listener's confidence is below the (1 − c) threshold.

The listener also generates a natural-language clarification question
(via a question template or an LLM call) so the full pipeline can be
evaluated end-to-end.

Extension: `CostAwareLLMListener` overrides question generation with an LLM
to produce more natural questions.
"""

from __future__ import annotations

from ..data.schema import (
    ClarificationDecision, ListenerOutput, Scene, Utterance,
)
from .base import BaseListener


# ── Template-based question generation ───────────────────────────────────────

def _generate_question_template(scene: Scene, posterior: list[float]) -> str:
    """
    Produce a simple clarification question by asking about the most
    discriminative feature among the top-2 candidate objects.
    """
    # Find top-2 candidates
    indexed = sorted(enumerate(posterior), key=lambda x: -x[1])
    if len(indexed) < 2:
        return "Could you be more specific?"

    top1_obj = scene.objects[indexed[0][0]]
    top2_obj = scene.objects[indexed[1][0]]

    # Find first feature dimension where they differ
    for key in ("color", "shape", "size", "location"):
        v1 = top1_obj.features()[key]
        v2 = top2_obj.features()[key]
        if v1 != v2:
            return f"Are you referring to the {v1} or the {v2} one?"

    return "Could you describe the object more specifically?"


# ── Cost-Aware Listener ───────────────────────────────────────────────────────

class CostAwareListener(BaseListener):
    """
    Wraps a base listener with an expected-utility clarification policy.

    Parameters
    ----------
    base_listener : underlying listener (literal or rsa)
    cost_c        : clarification cost in [0, 1]
                    lower → more willing to ask
    """

    def __init__(self, base_listener: BaseListener, cost_c: float = 0.25) -> None:
        if not (0.0 <= cost_c <= 1.0):
            raise ValueError(f"cost_c must be in [0, 1]; got {cost_c}")
        self.base_listener = base_listener
        self.cost_c        = cost_c

    @property
    def name(self) -> str:
        return f"cost_aware({self.base_listener.name},c={self.cost_c})"

    def listen(self, scene: Scene, utterance: Utterance, cost_c: float = 0.25) -> ListenerOutput:
        """
        Run the base listener and attach a clarification decision.

        The `listener_meta` dict of the returned output contains the full
        `ClarificationDecision` under the key `"clarification"`.
        """
        base_out  = self.base_listener.listen(scene, utterance)
        decision  = self._decide(scene, base_out)

        # Propagate decision into meta so downstream code can inspect it
        out = ListenerOutput(
            posterior=base_out.posterior,
            predicted_idx=base_out.predicted_idx,
            listener_type=self.name,
            listener_meta={**base_out.listener_meta, "clarification": decision},
        )
        return out

    def decide(self, scene: Scene, utterance: Utterance) -> ClarificationDecision:
        """Convenience: run listener and return only the clarification decision."""
        base_out = self.base_listener.listen(scene, utterance)
        return self._decide(scene, base_out)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _decide(self, scene: Scene, base_out: ListenerOutput) -> ClarificationDecision:
        eu_commit = max(base_out.posterior)
        eu_ask    = 1.0 - self.cost_c

        if eu_ask > eu_commit:
            action   = "ask"
            question = _generate_question_template(scene, base_out.posterior)
        else:
            action   = "commit"
            question = None

        return ClarificationDecision(
            action=action,
            eu_commit=eu_commit,
            eu_ask=eu_ask,
            cost_c=self.cost_c,
            question=question,
            base_output=base_out,
        )
