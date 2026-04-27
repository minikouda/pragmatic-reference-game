"""
DialogueListener: multi-turn image-only listener with active clarification.

When the EU rule fires (max posterior < 1 - c), instead of abstaining the
listener generates a targeted natural-language question aimed at resolving
the ambiguity between its top-2 confused candidates.  The speaker answers,
and the listener re-evaluates its posterior using the full context:
original utterance + Q&A history.  This repeats for up to `max_rounds`
turns, then commits to the current argmax.

Architecture
------------
  Round 0 (initial):
    annotated_image + utterance  → VLM → posterior_0

  Round k (clarification, if max(posterior_{k-1}) < 1 - c):
    annotated_image + utterance + posterior_{k-1}
      → question_model → question_k
    plain_image + target_desc + question_k
      → speaker_model → answer_k
    annotated_image + utterance + [(q_1,a_1), ..., (q_k,a_k)]
      → VLM → posterior_k

  Final: commit to argmax of posterior after last round.

Each dialogue turn costs 2 extra LLM calls (question + answer).
"""

from __future__ import annotations

from ..data.schema import ListenerOutput, Scene, Utterance
from ..utils.llm_client import ChatMessage, LLMClient
from ..speakers.base import BaseSpeaker
from .base import BaseListener
from .direct_rank import _annotate_indices, _parse_probs, _normalize


# ── Prompts ───────────────────────────────────────────────────────────────────

_CALIBRATED_PRIOR = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
Look at each numbered object carefully — note its color, shape, and position.

The speaker said: "{utterance}"

Assign a probability to each numbered object being the one the speaker refers to.
Use ONLY what you see in the image. Probabilities must sum to 1.0.

IMPORTANT: Be honest about uncertainty. If two objects could both match the
description, split probability between them rather than assigning 1.0 to one.
Only assign probability ≥ 0.9 if you are very confident exactly one object matches.

Output ONLY a JSON array of {n} floats:  [p0, p1, ..., p{n1}]
No explanation."""

_QUESTION_PROMPT = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.
The speaker said: "{utterance}"

You are uncertain. Your current best guesses are objects {top2} \
(you cannot distinguish between them from the utterance alone).
{prior_qa}
Generate ONE short clarifying question that would help you decide between \
those candidates. The question should ask about a single visual property \
(color, shape, size, or position) that differs between them.
Do NOT repeat a question that has already been asked above.

Examples:
  "Is it on the left side or the right side?"
  "Is it the larger or the smaller one?"
  "Is it a circle or a triangle?"

Output ONLY the question. No explanation."""


_UPDATE_PROMPT = """\
You are a listener in a visual reference game.

The image shows {n} objects labelled 0 to {n1}.

The speaker originally said: "{utterance}"

You asked for clarification and received the following exchange:
{qa_history}

Using the original description AND the clarifications above, assign an \
updated probability to each numbered object.
Use ONLY what you see in the image and what the speaker told you.
Probabilities must sum to 1.0.

Output ONLY a JSON array of {n} floats:  [p0, p1, ..., p{n1}]
No explanation."""


# ── DialogueListener ──────────────────────────────────────────────────────────

class DialogueListener(BaseListener):
    """
    Image-only listener that actively clarifies when uncertain.

    Parameters
    ----------
    listener_client : VLM for initial + updated posteriors and question generation
    speaker_client  : VLM that plays the speaker role (answers questions)
    cost_c          : EU threshold — ask when max(p) < 1 - cost_c
    max_rounds      : maximum clarification turns before forced commit
    """

    def __init__(
        self,
        listener_client: LLMClient,
        speaker:         BaseSpeaker,
        cost_c:          float = 0.5,
        max_rounds:      int   = 2,
    ) -> None:
        self.listener_client = listener_client
        self.speaker         = speaker
        self.cost_c          = cost_c
        self.max_rounds      = max_rounds

    @property
    def name(self) -> str:
        lm = self.listener_client.model.split("/")[-1]
        tag = f"dialogue(l={lm},s={self.speaker.name},c={self.cost_c},r={self.max_rounds})"
        return tag

    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        if scene.image_path is None:
            raise ValueError(f"Scene {scene.id} has no image_path.")

        n          = len(scene.objects)
        annotated  = _annotate_indices(scene.image_path, scene.objects)
        qa_history: list[tuple[str, str]] = []

        # ── Round 0: initial posterior ────────────────────────────────────────
        posterior = self._get_posterior(annotated, utterance.text, n, qa_history)

        # target_idx stored in speaker_meta by StrategicVLLMSpeaker
        target_idx: int = utterance.speaker_meta.get("target_idx", 0)

        # ── Clarification rounds ──────────────────────────────────────────────
        rounds_taken = 0
        for _ in range(self.max_rounds):
            if max(posterior) >= 1.0 - self.cost_c:
                break  # confident enough

            # Generate question targeting top-2 confused objects
            question = self._generate_question(
                annotated, utterance.text, n, posterior, qa_history
            )

            # Speaker answers by visually inspecting the image — genuine perception
            answer = self.speaker.answer_question(scene, target_idx, question)

            qa_history.append((question, answer))
            rounds_taken += 1

            # Update posterior with full dialogue context
            posterior = self._get_posterior(annotated, utterance.text, n, qa_history)

        predicted = posterior.index(max(posterior))
        return ListenerOutput(
            posterior=posterior,
            predicted_idx=predicted,
            listener_type=self.name,
            listener_meta={
                "rounds": rounds_taken,
                "qa_history": qa_history,
                "final_max_p": max(posterior),
            },
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_posterior(
        self,
        annotated_path: str,
        utterance_text: str,
        n: int,
        qa_history: list[tuple[str, str]],
    ) -> list[float]:
        if not qa_history:
            # Calibrated initial prompt: explicitly asks for honest uncertainty
            prompt = _CALIBRATED_PRIOR.format(n=n, n1=n - 1, utterance=utterance_text)
            user_msg = "Assign probabilities honestly reflecting your visual uncertainty."
        else:
            qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_history)
            prompt = _UPDATE_PROMPT.format(
                n=n, n1=n - 1,
                utterance=utterance_text,
                qa_history=qa_text,
            )
            user_msg = "Update your probability assignment given the clarifications."

        raw = self.listener_client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user",   content=user_msg),
            ],
            image_path=annotated_path,
        )
        probs = _parse_probs(raw, n)
        return _normalize(probs) if probs is not None else [1.0 / n] * n

    def _generate_question(
        self,
        annotated_path: str,
        utterance_text: str,
        n: int,
        posterior: list[float],
        qa_history: list[tuple[str, str]],
    ) -> str:
        # Top-2 confused candidates
        top2 = sorted(range(n), key=lambda i: -posterior[i])[:2]
        prior_qa = ""
        if qa_history:
            lines = "\n".join(f"  Q: {q}\n  A: {a}" for q, a in qa_history)
            prior_qa = f"\nYou have already asked:\n{lines}\n"
        prompt = _QUESTION_PROMPT.format(
            n=n, n1=n - 1,
            utterance=utterance_text,
            top2=top2,
            prior_qa=prior_qa,
        )
        raw = self.listener_client.complete(
            messages=[
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user",   content="What is your clarifying question?"),
            ],
            image_path=annotated_path,
        )
        # Strip surrounding quotes/punctuation
        return raw.strip().strip('"').strip("'").rstrip(".")

