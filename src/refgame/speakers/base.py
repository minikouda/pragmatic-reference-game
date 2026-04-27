"""
Abstract speaker interface.

All speakers implement one method: `speak(scene, target_idx) -> Utterance`.
This clean contract lets the evaluation harness treat all speakers uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.schema import Scene, Utterance


class BaseSpeaker(ABC):
    """Abstract base for all speaker models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result tables (e.g. 'literal', 'rsa', 'llm-gpt4o')."""
        ...

    @abstractmethod
    def speak(self, scene: Scene, target_idx: int) -> Utterance:
        """
        Produce a referring expression for `scene.objects[target_idx]`.

        Parameters
        ----------
        scene      : full scene (objects + optional annotation)
        target_idx : index of the target object in scene.objects

        Returns
        -------
        Utterance with `.text` set and `.speaker_type` = self.name
        """
        ...

    def answer_question(
        self, scene: Scene, target_idx: int, question: str
    ) -> str:
        """
        Answer a listener's clarifying question about the target object.

        Default implementation raises NotImplementedError.
        Speakers that participate in dialogue must override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support answer_question(). "
            "Override this method to enable dialogue."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
