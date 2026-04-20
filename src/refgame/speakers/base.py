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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
