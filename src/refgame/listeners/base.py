"""
Abstract listener interface.

All listeners implement `listen(scene, utterance) -> ListenerOutput`.
The returned posterior is a probability distribution over scene objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.schema import ListenerOutput, Scene, Utterance


class BaseListener(ABC):
    """Abstract base for all listener models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result tables."""
        ...

    @abstractmethod
    def listen(self, scene: Scene, utterance: Utterance) -> ListenerOutput:
        """
        Infer which object the speaker intended.

        Parameters
        ----------
        scene     : full scene
        utterance : the speaker's referring expression

        Returns
        -------
        ListenerOutput with posterior P(object_i | utterance) for all i.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
