"""
Abstract listener interface.

All listeners implement `listen(scene, utterance) -> ListenerOutput`.
The returned posterior is a probability distribution over scene objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.schema import ListenerOutput, Scene, Utterance


def annotate_indices(image_path: str, objects, canvas_w=330, canvas_h=328, margin=5) -> str:
    """Overlay index numbers at each object's center. Returns path to temp PNG."""
    import tempfile
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    sx = w / canvas_w
    mx = margin * sx
    my = margin * (h / canvas_h)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(12, int(14 * sx)))
    except Exception:
        font = ImageFont.load_default()

    for i, obj in enumerate(objects):
        px = int(mx + obj.x_loc / 100 * (w - 2 * mx))
        py = int((h - my) - obj.y_loc / 100 * (h - 2 * my))
        label = str(i)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((px + dx - 5, py + dy - 8), label, fill="white", font=font)
        draw.text((px - 5, py - 8), label, fill="black", font=font)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


class BaseListener(ABC):
    """Abstract base for all listener models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result tables."""
        ...

    @abstractmethod
    def listen(self, scene: Scene, utterance: Utterance, cost_c: float = 0.25) -> ListenerOutput:
        """
        Infer which object the speaker intended.

        Parameters
        ----------
        scene     : full scene
        utterance : the speaker's referring expression
        cost_c    : clarification cost in [0, 1]. Passed into the prompt so
                    the model knows the commit threshold (1 - cost_c) when
                    assigning confidence to each object.

        Returns
        -------
        ListenerOutput with posterior P(object_i | utterance) for all i.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
