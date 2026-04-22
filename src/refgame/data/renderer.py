"""
Scene renderer: generates PNG images matching the reference_game_dataset style.

Reverse-engineered rendering parameters (from pixel measurements):
  Canvas   : 330 × 328 px, white background, 1px black border
  Margin   : 5 px inside the border
  Coords   : x_loc ∈ [0,100] maps left→right (normal)
             y_loc ∈ [0,100] maps bottom→top (mathematical / y-inverted)
  Scale    : pixel_x = MARGIN + x_loc/100 * (W - 2*MARGIN)
             pixel_y = (H - MARGIN) - y_loc/100 * (H - 2*MARGIN)
  Size     : half_size (radius or square half-width) = raw_size * SIZE_SCALE
             raw_size ∈ {8, 12, 16} → {small, medium, large}
  Shapes   : circle  — filled ellipse inscribed in bounding box
             square  — filled axis-aligned rectangle
             triangle — filled upward-pointing equilateral triangle

Color map aligns with dataset vocabulary (Pillow RGB tuples):
  black, blue, green, red, yellow
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

# ── Rendering constants ──────────────────────────────────────────────────────

CANVAS_W   = 330
CANVAS_H   = 328
MARGIN     = 5          # px inside the border on each side
SIZE_SCALE = 3          # half_size (px) = raw_size * SIZE_SCALE
BORDER_W   = 1          # black border width

COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "black":  (0,   0,   0),
    "blue":   (30,  144, 255),   # dodger blue, matches dataset
    "green":  (50,  205, 50),    # lime green
    "red":    (220, 50,  50),
    "yellow": (255, 215, 0),
    # Extended palette for synthetically generated scenes
    "purple": (148, 0,   211),
    "orange": (255, 140, 0),
    "pink":   (255, 105, 180),
    "brown":  (139, 69,  19),
}


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _to_pixel(x_loc: float, y_loc: float) -> tuple[int, int]:
    """Convert (x_loc, y_loc) in [0,100]² to pixel (px, py) on the canvas."""
    draw_w = CANVAS_W - 2 * MARGIN
    draw_h = CANVAS_H - 2 * MARGIN
    px = int(MARGIN + x_loc / 100 * draw_w)
    py = int((CANVAS_H - MARGIN) - y_loc / 100 * draw_h)
    return px, py


def _half_size(raw_size: int) -> int:
    return raw_size * SIZE_SCALE


# ── Shape drawers ─────────────────────────────────────────────────────────────

def _draw_circle(draw, cx: int, cy: int, r: int, color: tuple) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def _draw_square(draw, cx: int, cy: int, r: int, color: tuple) -> None:
    draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)


def _draw_triangle(draw, cx: int, cy: int, r: int, color: tuple) -> None:
    """Upward-pointing equilateral triangle centered at (cx, cy) with half-size r."""
    # Apex at top, base at bottom (pixel space has y increasing downward,
    # but y_loc is inverted so "up" in data = lower pixel y)
    apex   = (cx,     cy - r)
    base_l = (cx - r, cy + r)
    base_r = (cx + r, cy + r)
    draw.polygon([apex, base_l, base_r], fill=color)


_SHAPE_DRAWERS = {
    "circle":   _draw_circle,
    "square":   _draw_square,
    "triangle": _draw_triangle,
}


# ── Public API ────────────────────────────────────────────────────────────────

def render_scene(
    objects: list[dict],
    out_path: str | Path,
    highlight_idx: int | None = None,
    highlight_color: str = "red",
) -> Path:
    """
    Render a list of raw object dicts to a PNG file matching the dataset style.

    Parameters
    ----------
    objects        : list of dicts with keys {x_loc, y_loc, color, type, size}
                     where size is numeric (8 / 12 / 16) or string ("small"/…)
    out_path       : output file path (created including parents)
    highlight_idx  : if set, draw a bounding box around objects[highlight_idx]
    highlight_color: color of the highlight box

    Returns
    -------
    Path to the written PNG.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError("Pillow is required for rendering: pip install pillow") from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img  = Image.new("RGB", (CANVAS_W, CANVAS_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw each object
    for i, obj in enumerate(objects):
        raw_size = obj["size"] if isinstance(obj["size"], int) else _size_str_to_int(obj["size"])
        color    = COLOR_MAP.get(obj["color"], (128, 128, 128))
        shape    = obj.get("type") or obj.get("shape", "circle")
        px, py   = _to_pixel(obj["x_loc"], obj["y_loc"])
        r        = _half_size(raw_size)

        drawer = _SHAPE_DRAWERS.get(shape)
        if drawer is None:
            raise ValueError(f"Unknown shape: {shape!r}")
        drawer(draw, px, py, r, color)

    # Draw highlight box
    if highlight_idx is not None:
        obj      = objects[highlight_idx]
        raw_size = obj["size"] if isinstance(obj["size"], int) else _size_str_to_int(obj["size"])
        px, py   = _to_pixel(obj["x_loc"], obj["y_loc"])
        r        = _half_size(raw_size) + 4   # slight padding
        hc       = COLOR_MAP.get(highlight_color, (255, 0, 0))
        draw.rectangle([px - r, py - r, px + r, py + r], outline=hc, width=3)

    # Draw canvas border
    draw.rectangle(
        [0, 0, CANVAS_W - 1, CANVAS_H - 1],
        outline=(0, 0, 0),
        width=BORDER_W,
    )

    img.save(out_path, "PNG")
    return out_path


def render_dataset(
    scenes_raw: list[dict],
    out_dir: str | Path,
    prefix: str = "scene",
) -> list[str]:
    """
    Render a list of raw scene dicts (as in dataset.json format) to PNG files.

    Parameters
    ----------
    scenes_raw : list of {"objects": [...]} dicts
    out_dir    : directory to write images into
    prefix     : filename prefix (e.g. "scene" → scene_0.png, scene_1.png, …)

    Returns
    -------
    List of relative image paths (same format as dataset.json).
    """
    out_dir   = Path(out_dir)
    img_paths = []
    for i, scene in enumerate(scenes_raw):
        path = out_dir / f"{prefix}_{i}.png"
        render_scene(scene["objects"], path)
        img_paths.append(str(path))
    return img_paths


def render_scene_from_objects(
    objects: "list[Object]",  # type: ignore[name-defined]
    out_path: str | Path,
    highlight_idx: int | None = None,
) -> Path:
    """
    Render from typed Object instances (used by the generator pipeline).
    Converts symbolic size strings back to numeric for rendering.
    """
    SIZE_STR_TO_INT = {"small": 8, "medium": 12, "large": 16}
    raw_dicts = [
        {
            "x_loc": obj.x_loc,
            "y_loc": obj.y_loc,
            "color": obj.color,
            "type":  obj.shape,
            "size":  SIZE_STR_TO_INT.get(obj.size, 12),
        }
        for obj in objects
    ]
    return render_scene(raw_dicts, out_path, highlight_idx=highlight_idx)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _size_str_to_int(s: str) -> int:
    return {"small": 8, "medium": 12, "large": 16}.get(s, 12)
