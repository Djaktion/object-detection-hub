from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .types import Detection


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color for a given class id."""
    import colorsys

    # Spread hues deterministically; keep high saturation/value for visibility.
    h = (class_id * 0.61803398875) % 1.0  # golden ratio
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def _draw_legend(out: np.ndarray, class_names: list[str], counts: dict[str, int], colors: dict[str, tuple[int, int, int]] | None = None) -> None:
    """Draw a simple legend on the image (top-right)."""
    import cv2

    if not class_names:
        return

    pad = 10
    line_h = 22
    box_w = 220
    box_h = pad * 2 + line_h * len(class_names)

    h, w = out.shape[:2]
    x2 = w - 10
    y1 = 10
    x1 = max(10, x2 - box_w)
    y2 = min(h - 10, y1 + box_h)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    out[:] = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), (203, 213, 225), 1)

    y = y1 + pad + 16
    for name in class_names:
        # Use the same colors as bounding boxes when available; otherwise derive a stable color from the name.
        if colors and name in colors:
            color = colors[name]
        else:
            import hashlib
            h = int(hashlib.md5(name.encode('utf-8')).hexdigest()[:8], 16)
            color = _color_for_class(h % 1000)
        cv2.rectangle(out, (x1 + pad, y - 12), (x1 + pad + 14, y + 2), color, -1)
        txt = f"{name}: {counts.get(name, 0)}"
        cv2.putText(out, txt, (x1 + pad + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2)
        y += line_h


def draw_detections_bgr(image_bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the image (BGR)."""
    import cv2  # lazy

    out = image_bgr.copy()
    class_counts: dict[str, int] = {}
    class_colors: dict[str, tuple[int, int, int]] = {}
    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        color = _color_for_class(det.class_id)
        class_colors.setdefault(det.class_name, color)
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - baseline - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Legend (top-right)
    if class_counts:
        # sort by count desc
        names = sorted(class_counts.keys(), key=lambda k: (-class_counts[k], k))
        _draw_legend(out, names[:12], class_counts, class_colors)
    return out


def save_image_bgr(path: Path, image_bgr: np.ndarray) -> None:
    import cv2  # lazy

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise ValueError(f"Failed to write image: {path}")
