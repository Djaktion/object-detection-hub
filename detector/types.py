from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # Bounding box in absolute pixel coordinates
    x1: float
    y1: float
    x2: float
    y2: float


MediaType = Literal["image", "video"]
