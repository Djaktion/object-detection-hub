from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .types import Detection


class YOLODetector:
    """Ultralytics YOLO detector wrapper.

    This class lazily imports ultralytics to keep import time low.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Ultralytics is not installed or failed to import. "
                "Install requirements.txt and ensure your environment supports it."
            ) from e
        self._model = YOLO(self.model_name)

    def predict_image(self, image_bgr: np.ndarray, conf: float = 0.25) -> List[Detection]:
        """Run inference on a BGR uint8 image."""
        self._load()
        assert self._model is not None

        # Ultralytics accepts numpy arrays (BGR ok); returns Results list
        results = self._model.predict(
            source=image_bgr,
            conf=conf,
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        names = getattr(self._model, "names", None) or getattr(r0, "names", None) or {}

        detections: List[Detection] = []
        for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
            name = names.get(int(cid), str(int(cid)))
            detections.append(
                Detection(
                    class_id=int(cid),
                    class_name=str(name),
                    confidence=float(c),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return detections

    @staticmethod
    def to_jsonable(detections: Iterable[Detection]) -> list[dict]:
        return [asdict(d) for d in detections]


def load_image_bgr(path: Path) -> np.ndarray:
    import cv2  # lazy

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img
