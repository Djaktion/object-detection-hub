from __future__ import annotations

import argparse
import json
from pathlib import Path

from .visualize import draw_detections_bgr, save_image_bgr
from .yolo_detector import YOLODetector, load_image_bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Object Detection Hub - CLI")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model name/path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    image_path = Path(args.image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(args.model)
    img = load_image_bgr(image_path)
    dets = detector.predict_image(img, conf=args.conf)

    # Save JSON
    json_path = outdir / f"{image_path.stem}_detections.json"
    json_path.write_text(json.dumps(detector.to_jsonable(dets), indent=2), encoding="utf-8")

    # Save preview
    preview = draw_detections_bgr(img, dets)
    preview_path = outdir / f"{image_path.stem}_preview.jpg"
    save_image_bgr(preview_path, preview)

    print(f"Saved: {json_path}")
    print(f"Saved: {preview_path}")
    print(f"Detections: {len(dets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
