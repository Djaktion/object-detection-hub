from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from detector.types import Detection


def new_analysis_id() -> str:
    return uuid.uuid4().hex


def analysis_dir(results_dir: Path, analysis_id: str) -> Path:
    return results_dir / analysis_id


def save_upload(uploads_dir: Path, filename: str, content: bytes) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
    out_path = uploads_dir / safe_name
    out_path.write_bytes(content)
    return out_path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_analysis_bundle(
    results_dir: Path,
    analysis_id: str,
    input_path: Path,
    preview_path: Path,
    detections: List[Detection],
    model_name: str,
    conf_threshold: float,
    duration_ms: int,
) -> None:
    adir = analysis_dir(results_dir, analysis_id)
    adir.mkdir(parents=True, exist_ok=True)

    # Metadata
    meta = {
        "analysis_id": analysis_id,
        "type": "image",
        "input_file": str(input_path.name),
        "preview_file": str(preview_path.name),
        "model": model_name,
        "conf_threshold": conf_threshold,
        "duration_ms": duration_ms,
        "created_at": int(time.time()),
        "num_detections": len(detections),
    }

    # Save detections + meta
    write_json(adir / "detections.json", [asdict(d) for d in detections])
    write_json(adir / "meta.json", meta)


def save_analysis_bundle_video(
    results_dir: Path,
    analysis_id: str,
    input_path: Path,
    preview_path: Path,
    output_video_path: Path,
    detections_flat: List[Detection],
    detections_per_frame: List[Dict[str, Any]],
    model_name: str,
    conf_threshold: float,
    duration_ms: int,
    video_meta: Dict[str, Any],
) -> None:
    adir = analysis_dir(results_dir, analysis_id)
    adir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "type": "video",
        "input_file": str(input_path.name),
        "preview_file": str(preview_path.name),
        "output_video_file": str(output_video_path.name),
        "model": model_name,
        "conf_threshold": conf_threshold,
        "duration_ms": duration_ms,
        "created_at": int(time.time()),
        "num_detections": len(detections_flat),
    }
    meta.update(video_meta)

    write_json(adir / "detections.json", [asdict(d) for d in detections_flat])
    write_json(adir / "detections_per_frame.json", detections_per_frame)
    write_json(adir / "meta.json", meta)


def load_analysis_meta(results_dir: Path, analysis_id: str) -> Dict[str, Any]:
    return load_json(analysis_dir(results_dir, analysis_id) / "meta.json")


def load_analysis_detections(results_dir: Path, analysis_id: str) -> List[Dict[str, Any]]:
    return load_json(analysis_dir(results_dir, analysis_id) / "detections.json")


def load_analysis_detections_per_frame(results_dir: Path, analysis_id: str) -> List[Dict[str, Any]]:
    return load_json(analysis_dir(results_dir, analysis_id) / "detections_per_frame.json")
