from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .types import Detection
from .visualize import draw_detections_bgr
from .yolo_detector import YOLODetector


def process_video(
    detector: YOLODetector,
    input_path: Path,
    output_path: Path,
    *,
    conf: float = 0.25,
    frame_step: int = 5,
    max_frames: int = 0,
) -> Tuple[Dict, List[Detection], List[Dict]]:
    import cv2

    frame_step = max(1, int(frame_step))
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to read frames from video: {input_path}")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_path = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Failed to open video writer: {output_path}")

    per_frame: List[Dict] = []
    flat: List[Detection] = []
    frame_idx = 0
    sampled = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % frame_step == 0:
            dets = detector.predict_image(frame, conf=conf)
            flat.extend(dets)
            sampled += 1
            per_frame.append(
                {
                    "frame_index": frame_idx,
                    "timestamp_s": float(frame_idx / fps) if fps else 0.0,
                    "detections": [asdict(d) for d in dets],
                }
            )
            frame = draw_detections_bgr(frame, dets)

        writer.write(frame)
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()


    # Transcode to H.264 for best browser compatibility (HTML5 <video>).
    # Uses a bundled ffmpeg binary via imageio-ffmpeg (no system ffmpeg required).
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        import subprocess

        ffmpeg = get_ffmpeg_exe()
        # overwrite output_path
        cmd = [
            ffmpeg, "-y",
            "-i", str(raw_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # Fallback: keep the raw mp4v file if transcoding fails
        try:
            if raw_path.exists() and not output_path.exists():
                raw_path.replace(output_path)
        except Exception:
            pass
    finally:
        # Cleanup raw file if separate
        try:
            if raw_path.exists() and raw_path != output_path:
                raw_path.unlink()
        except Exception:
            pass

    meta = {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames_read": frame_idx,
        "frame_step": frame_step,
        "sampled_frames": sampled,
    }
    return meta, flat, per_frame
