from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def generate_pdf_report(
    *,
    out_path: Path,
    meta: Dict[str, Any],
    detections: List[Dict[str, Any]],
    preview_image_path: Path | None = None,
    extra_lines: List[str] | None = None,
) -> None:
    """Generate a simple PDF report for an analysis.

    Uses reportlab (already available in the runtime).
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, h - 20 * mm, "Object Detection Hub — Raport")

    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, h - 26 * mm, f"Analysis ID: {meta.get('analysis_id', '')}")
    c.drawString(20 * mm, h - 31 * mm, f"Typ: {meta.get('type', 'image')}")
    c.drawString(20 * mm, h - 36 * mm, f"Model: {meta.get('model', '')} | conf: {meta.get('conf_threshold', '')} | czas: {meta.get('duration_ms', '')} ms")

    y = h - 46 * mm

    # Preview image
    if preview_image_path and preview_image_path.exists():
        try:
            img = ImageReader(str(preview_image_path))
            # Fit into a box
            box_w = 170 * mm
            box_h = 95 * mm
            c.drawImage(img, 20 * mm, y - box_h, width=box_w, height=box_h, preserveAspectRatio=True, mask='auto')
            y -= box_h + 8 * mm
        except Exception:
            pass

    # Stats
    counts: Dict[str, int] = {}
    for d in detections:
        name = str(d.get("class_name", ""))
        if name:
            counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Podsumowanie")
    y -= 6 * mm

    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Liczba detekcji: {len(detections)}")
    y -= 6 * mm

    if extra_lines:
        for line in extra_lines[:6]:
            c.drawString(20 * mm, y, line)
            y -= 5 * mm

    y -= 2 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y, "Wykryte klasy (Top)")
    y -= 6 * mm

    c.setFont("Helvetica", 10)
    for name, cnt in top[:20]:
        c.drawString(24 * mm, y, f"• {name}: {cnt}")
        y -= 5 * mm
        if y < 20 * mm:
            c.showPage()
            y = h - 20 * mm
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
