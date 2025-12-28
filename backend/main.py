from __future__ import annotations

import csv
import io
import time
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html
from sqlalchemy.orm import Session

from backend.config import settings
from backend.crud import (
    bulk_create_detections,
    create_analysis,
    create_file,
    get_analysis_by_analysis_id,
    get_class_counts_for_analysis,
    get_global_class_counts,
    get_timeseries_for_class,
    list_analyses,
)
from backend.db import get_db
from backend.init_db import init_db
from backend.schemas import (
    AnalysisListItem,
    AnalysisListResponse,
    AnalysisResponse,
    ClassCount,
    ClassCountsResponse,
    TimePoint,
    TimeSeriesResponse,
    UploadResponse,
)
from backend.storage import (
    analysis_dir,
    load_analysis_detections,
    load_analysis_meta,
    load_analysis_detections_per_frame,
    new_analysis_id,
    save_analysis_bundle,
    save_analysis_bundle_video,
    save_upload,
)
from backend.reporting import generate_pdf_report
from detector.visualize import draw_detections_bgr, save_image_bgr
from detector.video import process_video
from detector.yolo_detector import YOLODetector, load_image_bgr

app = FastAPI(title="Object Detection Hub", version="0.2.0", docs_url=None, redoc_url=None)

# Frontend
templates = Jinja2Templates(directory=str(settings.project_root / "frontend" / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(settings.project_root / "frontend" / "static")),
    name="static",
)

detector = YOLODetector(settings.yolo_model)

@app.get("/swagger", include_in_schema=False)
def swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Object Detection Hub - Swagger UI",
    )


@app.get("/docs", include_in_schema=False)
def docs_with_uploader() -> HTMLResponse:
    # Custom landing page: simple upload form + embedded Swagger UI below.
    html = """<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Object Detection Hub - Docs</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #f6f7fb; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
    .card { background: #fff; border-radius: 14px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); padding: 16px; }
    h1 { margin: 0 0 6px; font-size: 22px; }
    .muted { color: #64748b; font-size: 14px; margin: 0 0 14px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
    label { display:block; font-size: 13px; color: #334155; margin-bottom: 6px; }
    input[type="file"], input[type="number"] { width: 100%; padding: 10px; border: 1px solid #cbd5e1; border-radius: 10px; background: #fff; }
    .col { flex: 1 1 240px; }
    button { padding: 11px 14px; border: 0; border-radius: 10px; background: #2563eb; color: #fff; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: .6; cursor: default; }
    .links a { display: inline-block; margin-right: 10px; margin-top: 8px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }
    @media (min-width: 900px) { .grid { grid-template-columns: 1.2fr .8fr; } }
    img { max-width: 100%; border-radius: 12px; border: 1px solid #e2e8f0; background:#fff; }
    pre { background: #0b1020; color:#e2e8f0; padding: 12px; border-radius: 12px; overflow:auto; font-size: 12px; }
    .divider { height: 1px; background: #e2e8f0; margin: 18px 0; }
    iframe { width: 100%; height: 78vh; border: 0; border-top: 1px solid #e2e8f0; background: #fff; border-radius: 14px; }
    .topbar { display:flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
    .topbar a { color: #2563eb; text-decoration: none; font-weight: 600; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Object Detection Hub</h1>
        <p class="muted">Szybki formularz dla użytkownika + pełna dokumentacja API poniżej.</p>
      </div>
      <div>
        <a href="/" target="_blank" rel="noopener">Otwórz aplikację (Frontend) ↗</a>
      </div>
    </div>

    <div class="card">
      <div class="row">
        <div class="col">
          <label>Zdjęcie (JPG/PNG)</label>
          <input id="file" type="file" accept="image/*" />
        </div>
        <div class="col" style="max-width:220px">
          <label>Confidence (conf)</label>
          <input id="conf" type="number" min="0" max="1" step="0.05" value="0.25" />
        </div>
        <div style="padding-bottom:2px">
          <button id="runBtn" onclick="run()">Wykryj obiekty</button>
        </div>
      </div>

      <div class="grid" id="out" style="display:none;">
        <div>
          <div class="muted" style="margin:10px 0 6px;">Podgląd (bbox)</div>
          <img id="previewImg" alt="preview" />
          <div class="links" id="links"></div>
        </div>
        <div>
          <div class="muted" style="margin:10px 0 6px;">Odpowiedź JSON</div>
          <pre id="json"></pre>
        </div>
      </div>

      <div id="err" class="muted" style="color:#b91c1c; margin-top:10px; display:none;"></div>
    </div>

    <div class="divider"></div>

    <iframe src="/swagger" title="Swagger UI"></iframe>
  </div>

<script>
async function run() {
  const fileInput = document.getElementById('file');
  const conf = document.getElementById('conf').value || '0.25';
  const btn = document.getElementById('runBtn');
  const err = document.getElementById('err');
  err.style.display = 'none';
  err.textContent = '';

  if (!fileInput.files || fileInput.files.length === 0) {
    err.textContent = 'Wybierz plik ze zdjęciem.';
    err.style.display = 'block';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Analizuję...';

  try {
    const fd = new FormData();
    fd.append('file', fileInput.files[0]);

    const res = await fetch(`/api/upload/image?conf=${encodeURIComponent(conf)}`, {
      method: 'POST',
      body: fd
    });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(`HTTP ${res.status}: ${t}`);
    }

    const data = await res.json();

    // Show output
    document.getElementById('out').style.display = 'grid';
    document.getElementById('json').textContent = JSON.stringify(data, null, 2);

    // Preview image
    if (data.preview_url) {
      document.getElementById('previewImg').src = data.preview_url;
    }

    // Links (clickable, new tab)
    const links = [];
    if (data.analysis_url) links.push(['JSON', data.analysis_url]);
    if (data.preview_url) links.push(['Preview', data.preview_url]);
    if (data.export_csv_url) links.push(['CSV', data.export_csv_url]);
    if (data.analysis_page_url) links.push(['Strona wyniku', data.analysis_page_url]);

    const linksDiv = document.getElementById('links');
    linksDiv.innerHTML = links.map(([label, url]) =>
      `<a href="${url}" target="_blank" rel="noopener">${label} ↗</a>`
    ).join('');
  } catch (e) {
    err.textContent = String(e);
    err.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Wykryj obiekty';
  }
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health", tags=["API"], summary="GET /health")
def health():
    return {"status": "ok"}


def _ensure_image(upload: UploadFile) -> None:
    if upload.content_type is None or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported in this endpoint.")


def _ensure_video(upload: UploadFile) -> None:
    # Some browsers send video/mp4, others application/octet-stream; allow mp4 by extension as fallback.
    ct = upload.content_type or ""
    if ct.startswith("video/"):
        return
    name = (upload.filename or "").lower()
    if name.endswith(".mp4") or name.endswith(".mov") or name.endswith(".avi"):
        return
    raise HTTPException(status_code=400, detail="Only video uploads are supported in this endpoint (mp4 recommended).")




def _abs_url(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip('/')
    if not path.startswith('/'):
        path = '/' + path
    return base + path
# -----------------
# API
# -----------------

@app.post("/api/upload/image", response_model=UploadResponse, tags=["API"], summary="POST /api/upload/image")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(default=settings.default_conf, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    _ensure_image(file)

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.max_upload_mb}MB).")

    input_path = save_upload(settings.uploads_dir, file.filename or "upload", content)

    analysis_id = new_analysis_id()
    adir = analysis_dir(settings.results_dir, analysis_id)
    adir.mkdir(parents=True, exist_ok=True)

    # Inference
    t0 = time.perf_counter()
    img = load_image_bgr(input_path)
    detections = detector.predict_image(img, conf=conf)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Preview image
    preview_img = draw_detections_bgr(img, detections)
    preview_path = adir / "preview.jpg"
    save_image_bgr(preview_path, preview_img)

    # Save meta + detections (filesystem)
    save_analysis_bundle(
        results_dir=settings.results_dir,
        analysis_id=analysis_id,
        input_path=input_path,
        preview_path=preview_path,
        detections=detections,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
    )

    # Persist to DB
    fobj = create_file(db, filename=file.filename or input_path.name, content_type=file.content_type or "image", stored_path=input_path)
    aobj = create_analysis(
        db,
        analysis_id=analysis_id,
        file_id=fobj.id,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
    )
    bulk_create_detections(db, analysis_pk=aobj.id, detections=detections)
    db.commit()

    # PDF report
    try:
        meta = load_analysis_meta(settings.results_dir, analysis_id)
        dets = load_analysis_detections(settings.results_dir, analysis_id)
        report_path = analysis_dir(settings.results_dir, analysis_id) / "report.pdf"
        generate_pdf_report(
            out_path=report_path,
            meta=meta,
            detections=dets,
            preview_image_path=analysis_dir(settings.results_dir, analysis_id) / "preview.jpg",
            extra_lines=None,
        )
    except Exception:
        pass

    return UploadResponse(
        analysis_id=analysis_id,
        analysis_url=_abs_url(request, f"/api/analysis/{analysis_id}"),
        preview_url=_abs_url(request, f"/api/analysis/{analysis_id}/preview"),
        export_csv_url=_abs_url(request, f"/api/analysis/{analysis_id}/export.csv"),
        analysis_page_url=_abs_url(request, f"/analysis/{analysis_id}"),
        report_pdf_url=_abs_url(request, f"/api/analysis/{analysis_id}/report.pdf"),
    )


@app.post("/api/upload/video", response_model=UploadResponse, tags=["API"], summary="POST /api/upload/video")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(default=settings.default_conf, ge=0.0, le=1.0),
    frame_step: int = Query(5, ge=1, le=60, description="Process every Nth frame"),
    max_frames: int = Query(0, ge=0, le=20000, description="Optional safety limit"),
    db: Session = Depends(get_db),
):
    _ensure_video(file)
    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.max_upload_mb}MB).")

    input_path = save_upload(settings.uploads_dir, file.filename or "upload.mp4", content)
    analysis_id = new_analysis_id()
    adir = analysis_dir(settings.results_dir, analysis_id)
    adir.mkdir(parents=True, exist_ok=True)

    output_video_path = adir / "output.mp4"

    t0 = time.perf_counter()
    video_meta, flat_detections, per_frame = process_video(
        detector,
        input_path,
        output_video_path,
        conf=conf,
        frame_step=frame_step,
        max_frames=max_frames,
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Preview frame (first sampled frame)
    preview_path = adir / "preview.jpg"
    try:
        import cv2

        cap = cv2.VideoCapture(str(output_video_path))
        ok, frame0 = cap.read()
        cap.release()
        if ok and frame0 is not None:
            cv2.imwrite(str(preview_path), frame0)
    except Exception:
        pass

    save_analysis_bundle_video(
        results_dir=settings.results_dir,
        analysis_id=analysis_id,
        input_path=input_path,
        preview_path=preview_path,
        output_video_path=output_video_path,
        detections_flat=flat_detections,
        detections_per_frame=per_frame,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
        video_meta={"video": video_meta},
    )

    fobj = create_file(db, filename=file.filename or input_path.name, content_type=file.content_type or "video", stored_path=input_path)
    aobj = create_analysis(
        db,
        analysis_id=analysis_id,
        file_id=fobj.id,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
    )
    # Store all detections across sampled frames (no frame index in DB)
    bulk_create_detections(db, analysis_pk=aobj.id, detections=flat_detections)
    db.commit()

    # PDF report
    try:
        meta = load_analysis_meta(settings.results_dir, analysis_id)
        dets = load_analysis_detections(settings.results_dir, analysis_id)
        report_path = adir / "report.pdf"
        extra = [
            f"FPS: {video_meta.get('fps')} | frame_step: {video_meta.get('frame_step')} | sampled_frames: {video_meta.get('sampled_frames')}",
            f"Rozdzielczość: {video_meta.get('width')}x{video_meta.get('height')} | frames_read: {video_meta.get('frames_read')}",
        ]
        generate_pdf_report(
            out_path=report_path,
            meta=meta,
            detections=dets,
            preview_image_path=preview_path if preview_path.exists() else None,
            extra_lines=extra,
        )
    except Exception:
        pass

    return UploadResponse(
        analysis_id=analysis_id,
        analysis_url=_abs_url(request, f"/api/analysis/{analysis_id}"),
        preview_url=_abs_url(request, f"/api/analysis/{analysis_id}/preview"),
        export_csv_url=_abs_url(request, f"/api/analysis/{analysis_id}/export.csv"),
        analysis_page_url=_abs_url(request, f"/analysis/{analysis_id}"),
        report_pdf_url=_abs_url(request, f"/api/analysis/{analysis_id}/report.pdf"),
        output_video_url=_abs_url(request, f"/api/analysis/{analysis_id}/video"),
    )


@app.get("/api/analysis/{analysis_id}", response_model=AnalysisResponse, tags=["API"], summary="GET /api/analysis/{analysis_id}")
def get_analysis_api(analysis_id: str):
    try:
        meta = load_analysis_meta(settings.results_dir, analysis_id)
        dets = load_analysis_detections(settings.results_dir, analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {"meta": meta, "detections": dets}


@app.get("/api/analysis/{analysis_id}/preview", tags=["API"], summary="GET /api/analysis/{analysis_id}/preview")
def get_preview(analysis_id: str):
    preview_path = analysis_dir(settings.results_dir, analysis_id) / "preview.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(preview_path)


@app.get("/api/analysis/{analysis_id}/video", tags=["API"], summary="GET /api/analysis/{analysis_id}/video")
def get_output_video(analysis_id: str):
    meta_path = analysis_dir(settings.results_dir, analysis_id) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    meta = load_analysis_meta(settings.results_dir, analysis_id)
    if meta.get("type") != "video":
        raise HTTPException(status_code=400, detail="This analysis is not a video analysis")
    vfile = meta.get("output_video_file") or "output.mp4"
    vpath = analysis_dir(settings.results_dir, analysis_id) / vfile
    if not vpath.exists():
        raise HTTPException(status_code=404, detail="Output video not found")
    return FileResponse(vpath, media_type="video/mp4")


@app.get("/api/analysis/{analysis_id}/report.pdf", tags=["API"], summary="GET /api/analysis/{analysis_id}/report.pdf")
def get_report_pdf(analysis_id: str):
    rpath = analysis_dir(settings.results_dir, analysis_id) / "report.pdf"
    if not rpath.exists():
        # Generate on-demand if missing
        try:
            meta = load_analysis_meta(settings.results_dir, analysis_id)
            dets = load_analysis_detections(settings.results_dir, analysis_id)
            preview = analysis_dir(settings.results_dir, analysis_id) / "preview.jpg"
            extra = None
            if meta.get("type") == "video":
                v = (meta.get("video") or {})
                extra = [
                    f"FPS: {v.get('fps')} | frame_step: {v.get('frame_step')} | sampled_frames: {v.get('sampled_frames')}",
                    f"Rozdzielczość: {v.get('width')}x{v.get('height')} | frames_read: {v.get('frames_read')}",
                ]
            generate_pdf_report(out_path=rpath, meta=meta, detections=dets, preview_image_path=preview if preview.exists() else None, extra_lines=extra)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Analysis not found")
    return FileResponse(rpath, media_type="application/pdf")


@app.get("/api/analysis/{analysis_id}/export.csv", tags=["API"], summary="GET /api/analysis/{analysis_id}/export.csv")
def export_csv(analysis_id: str):
    try:
        meta = load_analysis_meta(settings.results_dir, analysis_id)
        dets = load_analysis_detections(settings.results_dir, analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")

    buf = io.StringIO()
    w = csv.writer(buf)
    # `load_analysis_meta` returns a dict (saved from JSON)
    w.writerow(["analysis_id", meta.get("analysis_id", analysis_id)])
    w.writerow(["model", meta.get("model", "")])
    w.writerow(["conf_threshold", meta.get("conf_threshold", "")])
    w.writerow(["duration_ms", meta.get("duration_ms", "")])
    w.writerow([])
    w.writerow(["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    # `load_analysis_detections` returns a list of dicts (saved from JSON)
    for d in dets:
        w.writerow([
            d.get("class_id"),
            d.get("class_name"),
            d.get("confidence"),
            d.get("x1"),
            d.get("y1"),
            d.get("x2"),
            d.get("y2"),
        ])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={analysis_id}_detections.csv"},
    )


@app.get("/api/analyses", response_model=AnalysisListResponse, tags=["API"], summary="GET /api/analyses")
def list_analyses_api(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    rows = list_analyses(db, limit=limit, offset=offset)
    items = [
        AnalysisListItem(
            analysis_id=a.analysis_id,
            filename=f.filename,
            created_at=int(a.created_at.timestamp()),
            model=a.model_name,
            conf_threshold=a.conf_threshold,
            num_detections=cnt,
        )
        for a, f, cnt in rows
    ]
    return AnalysisListResponse(items=items, limit=limit, offset=offset)


@app.get("/api/stats/classes", response_model=ClassCountsResponse, tags=["API"], summary="GET /api/stats/classes")
def stats_classes_global(db: Session = Depends(get_db)):
    items = [ClassCount(class_name=name, count=cnt) for name, cnt in get_global_class_counts(db, limit=50)]
    return ClassCountsResponse(items=items)


@app.get("/api/stats/analysis/{analysis_id}/classes", response_model=ClassCountsResponse, tags=["API"], summary="GET /api/stats/analysis/{analysis_id}/classes")
def stats_classes_for_analysis(analysis_id: str, db: Session = Depends(get_db)):
    a = get_analysis_by_analysis_id(db, analysis_id=analysis_id)
    if a is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    counts = get_class_counts_for_analysis(db, analysis_pk=a.id)
    items = [ClassCount(class_name=k, count=v) for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    return ClassCountsResponse(items=items)


@app.get("/api/stats/timeseries", response_model=TimeSeriesResponse, tags=["API"], summary="GET /api/stats/timeseries")
def stats_timeseries(
    class_name: str = Query(..., description="Exact class name, e.g. 'car'"),
    db: Session = Depends(get_db),
):
    points = [TimePoint(date=d, count=cnt) for d, cnt in get_timeseries_for_class(db, class_name=class_name)]
    return TimeSeriesResponse(class_name=class_name, points=points)


# -----------------
# Frontend pages
# -----------------

@app.get("/", response_class=HTMLResponse, tags=["Frontend"], summary="GET /")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_conf": settings.default_conf,
            "max_upload_mb": settings.max_upload_mb,
        },
    )


@app.post("/upload", tags=["Frontend"], summary="POST /upload")
async def upload_page(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    conf = float(form.get("conf", settings.default_conf))
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    # Re-use the API logic by mimicking the call
    # (FastAPI gives us an UploadFile here as well)
    upload: UploadFile = file
    _ensure_image(upload)
    content = await upload.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.max_upload_mb}MB).")

    input_path = save_upload(settings.uploads_dir, upload.filename or "upload", content)

    analysis_id = new_analysis_id()
    adir = analysis_dir(settings.results_dir, analysis_id)
    adir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    img = load_image_bgr(input_path)
    detections = detector.predict_image(img, conf=conf)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    preview_img = draw_detections_bgr(img, detections)
    preview_path = adir / "preview.jpg"
    save_image_bgr(preview_path, preview_img)

    save_analysis_bundle(
        results_dir=settings.results_dir,
        analysis_id=analysis_id,
        input_path=input_path,
        preview_path=preview_path,
        detections=detections,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
    )

    fobj = create_file(db, filename=upload.filename or input_path.name, content_type=upload.content_type or "image", stored_path=input_path)
    aobj = create_analysis(
        db,
        analysis_id=analysis_id,
        file_id=fobj.id,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
    )
    bulk_create_detections(db, analysis_pk=aobj.id, detections=detections)
    db.commit()

    return RedirectResponse(url=f"/analysis/{analysis_id}", status_code=303)


@app.post("/upload/video", tags=["Frontend"], summary="POST /upload/video")
async def upload_video_page(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    conf = float(form.get("conf", settings.default_conf))
    frame_step = int(form.get("frame_step", 5))
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    upload: UploadFile = file
    _ensure_video(upload)
    content = await upload.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (>{settings.max_upload_mb}MB).")

    input_path = save_upload(settings.uploads_dir, upload.filename or "upload.mp4", content)
    analysis_id = new_analysis_id()
    adir = analysis_dir(settings.results_dir, analysis_id)
    adir.mkdir(parents=True, exist_ok=True)

    output_video_path = adir / "output.mp4"
    t0 = time.perf_counter()
    video_meta, flat_detections, per_frame = process_video(
        detector,
        input_path,
        output_video_path,
        conf=conf,
        frame_step=frame_step,
        max_frames=0,
    )
    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Preview from first frame of output
    preview_path = adir / "preview.jpg"
    try:
        import cv2

        cap = cv2.VideoCapture(str(output_video_path))
        ok, frame0 = cap.read()
        cap.release()
        if ok and frame0 is not None:
            cv2.imwrite(str(preview_path), frame0)
    except Exception:
        pass

    save_analysis_bundle_video(
        results_dir=settings.results_dir,
        analysis_id=analysis_id,
        input_path=input_path,
        preview_path=preview_path,
        output_video_path=output_video_path,
        detections_flat=flat_detections,
        detections_per_frame=per_frame,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
        video_meta={"video": video_meta},
    )

    fobj = create_file(db, filename=upload.filename or input_path.name, content_type=upload.content_type or "video", stored_path=input_path)
    aobj = create_analysis(
        db,
        analysis_id=analysis_id,
        file_id=fobj.id,
        model_name=settings.yolo_model,
        conf_threshold=conf,
        duration_ms=duration_ms,
    )
    bulk_create_detections(db, analysis_pk=aobj.id, detections=flat_detections)
    db.commit()

    # Create PDF report
    try:
        meta = load_analysis_meta(settings.results_dir, analysis_id)
        dets = load_analysis_detections(settings.results_dir, analysis_id)
        report_path = adir / "report.pdf"
        extra = [
            f"FPS: {video_meta.get('fps')} | frame_step: {video_meta.get('frame_step')} | sampled_frames: {video_meta.get('sampled_frames')}",
        ]
        generate_pdf_report(out_path=report_path, meta=meta, detections=dets, preview_image_path=preview_path if preview_path.exists() else None, extra_lines=extra)
    except Exception:
        pass

    return RedirectResponse(url=f"/analysis/{analysis_id}", status_code=303)


@app.get("/analysis/{analysis_id}", response_class=HTMLResponse, tags=["Frontend"], summary="GET /analysis/{analysis_id}")
def analysis_page(analysis_id: str, request: Request):
    try:
        meta = load_analysis_meta(settings.results_dir, analysis_id)
        dets = load_analysis_detections(settings.results_dir, analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return templates.TemplateResponse(
        "analysis.html",
        {
            "request": request,
            "meta": meta,
            "detections": dets,
        },
    )


@app.get("/history", response_class=HTMLResponse, tags=["Frontend"], summary="GET /history")
def history_page(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    rows = list_analyses(db, limit=limit, offset=offset)
    items = [
        {
            "analysis_id": a.analysis_id,
            "filename": f.filename,
            "created_at": a.created_at,
            "model": a.model_name,
            "conf": a.conf_threshold,
            "count": cnt,
        }
        for a, f, cnt in rows
    ]

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "items": items,
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit,
            "prev_offset": max(0, offset - limit),
        },
    )


@app.get("/dashboard", response_class=HTMLResponse, tags=["Frontend"], summary="GET /dashboard")
def dashboard_page(request: Request):
    # data is loaded via JS from /api/stats/*
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
        },
    )