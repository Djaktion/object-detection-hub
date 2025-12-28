from __future__ import annotations

from collections import Counter
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models import Analysis, Detection, File
from detector.types import Detection as Det


def create_file(db: Session, *, filename: str, content_type: str, stored_path: Path) -> File:
    obj = File(filename=filename, content_type=content_type, stored_path=str(stored_path))
    db.add(obj)
    db.flush()
    return obj


def create_analysis(
    db: Session,
    *,
    analysis_id: str,
    file_id: int,
    model_name: str,
    conf_threshold: float,
    duration_ms: int,
) -> Analysis:
    obj = Analysis(
        analysis_id=analysis_id,
        file_id=file_id,
        model_name=model_name,
        conf_threshold=conf_threshold,
        duration_ms=duration_ms,
    )
    db.add(obj)
    db.flush()
    return obj


def bulk_create_detections(db: Session, *, analysis_pk: int, detections: list[Det]) -> None:
    rows = [
        Detection(
            analysis_fk=analysis_pk,
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=float(d.confidence),
            # detector.types.Detection exposes absolute pixel coords as x1..y2
            x1=int(d.x1),
            y1=int(d.y1),
            x2=int(d.x2),
            y2=int(d.y2),
        )
        for d in detections
    ]
    db.add_all(rows)


def list_analyses(db: Session, *, limit: int = 50, offset: int = 0) -> list[tuple[Analysis, File, int]]:
    # returns (analysis, file, detections_count)
    sub = (
        select(Detection.analysis_fk, func.count(Detection.id).label("cnt"))
        .group_by(Detection.analysis_fk)
        .subquery()
    )

    q = (
        select(Analysis, File, func.coalesce(sub.c.cnt, 0))
        .join(File, File.id == Analysis.file_id)
        .outerjoin(sub, sub.c.analysis_fk == Analysis.id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [(a, f, int(cnt)) for a, f, cnt in db.execute(q).all()]


def get_analysis_by_analysis_id(db: Session, *, analysis_id: str) -> Analysis | None:
    return db.scalar(select(Analysis).where(Analysis.analysis_id == analysis_id))


def get_class_counts_for_analysis(db: Session, *, analysis_pk: int) -> dict[str, int]:
    q = select(Detection.class_name, func.count(Detection.id)).where(Detection.analysis_fk == analysis_pk).group_by(Detection.class_name)
    return {name: int(cnt) for name, cnt in db.execute(q).all()}


def get_global_class_counts(db: Session, *, limit: int = 20) -> list[tuple[str, int]]:
    q = (
        select(Detection.class_name, func.count(Detection.id).label("cnt"))
        .group_by(Detection.class_name)
        .order_by(func.count(Detection.id).desc())
        .limit(limit)
    )
    return [(name, int(cnt)) for name, cnt in db.execute(q).all()]


def get_timeseries_for_class(db: Session, *, class_name: str) -> list[tuple[str, int]]:
    # group by date (YYYY-MM-DD)
    q = (
        select(func.strftime('%Y-%m-%d', Analysis.created_at).label('d'), func.count(Detection.id))
        .join(Detection, Detection.analysis_fk == Analysis.id)
        .where(Detection.class_name == class_name)
        .group_by('d')
        .order_by('d')
    )
    return [(d, int(cnt)) for d, cnt in db.execute(q).all()]
