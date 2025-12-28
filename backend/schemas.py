from __future__ import annotations

from typing import List

from pydantic import BaseModel, AnyUrl


class UploadResponse(BaseModel):
    analysis_id: str
    analysis_url: AnyUrl
    preview_url: AnyUrl
    export_csv_url: AnyUrl
    analysis_page_url: AnyUrl
    report_pdf_url: AnyUrl | None = None
    output_video_url: AnyUrl | None = None


class AnalysisMeta(BaseModel):
    analysis_id: str
    type: str = "image"
    input_file: str
    preview_file: str
    output_video_file: str | None = None
    model: str
    conf_threshold: float
    duration_ms: int
    created_at: int
    num_detections: int
    video: dict | None = None

    class Config:
        extra = "allow"


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class AnalysisResponse(BaseModel):
    meta: AnalysisMeta
    detections: List[DetectionOut]


class AnalysisListItem(BaseModel):
    analysis_id: str
    filename: str
    created_at: int
    model: str
    conf_threshold: float
    num_detections: int


class AnalysisListResponse(BaseModel):
    items: List[AnalysisListItem]
    limit: int
    offset: int


class ClassCount(BaseModel):
    class_name: str
    count: int


class ClassCountsResponse(BaseModel):
    items: List[ClassCount]


class TimePoint(BaseModel):
    date: str
    count: int


class TimeSeriesResponse(BaseModel):
    class_name: str
    points: List[TimePoint]
