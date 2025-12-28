# Architektura (skrót)

```mermaid
graph TD
  UI[Frontend (Jinja2 + Bootstrap)] -->|HTTP| API[FastAPI]
  API -->|Upload| FS[uploads/]
  API -->|Inference| YOLO[Ultralytics YOLO]
  YOLO -->|Detections| API
  API -->|Save bundle| RES[results/{analysis_id}/]
  API -->|Persist| DB[(SQLite: data/odh.db)]
  UI -->|Fetch stats JSON| API
```

## Dane
- `results/{analysis_id}/meta.json` – metadane
- `results/{analysis_id}/detections.json` – detekcje (bbox + klasa)
- `results/{analysis_id}/preview.jpg` – podgląd z bboxami

## Endpointy
Opis w Swagger: `/docs`.
