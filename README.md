# Object Detection Hub

## Opis projektu
Object Detection Hub to aplikacja webowa służąca do automatycznej analizy obrazów oraz wideo z wykorzystaniem metod uczenia maszynowego (deep learning). System umożliwia wykrywanie obiektów, ich wizualizację, analizę statystyczną, archiwizację wyników oraz generowanie raportów.

Projekt nie zajmuje się identyfikacją tożsamości ani rozpoznawaniem twarzy – analizowane są wyłącznie klasy obiektów (np. samochody, osoby, znaki drogowe).

---

## Główne funkcjonalności
- Upload i analiza zdjęć (JPG, PNG)
- Upload i analiza wideo (MP4) – detekcja na przestrzeni całego filmu
- Wykrywanie obiektów przy użyciu modelu YOLOv8
- Wizualizacja wyników:
  - bounding boxy
  - nazwy klas
  - wartości confidence
  - spójna kolorystyka klas + legenda
- Historia analiz zapisywana w bazie danych
- Dashboard statystyczny (liczność klas)
- Eksport wyników do CSV
- Automatyczne generowanie raportu PDF
- REST API z dokumentacją Swagger (OpenAPI)

---

## Zastosowane technologie
- Python – główny język aplikacji
- FastAPI – backend oraz REST API
- YOLOv8 (Ultralytics) – model deep learning do detekcji obiektów (zbiór COCO – 80 klas)
- OpenCV – przetwarzanie obrazu i wideo
- SQLite + SQLAlchemy – baza danych historii analiz
- HTML + Bootstrap – frontend aplikacji
- Chart.js – wizualizacja danych statystycznych
- ReportLab – generowanie raportów PDF

---

## Jak działa aplikacja (pipeline)
1. Użytkownik wgrywa obraz lub wideo przez interfejs webowy.
2. Backend zapisuje plik na serwerze.
3. Plik trafia do modułu detekcji (YOLOv8).
4. Model wykrywa obiekty i zwraca:
   - klasę obiektu
   - confidence
   - bounding box
5. Wyniki są filtrowane według progu confidence.
6. Dane zapisywane są w bazie danych oraz plikach wynikowych.
7. Wyniki prezentowane są w UI (obraz/wideo + tabela + statystyki).
8. Opcjonalnie generowany jest raport PDF.

---

## Wyjaśnienie wyników detekcji
### Confidence (`conf`)
- Wartość z zakresu 0–1
- Określa pewność modelu, że dany obiekt należy do wskazanej klasy
- Użytkownik może ustawić minimalny próg confidence

### Bounding Box (`bbox`)
- Prostokąt opisujący lokalizację obiektu na obrazie
- Format: `(x1, y1, x2, y2)` – współrzędne pikselowe
- `(x1, y1)` – lewy górny róg  
- `(x2, y2)` – prawy dolny róg

---

## Analiza wideo
- Wideo analizowane jest na całej długości
- Detekcja wykonywana jest co `N` klatek (`frame_step`)
- Pozwala to zachować kompromis między dokładnością a wydajnością
- Wynikiem jest:
  - wideo MP4 z naniesionymi bounding boxami
  - statystyki zbiorcze

---

## Struktura projektu
object_detection_hub/
│
├── backend/ # Backend FastAPI i API
├── detector/ # Moduł detekcji (YOLO, wideo, bboxy)
├── templates/ # Szablony HTML (frontend)
├── static/ # CSS i zasoby statyczne
├── uploads/ # Pliki wgrane przez użytkownika
├── results/ # Wyniki analiz (obrazy, wideo, JSON, CSV, PDF)
├── data/ # Baza danych SQLite
└── README.md


## Uruchomienie projektu lokalnie

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Uruchomienie

Z katalogu projektu:

```bash
uvicorn backend.main:app --reload
```

- Frontend: http://127.0.0.1:8000/
- Historia: http://127.0.0.1:8000/history
- Dashboard: http://127.0.0.1:8000/dashboard
- Swagger: http://127.0.0.1:8000/docs

## API (najważniejsze)
- POST /api/upload/image?conf=0.25
- GET  /api/analysis/{analysis_id}
- GET  /api/analysis/{analysis_id}/preview
- GET  /api/analysis/{analysis_id}/export.csv
- GET  /api/analyses?limit=50&offset=0
- GET  /api/stats/classes
- GET  /api/stats/analysis/{analysis_id}/classes
- GET  /api/stats/timeseries?class_name=car

## Dane na dysku
- uploads/ – wgrane pliki
- results/{analysis_id}/ – meta.json, detections.json, preview.jpg
- data/odh.db – baza SQLite

## CLI detektor (bez API)

```bash
python -m detector.cli --image path/to/image.jpg --conf 0.25
```