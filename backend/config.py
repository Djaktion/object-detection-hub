from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    You can override any setting via environment variables using ODH_ prefix.
    Example: ODH_DEFAULT_CONF=0.35
    """

    model_config = SettingsConfigDict(env_prefix="ODH_", env_file=".env", extra="ignore")

    # Storage
    project_root: Path = Path(__file__).resolve().parents[1]
    uploads_dir: Path = project_root / "uploads"
    results_dir: Path = project_root / "results"
    data_dir: Path = project_root / "data"

    # Database
    # If db_url is not provided via env, it defaults to a local SQLite file in data/odh.db
    db_url: str | None = None

    # Detection
    yolo_model: str = "yolov8n.pt"  # ultralytics will download weights if missing
    default_conf: float = 0.25

    # Limits
    max_upload_mb: int = 20


settings = Settings()

# Ensure directories exist
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)

# Default DB URL
if settings.db_url is None:
    settings.db_url = f"sqlite:///{(settings.data_dir / 'odh.db').as_posix()}"
