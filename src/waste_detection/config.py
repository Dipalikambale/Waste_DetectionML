from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str, default: str) -> Path:
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    metadata_path: Path
    confidence: float
    max_upload_bytes: int
    host: str
    port: int
    debug: bool

    @classmethod
    def from_env(cls) -> "Settings":
        confidence = float(os.getenv("WASTE_CONFIDENCE", "0.25"))
        if not 0 <= confidence <= 1:
            raise ValueError("WASTE_CONFIDENCE must be between 0 and 1")
        max_upload_mb = int(os.getenv("WASTE_MAX_UPLOAD_MB", "16"))
        if max_upload_mb < 1:
            raise ValueError("WASTE_MAX_UPLOAD_MB must be at least 1")
        return cls(
            model_path=_project_path(
                os.getenv("WASTE_MODEL_PATH", ""), "artifacts/best.pt"
            ),
            metadata_path=_project_path(
                os.getenv("WASTE_METADATA_PATH", ""), "artifacts/metadata.json"
            ),
            confidence=confidence,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            host=os.getenv("WASTE_HOST", "0.0.0.0"),
            port=int(os.getenv("WASTE_PORT", "5000")),
            debug=os.getenv("WASTE_DEBUG", "").lower() in {"1", "true", "yes"},
        )
