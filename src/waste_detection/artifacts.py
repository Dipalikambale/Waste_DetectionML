from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_model_artifacts(
    best_weights: Path,
    output_dir: Path,
    class_names: list[str],
    metrics: dict[str, float],
    run_dir: Path,
) -> tuple[Path, Path]:
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce best weights: {best_weights}")
    output_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = output_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with best_weights.open("rb") as weights_file:
        for chunk in iter(lambda: weights_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_path = versions_dir / f"waste-yolov5-{version}-{digest[:8]}.pt"
    shutil.copy2(best_weights, version_path)
    active_path = output_dir / "best.pt"
    temporary_model = output_dir / ".best.pt.tmp"
    shutil.copy2(version_path, temporary_model)
    os.replace(temporary_model, active_path)
    metadata: dict[str, Any] = {
        "model_family": "YOLOv5",
        "version": version_path.stem,
        "weights": active_path.name,
        "sha256": digest,
        "class_names": class_names,
        "metrics": metrics,
        "run_dir": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / "metadata.json"
    temporary_metadata = output_dir / ".metadata.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_metadata, metadata_path)
    return active_path, metadata_path
