from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from waste_detection.artifacts import save_model_artifacts
from waste_detection.config import PROJECT_ROOT
from waste_detection.validation import ValidationReport, validate_dataset


def _write_training_data_yaml(report: ValidationReport, destination: Path) -> None:
    dataset = yaml.safe_load(report.dataset_yaml.read_text(encoding="utf-8"))
    dataset["path"] = str(report.dataset_yaml.parent)
    for split, image_path in report.split_paths.items():
        dataset[split] = str(image_path)
    destination.write_text(
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def train_model(
    data_yaml: Path,
    weights: str = "yolov5nu.pt",
    epochs: int = 30,
    image_size: int = 416,
    batch_size: int = 2,
    project_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    if epochs < 1 or image_size < 32 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive; image size must be at least 32")
    project_dir = (project_dir or PROJECT_ROOT / "artifacts" / "runs").expanduser().resolve()
    artifact_dir = (artifact_dir or PROJECT_ROOT / "artifacts").expanduser().resolve()
    report: ValidationReport = validate_dataset(data_yaml)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Training requires the dependencies in requirements.txt; install them first"
        ) from error

    model = YOLO(weights)
    with tempfile.TemporaryDirectory(prefix="waste-detection-data-") as temporary_dir:
        training_data_yaml = Path(temporary_dir) / "data.yaml"
        _write_training_data_yaml(report, training_data_yaml)
        results = model.train(
            data=str(training_data_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            workers=0,
            project=str(project_dir),
            name="waste-yolov5",
            exist_ok=False,
        )
    run_dir = Path(results.save_dir)
    metrics: dict[str, float] = {}
    for key, value in getattr(results, "results_dict", {}).items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    best_weights = run_dir / "weights" / "best.pt"
    active_model, metadata = save_model_artifacts(
        best_weights=best_weights,
        output_dir=artifact_dir,
        class_names=list(report.class_names),
        metrics=metrics,
        run_dir=run_dir,
    )
    return {
        "run_dir": str(run_dir),
        "model_path": str(active_model),
        "metadata_path": str(metadata),
        "class_names": list(report.class_names),
        "image_counts": report.image_counts,
        "metrics": metrics,
    }
