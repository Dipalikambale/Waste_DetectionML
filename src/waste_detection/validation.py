from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ValidationReport:
    dataset_yaml: Path
    class_names: tuple[str, ...]
    image_counts: dict[str, int]
    label_counts: dict[str, int]
    split_paths: dict[str, Path]


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Dataset validation failed:\n- " + "\n- ".join(errors))


def _class_names(names: Any) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            indexed_names = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("class names must use integer IDs starting at zero") from error
        if sorted(indexed_names) != list(range(len(indexed_names))):
            raise ValueError("class IDs in names must be contiguous and start at zero")
        return [indexed_names[index] for index in range(len(indexed_names))]
    raise ValueError("data.yaml must define names as a list or integer-keyed mapping")


def _resolve_split(dataset: dict[str, Any], yaml_path: Path, split: str) -> Path:
    raw_path = dataset.get(split)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"data.yaml must define a non-empty {split} path")
    dataset_root = Path(dataset.get("path", yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = yaml_path.parent / dataset_root
    split_path = Path(raw_path)
    return (split_path if split_path.is_absolute() else dataset_root / split_path).resolve()


def validate_dataset(data_yaml: Path) -> ValidationReport:
    yaml_path = data_yaml.expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset configuration does not exist: {yaml_path}")
    try:
        dataset = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid dataset YAML: {error}") from error
    if not isinstance(dataset, dict):
        raise ValueError("Dataset YAML must contain a mapping")

    try:
        names = _class_names(dataset.get("names"))
    except ValueError as error:
        raise ValueError(f"Invalid class names: {error}") from error
    if not names or any(not name.strip() for name in names):
        raise ValueError("Dataset must define at least one non-empty class name")

    errors: list[str] = []
    image_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for split in ("train", "val"):
        try:
            image_dir = _resolve_split(dataset, yaml_path, split)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
            continue
        if not image_dir.is_dir():
            errors.append(f"{split} images directory does not exist: {image_dir}")
            continue
        label_dir = image_dir.parent.parent / "labels" / image_dir.name
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        image_counts[split] = len(images)
        if not images:
            errors.append(f"{split} images directory contains no supported images")
        labels_seen = 0
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                errors.append(f"Unreadable image {image_path}: {error}")
                continue
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if not label_path.is_file():
                continue
            labels_seen += 1
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as error:
                errors.append(f"Cannot read label file {label_path}: {error}")
                continue
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                location = f"{label_path}:{line_number}"
                if len(parts) != 5:
                    errors.append(f"{location}: expected class_id x_center y_center width height")
                    continue
                try:
                    class_id = int(parts[0])
                    box = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{location}: class ID and box coordinates must be numeric")
                    continue
                if not 0 <= class_id < len(names):
                    errors.append(f"{location}: class ID {class_id} is outside [0, {len(names) - 1}]")
                if not all(0 <= value <= 1 for value in box):
                    errors.append(f"{location}: box coordinates must be normalized to [0, 1]")
                x_center, y_center, width, height = box
                if width <= 0 or height <= 0:
                    errors.append(f"{location}: bounding box width and height must be positive")
                if x_center - width / 2 < 0 or x_center + width / 2 > 1:
                    errors.append(f"{location}: bounding box extends outside image width")
                if y_center - height / 2 < 0 or y_center + height / 2 > 1:
                    errors.append(f"{location}: bounding box extends outside image height")
        label_counts[split] = labels_seen

    if errors:
        raise DatasetValidationError(errors)
    return ValidationReport(
        dataset_yaml=yaml_path,
        class_names=tuple(names),
        image_counts=image_counts,
        label_counts=label_counts,
        split_paths={
            split: _resolve_split(dataset, yaml_path, split)
            for split in ("train", "val")
        },
    )
