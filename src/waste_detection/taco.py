from __future__ import annotations

import json
import random
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import requests
import yaml
from PIL import Image


TACO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/"
    "29de1a9ba05a647b83a90f18d7772e20bb23d846/data/annotations.json"
)
CLASS_NAMES = ["bottle", "can", "plastic_bag_or_wrapper"]
CATEGORY_TO_CLASS = {
    "Clear plastic bottle": 0,
    "Glass bottle": 0,
    "Other plastic bottle": 0,
    "Aerosol": 1,
    "Drink can": 1,
    "Food Can": 1,
    "Garbage bag": 2,
    "Single-use carrier bag": 2,
    "Polypropylene bag": 2,
    "Produce bag": 2,
    "Cereal bag": 2,
    "Bread bag": 2,
    "Plastic film": 2,
    "Crisp packet": 2,
    "Other plastic wrapper": 2,
    "Retort pouch": 2,
    "Six pack rings": 2,
}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            temporary.replace(destination)
            return
        except (OSError, requests.RequestException):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def _download_missing_images(
    image_ids: list[int],
    images_by_id: dict[int, dict[str, Any]],
    image_root: Path,
    progress: Callable[[int, int], None] | None,
) -> list[str]:
    failures: list[str] = []
    downloads = {}
    for image_id in image_ids:
        record = images_by_id[image_id]
        file_name = str(record["file_name"])
        image_path = _safe_image_path(image_root, file_name)
        if image_path.is_file():
            continue
        url = record.get("flickr_640_url") or record.get("flickr_url")
        if not url:
            failures.append(file_name)
            continue
        downloads[image_path] = (file_name, str(url))

    total = len(downloads)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_download_file, url, image_path): (image_path, file_name)
            for image_path, (file_name, url) in downloads.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            image_path, file_name = futures[future]
            try:
                future.result()
            except (OSError, requests.RequestException):
                failures.append(file_name)
            if progress is not None:
                progress(completed, total)
    return failures


def _safe_image_path(image_root: Path, file_name: str) -> Path:
    relative = Path(*PurePosixPath(file_name).parts)
    root = image_root.resolve()
    image_path = (root / relative).resolve()
    if image_path != root and root not in image_path.parents:
        raise ValueError(f"Unsafe image path in TACO annotations: {file_name}")
    return image_path


def _read_annotations(annotations_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read TACO annotations at {annotations_path}: {error}") from error
    required = {"images", "annotations", "categories"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("TACO annotations must contain images, annotations, and categories")
    return payload


def _yolo_box(
    annotation: dict[str, Any],
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> tuple[float, ...] | None:
    bbox = annotation.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or source_width <= 0
        or source_height <= 0
    ):
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    x *= width / source_width
    box_width *= width / source_width
    y *= height / source_height
    box_height *= height / source_height
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(float(width), x + box_width)
    bottom = min(float(height), y + box_height)
    if right <= left or bottom <= top:
        return None
    return (
        (left + right) / (2 * width),
        (top + bottom) / (2 * height),
        (right - left) / width,
        (bottom - top) / height,
    )


def _format_yolo_box(box: tuple[float, ...]) -> str:
    x_center, y_center, width, height = (round(value, 8) for value in box)
    for center, size_name in ((x_center, "width"), (y_center, "height")):
        size = width if size_name == "width" else height
        while center - size / 2 < 0 or center + size / 2 > 1:
            size = round(max(0.0, size - 1e-8), 8)
            if size == 0:
                raise ValueError("Bounding box became empty after coordinate rounding")
        if size_name == "width":
            width = size
        else:
            height = size
    return " ".join(f"{coordinate:.8f}" for coordinate in (x_center, y_center, width, height))


def prepare_taco_dataset(
    source_dir: Path = Path("data/taco-source"),
    output_dir: Path = Path("data/taco-yolo"),
    val_ratio: float = 0.2,
    seed: int = 42,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Download target TACO images and convert their COCO boxes to YOLO format."""
    if not 0 < val_ratio < 1:
        raise ValueError("Validation ratio must be greater than 0 and less than 1")
    if limit is not None and limit < 2:
        raise ValueError("Image limit must be at least 2 so train and validation both have images")

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the source directory")
    annotations_path = source_dir / "annotations.json"
    image_root = source_dir / "images"
    source_dir.mkdir(parents=True, exist_ok=True)
    if not annotations_path.is_file():
        _download_file(TACO_ANNOTATIONS_URL, annotations_path)
    dataset = _read_annotations(annotations_path)

    category_ids: dict[int, int] = {}
    for category in dataset["categories"]:
        if not isinstance(category, dict):
            continue
        class_id = CATEGORY_TO_CLASS.get(str(category.get("name")))
        if class_id is not None:
            category_ids[int(category["id"])] = class_id

    images_by_id = {
        int(image["id"]): image
        for image in dataset["images"]
        if isinstance(image, dict) and "id" in image
    }
    annotations_by_image: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        if not isinstance(annotation, dict) or annotation.get("iscrowd", 0):
            continue
        try:
            image_id = int(annotation["image_id"])
            class_id = category_ids[int(annotation["category_id"])]
        except (KeyError, TypeError, ValueError):
            continue
        if image_id in images_by_id:
            annotations_by_image[image_id].append((class_id, annotation))

    candidate_ids = sorted(annotations_by_image)
    if len(candidate_ids) < 2:
        raise ValueError("TACO annotations contain fewer than two images with the selected classes")
    random.Random(seed).shuffle(candidate_ids)
    if limit is not None:
        candidate_ids = candidate_ids[:limit]
    if len(candidate_ids) < 2:
        raise ValueError("The selected image limit must include at least two annotated images")

    if output_dir.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dir}")
    temporary_output = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_output.exists():
        raise FileExistsError(f"Temporary output directory already exists: {temporary_output}")
    failures = _download_missing_images(candidate_ids, images_by_id, image_root, progress)
    selected_ids = [
        image_id
        for image_id in candidate_ids
        if _safe_image_path(image_root, str(images_by_id[image_id]["file_name"])).is_file()
    ]
    if len(selected_ids) < 2:
        raise RuntimeError(
            "Fewer than two selected TACO images could be downloaded. "
            f"Failed images: {len(failures)}"
        )
    validation_count = max(1, round(len(selected_ids) * val_ratio))
    split_ids = {
        "val": set(selected_ids[:validation_count]),
        "train": set(selected_ids[validation_count:]),
    }
    if not split_ids["train"]:
        raise ValueError("The selected images leave no training images; increase the limit")

    temporary_output.mkdir(parents=True)
    try:
        counts: dict[str, Counter[str]] = {
            "train": Counter(),
            "val": Counter(),
        }
        for split, image_ids in split_ids.items():
            for image_id in sorted(image_ids):
                image_record = images_by_id[image_id]
                file_name = str(image_record["file_name"])
                source_image = _safe_image_path(image_root, file_name)

                try:
                    with Image.open(source_image) as image:
                        image.verify()
                        width, height = image.size
                except (OSError, Image.DecompressionBombError) as error:
                    raise ValueError(f"Downloaded TACO image is unreadable: {source_image}: {error}") from error
                if width <= 0 or height <= 0:
                    raise ValueError(f"TACO image has invalid dimensions: {source_image}")
                try:
                    source_width = int(image_record["width"])
                    source_height = int(image_record["height"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"TACO annotation has invalid dimensions for {file_name}"
                    ) from error

                relative_image = Path(*PurePosixPath(file_name).parts)
                destination_image = temporary_output / "images" / split / relative_image
                destination_label = (
                    temporary_output / "labels" / split / relative_image.with_suffix(".txt")
                )
                destination_image.parent.mkdir(parents=True, exist_ok=True)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, destination_image)

                label_lines: list[str] = []
                for class_id, annotation in annotations_by_image[image_id]:
                    box = _yolo_box(
                        annotation,
                        source_width,
                        source_height,
                        width,
                        height,
                    )
                    if box is None:
                        continue
                    label_lines.append(f"{class_id} {_format_yolo_box(box)}")
                    counts[split][CLASS_NAMES[class_id]] += 1
                if not label_lines:
                    raise ValueError(f"TACO image has no valid target boxes: {source_image}")
                destination_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        config = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "names": CLASS_NAMES,
        }
        (temporary_output / "data.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary_output.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_output)
        raise

    return {
        "dataset_yaml": str(output_dir / "data.yaml"),
        "image_counts": {split: len(image_ids) for split, image_ids in split_ids.items()},
        "class_counts": {split: dict(counts[split]) for split in ("train", "val")},
        "classes": CLASS_NAMES,
        "failed_downloads": failures,
    }
