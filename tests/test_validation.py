from pathlib import Path

import pytest
import yaml
from PIL import Image

from waste_detection.validation import DatasetValidationError, validate_dataset


def _dataset(tmp_path: Path, label_text: str = "0 0.5 0.5 0.4 0.4\n") -> Path:
    for split in ("train", "val"):
        image_dir = tmp_path / "images" / split
        label_dir = tmp_path / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32), color="white").save(image_dir / "sample.jpg")
        if label_text:
            (label_dir / "sample.txt").write_text(label_text, encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump({
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "plastic"},
        }),
        encoding="utf-8",
    )
    return config


def test_validate_yolo_dataset(tmp_path: Path) -> None:
    report = validate_dataset(_dataset(tmp_path))
    assert report.class_names == ("plastic",)
    assert report.image_counts == {"train": 1, "val": 1}
    assert report.label_counts == {"train": 1, "val": 1}


def test_allows_background_images_without_label_files(tmp_path: Path) -> None:
    report = validate_dataset(_dataset(tmp_path, label_text=""))
    assert report.label_counts == {"train": 0, "val": 0}


@pytest.mark.parametrize(
    "label, expected",
    [
        ("0 1.2 0.5 0.2 0.2\n", "normalized to [0, 1]"),
        ("1 0.5 0.5 0.2 0.2\n", "class ID 1 is outside"),
        ("0 0.1 0.5 0.4 0.2\n", "extends outside image width"),
        ("not a label\n", "expected class_id"),
    ],
)
def test_rejects_invalid_labels(tmp_path: Path, label: str, expected: str) -> None:
    with pytest.raises(DatasetValidationError) as error:
        validate_dataset(_dataset(tmp_path, label_text=label))
    assert expected in str(error.value)
