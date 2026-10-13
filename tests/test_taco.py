import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from waste_detection.taco import _format_yolo_box, prepare_taco_dataset
from waste_detection.training import _write_training_data_yaml
from waste_detection.validation import validate_dataset


def _taco_source(source_dir: Path) -> None:
    (source_dir / "images" / "batch_1").mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (100, 50), color="white").save(
            source_dir / "images" / "batch_1" / f"image_{index}.jpg"
        )
    payload = {
        "images": [
            {
                "id": index,
                "file_name": f"batch_1/image_{index}.jpg",
                "width": 200,
                "height": 100,
            }
            for index in range(2)
        ],
        "categories": [
            {"id": 1, "name": "Clear plastic bottle"},
            {"id": 2, "name": "Drink can"},
            {"id": 3, "name": "Cigarette"},
        ],
        "annotations": [
            {
                "id": 10 + index,
                "image_id": index,
                "category_id": index + 1,
                "bbox": [20, 10, 40, 20],
                "iscrowd": 0,
            }
            for index in range(2)
        ] + [
            {
                "id": 20,
                "image_id": 0,
                "category_id": 3,
                "bbox": [1, 1, 5, 5],
                "iscrowd": 0,
            }
        ],
    }
    (source_dir / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_taco_converts_and_splits_selected_categories(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted"
    _taco_source(source_dir)

    report = prepare_taco_dataset(source_dir, output_dir, seed=1)

    assert report["image_counts"] == {"train": 1, "val": 1}
    assert report["class_counts"]["train"] == {"bottle": 1}
    assert report["class_counts"]["val"] == {"can": 1}
    config = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == ["bottle", "can", "plastic_bag_or_wrapper"]
    report = validate_dataset(output_dir / "data.yaml")
    training_yaml = tmp_path / "ultralytics-data.yaml"
    _write_training_data_yaml(report, training_yaml)
    training_config = yaml.safe_load(training_yaml.read_text(encoding="utf-8"))
    assert Path(training_config["path"]).is_absolute()
    assert Path(training_config["train"]).is_absolute()
    assert Path(training_config["val"]).is_absolute()
    labels = list((output_dir / "labels").rglob("*.txt"))
    assert len(labels) == 2
    assert all(
        label.read_text(encoding="utf-8").split()[1:] == [
            "0.20000000",
            "0.20000000",
            "0.20000000",
            "0.20000000",
        ]
        for label in labels
    )


def test_format_yolo_box_keeps_rounded_edges_inside_image() -> None:
    formatted = _format_yolo_box((0.10663738, 0.45803513, 0.21327477, 0.32791152))
    x_center, y_center, width, height = map(float, formatted.split())

    assert x_center - width / 2 >= 0
    assert x_center + width / 2 <= 1
    assert y_center - height / 2 >= 0
    assert y_center + height / 2 <= 1


def test_prepare_taco_refuses_to_overwrite_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted"
    _taco_source(source_dir)
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        prepare_taco_dataset(source_dir, output_dir)
