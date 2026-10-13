from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from waste_detection.app import create_app
from waste_detection.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        model_path=tmp_path / "best.pt",
        metadata_path=tmp_path / "metadata.json",
        confidence=0.25,
        max_upload_bytes=1024 * 1024,
        host="127.0.0.1",
        port=5000,
        debug=False,
    )


class _Predictor:
    def predict(self, source: Image.Image, conf: float, verbose: bool) -> list[object]:
        assert source.size == (32, 24)
        assert conf == 0.25
        assert verbose is False
        box = SimpleNamespace(
            cls=SimpleNamespace(item=lambda: 0),
            conf=SimpleNamespace(item=lambda: 0.9),
            xyxy=[SimpleNamespace(tolist=lambda: [1, 2, 20, 18])],
        )
        return [SimpleNamespace(names={0: "plastic"}, boxes=[box])]


def test_health_readiness_and_web_page(tmp_path: Path) -> None:
    client = create_app(_settings(tmp_path)).test_client()

    assert client.get("/").status_code == 200
    assert client.get("/health").json == {"status": "ok", "model_available": False}
    assert client.get("/ready").status_code == 503


def test_predict_returns_detections_and_preview_image(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = create_app(settings, predictor=_Predictor()).test_client()
    image = Image.new("RGB", (32, 24), color="white")
    from io import BytesIO

    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)

    response = client.post(
        "/api/predict",
        data={"image": (stream, "test.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["detections"] == [{
        "class_id": 0,
        "class_name": "plastic",
        "confidence": 0.9,
        "box": [1.0, 2.0, 20.0, 18.0],
    }]
    assert response.json["width"] == 32
    assert response.json["image"]


def test_predict_rejects_missing_image(tmp_path: Path) -> None:
    response = create_app(_settings(tmp_path)).test_client().post("/api/predict")
    assert response.status_code == 400
    assert "image" in response.json["error"]


def test_predict_reports_missing_model(tmp_path: Path) -> None:
    from io import BytesIO

    image = BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    image.seek(0)
    response = create_app(_settings(tmp_path)).test_client().post(
        "/api/predict",
        data={"image": (image, "test.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503


def test_predict_rejects_oversized_dimensions(tmp_path: Path) -> None:
    from io import BytesIO

    image = BytesIO()
    Image.new("RGB", (5000, 5000)).save(image, format="PNG")
    image.seek(0)
    response = create_app(_settings(tmp_path)).test_client().post(
        "/api/predict",
        data={"image": (image, "large.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "pixel limit" in response.json["error"]
