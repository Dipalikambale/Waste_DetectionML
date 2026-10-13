from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from waste_detection import hf_app
from waste_detection.app import create_app
from waste_detection.config import Settings


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


def test_gradio_adapter_returns_preview_and_detection_rows(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        model_path=tmp_path / "best.pt",
        metadata_path=tmp_path / "metadata.json",
        confidence=0.25,
        max_upload_bytes=1024 * 1024,
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
    monkeypatch.setattr(
        hf_app,
        "_flask_client",
        create_app(settings, predictor=_Predictor()).test_client(),
    )

    preview, rows = hf_app.predict_image(Image.new("RGB", (32, 24), "white"))

    assert preview is not None
    assert preview.size == (32, 24)
    assert rows == [["plastic", 0.9, "1.00, 2.00, 20.00, 18.00"]]


def test_gradio_adapter_preserves_empty_input() -> None:
    assert hf_app.predict_image(None) == (None, [])
