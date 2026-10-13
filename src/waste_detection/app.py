from __future__ import annotations

import base64
import io
import json
from typing import Any

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from waste_detection.config import Settings

MAX_IMAGE_PIXELS = 20_000_000


def create_app(
    settings: Settings | None = None,
    predictor: Any | None = None,
) -> Flask:
    settings = settings or Settings.from_env()
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    active_predictor = predictor

    def get_predictor() -> Any:
        nonlocal active_predictor
        if active_predictor is None:
            if not settings.model_path.is_file():
                raise FileNotFoundError(
                    f"Model weights not found at {settings.model_path}. "
                    "Train a model or set WASTE_MODEL_PATH."
                )
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "Inference requires the dependencies in requirements.txt"
                ) from error
            active_predictor = YOLO(str(settings.model_path))
        return active_predictor

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok", "model_available": settings.model_path.is_file()})

    @app.get("/ready")
    def ready() -> Any:
        available = settings.model_path.is_file() or predictor is not None
        return (
            jsonify({"ready": available}),
            200 if available else 503,
        )

    @app.post("/api/predict")
    def predict() -> Any:
        upload = request.files.get("image")
        if upload is None:
            return jsonify({"error": "Upload an image using the 'image' field"}), 400
        if not upload.filename:
            return jsonify({"error": "The uploaded image must have a filename"}), 400
        try:
            uploaded_image = Image.open(upload.stream)
            if uploaded_image.width * uploaded_image.height > MAX_IMAGE_PIXELS:
                return jsonify({"error": f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel limit"}), 400
            image = uploaded_image.convert("RGB")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            return jsonify({"error": f"Uploaded file is not a valid image: {error}"}), 400

        try:
            results = get_predictor().predict(
                source=image,
                conf=settings.confidence,
                verbose=False,
            )
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 503
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 503
        except Exception:
            app.logger.exception("Model inference failed")
            return jsonify({"error": "Model inference failed"}), 500

        if not results:
            return jsonify({"error": "Model returned no prediction result"}), 500
        result = results[0]
        names = result.names
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                detections.append({
                    "class_id": class_id,
                    "class_name": str(label),
                    "confidence": confidence,
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                })
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return jsonify({
            "width": image.width,
            "height": image.height,
            "detections": detections,
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
        })

    @app.errorhandler(413)
    def request_too_large(_: Exception) -> Any:
        return jsonify({"error": f"Image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit"}), 413

    @app.get("/api/model-metadata")
    def model_metadata() -> Any:
        if not settings.metadata_path.is_file():
            return jsonify({"error": "Model metadata is not available"}), 404
        try:
            metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            app.logger.exception("Could not read model metadata")
            return jsonify({"error": "Model metadata could not be read"}), 500
        return jsonify(metadata)

    return app


app = create_app()


if __name__ == "__main__":
    settings = Settings.from_env()
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
