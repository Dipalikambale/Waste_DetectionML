from __future__ import annotations

import base64
import io
from typing import Any

import gradio as gr
import spaces
from flask.testing import FlaskClient
from PIL import Image, ImageDraw

from waste_detection.app import create_app

_flask_app = create_app()
_flask_client: FlaskClient = _flask_app.test_client()


@spaces.GPU
def predict_image(
    image: Image.Image | None,
) -> tuple[Image.Image | None, list[list[Any]]]:
    if image is None:
        return None, []

    upload = io.BytesIO()
    image.convert("RGB").save(upload, format="PNG")
    upload.seek(0)
    response = _flask_client.post(
        "/api/predict",
        data={"image": (upload, "upload.png")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    if response.status_code != 200 or payload is None:
        message = payload.get("error", "Prediction failed") if payload else "Prediction failed"
        raise gr.Error(message)

    preview = Image.open(
        io.BytesIO(base64.b64decode(payload["image"]))
    ).convert("RGB")
    draw = ImageDraw.Draw(preview)
    rows: list[list[Any]] = []
    for detection in payload["detections"]:
        x1, y1, x2, y2 = detection["box"]
        label = detection["class_name"]
        confidence = detection["confidence"]
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1, max(0, y1 - 14)), f"{label} {confidence:.2f}", fill="lime")
        rows.append([
            label,
            round(confidence, 4),
            f"{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}",
        ])
    return preview, rows


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload a waste image"),
    outputs=[
        gr.Image(type="pil", label="Predictions"),
        gr.Dataframe(
            headers=["Class", "Confidence", "Box (x1, y1, x2, y2)"],
            datatype=["str", "number", "str"],
            label="Detected objects",
        ),
    ],
    title="Waste Detection",
    description="Detect bottles, cans, and plastic bags or wrappers in an image.",
)
