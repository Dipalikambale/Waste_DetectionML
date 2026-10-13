import hashlib
import json
from pathlib import Path

from waste_detection.artifacts import save_model_artifacts


def test_saves_versioned_weights_and_metadata(tmp_path: Path) -> None:
    source_weights = tmp_path / "best.pt"
    source_weights.write_bytes(b"trained checkpoint")
    artifact_dir = tmp_path / "artifacts"

    active_model, metadata_path = save_model_artifacts(
        best_weights=source_weights,
        output_dir=artifact_dir,
        class_names=["plastic"],
        metrics={"mAP50": 0.91},
        run_dir=tmp_path / "runs" / "trial",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert active_model.read_bytes() == source_weights.read_bytes()
    assert metadata["weights"] == "best.pt"
    assert metadata["sha256"] == hashlib.sha256(source_weights.read_bytes()).hexdigest()
    assert metadata["class_names"] == ["plastic"]
    assert list((artifact_dir / "versions").glob("*.pt"))
