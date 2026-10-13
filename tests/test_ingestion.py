from pathlib import Path

import pytest

from waste_detection.ingestion import ingest_dataset


def test_ingest_copies_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.yaml").write_text("names: [plastic]\n", encoding="utf-8")

    destination = ingest_dataset(source, tmp_path / "nested" / "dataset")

    assert (destination / "data.yaml").read_text(encoding="utf-8") == "names: [plastic]\n"


def test_ingest_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        ingest_dataset(source, destination)


def test_ingest_refuses_destination_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="inside the source"):
        ingest_dataset(source, source / "copy")
