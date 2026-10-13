from __future__ import annotations

import shutil
from pathlib import Path


def ingest_dataset(source: Path, destination: Path) -> Path:
    """Copy a local dataset into a new destination without merging silently."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset source directory does not exist: {source}")
    if source == destination or source in destination.parents:
        raise ValueError("Dataset destination must not be inside the source directory")
    if destination.exists():
        raise FileExistsError(f"Dataset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination
