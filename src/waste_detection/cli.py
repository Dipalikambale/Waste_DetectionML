from __future__ import annotations

import argparse
import json
from pathlib import Path

from waste_detection.ingestion import ingest_dataset
from waste_detection.taco import prepare_taco_dataset
from waste_detection.training import train_model
from waste_detection.validation import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Waste detection data and training tools")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Copy a dataset into a new destination")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("destination", type=Path)

    validate = commands.add_parser("validate", help="Validate a YOLO dataset")
    validate.add_argument("--data", type=Path, default=Path("data/waste.yaml"))

    taco = commands.add_parser(
        "prepare-taco",
        help="Download selected TACO images and convert COCO annotations to YOLO",
    )
    taco.add_argument("--source-dir", type=Path, default=Path("data/taco-source"))
    taco.add_argument("--output-dir", type=Path, default=Path("data/taco-yolo"))
    taco.add_argument("--val-ratio", type=float, default=0.2)
    taco.add_argument("--seed", type=int, default=42)
    taco.add_argument(
        "--limit",
        type=int,
        help="Optional image cap for a small first run; omit to use all matching images",
    )

    train = commands.add_parser("train", help="Train the YOLOv5 waste detector")
    train.add_argument("--data", type=Path, default=Path("data/waste.yaml"))
    train.add_argument("--weights", default="yolov5nu.pt")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--imgsz", type=int, default=416)
    train.add_argument("--batch", type=int, default=2)
    train.add_argument("--project-dir", type=Path)
    train.add_argument("--artifact-dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "ingest":
        destination = ingest_dataset(args.source, args.destination)
        print(f"Dataset copied to {destination}")
    elif args.command == "prepare-taco":
        def show_download_progress(completed: int, total: int) -> None:
            print(f"Downloading TACO images: {completed}/{total}", end="\r", flush=True)

        print(json.dumps(prepare_taco_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
            progress=show_download_progress,
        ), indent=2))
    elif args.command == "validate":
        report = validate_dataset(args.data)
        print(
            json.dumps(
                {
                    "data_yaml": str(report.dataset_yaml),
                    "classes": report.class_names,
                    "images": report.image_counts,
                    "label_files": report.label_counts,
                },
                indent=2,
            )
        )
    elif args.command == "train":
        print(json.dumps(train_model(
            data_yaml=args.data,
            weights=args.weights,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch_size=args.batch,
            project_dir=args.project_dir,
            artifact_dir=args.artifact_dir,
        ), indent=2))


if __name__ == "__main__":
    main()
