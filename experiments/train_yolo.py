"""Train the YOLO detector on the rebuilt Work-at-Height dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="0 for first GPU, cpu for CPU")
    parser.add_argument("--project", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--name", default="yolo11s_rebuild")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
    )
    print("Training finished. Check the weights under the requested project/name directory.")


if __name__ == "__main__":
    main()
