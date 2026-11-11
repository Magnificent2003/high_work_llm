"""Generate four-source inputs from YOLO predictions for end-to-end evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.config import ROOT
from main.evidence import CLASS_NAMES, collect_images, write_evidence, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/datasets/work_at_height/images/valid")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/evidence/predicted")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.limit <= 0 or not 0 <= args.conf <= 1:
        parser.error("--limit must be positive and --conf must be between 0 and 1")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Detector weights do not exist: {args.weights}")
    images = collect_images(args.image_dir)[:args.limit]
    if not images:
        raise ValueError(f"No images found in {args.image_dir}")
    from PIL import Image
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install detector dependencies: python -m pip install -r requirements-yolo.txt") from exc
    model = YOLO(str(args.weights))
    names = model.names.values() if isinstance(model.names, dict) else model.names
    if set(names) != set(CLASS_NAMES):
        raise ValueError(f"Expected a Work-at-Height detector with classes {CLASS_NAMES}; got {model.names}")
    entries = []
    for image_path in images:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        result = model.predict(source=str(image_path), conf=args.conf,
                               device=args.device, verbose=False)[0]
        objects = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            objects.append({
                "object_id": f"obj_{index:03d}",
                "class": str(result.names[class_id]),
                "class_id": class_id,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                "confidence": round(float(box.conf.item()), 5),
                "source": "yolo_prediction",
            })
        entries.append(write_evidence(image_path, image, objects, args.outdir, "yolo_prediction"))
    manifest = write_manifest(entries, args.outdir)
    print(f"Generated {len(entries)} prediction samples: {manifest.resolve()}")


if __name__ == "__main__":
    main()
