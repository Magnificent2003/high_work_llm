"""Create four-source inputs from YOLO predictions for the end-to-end run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from prepare_evidence_from_yolo_labels import (
    build_rule_result,
    build_scene_graph,
    draw_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install ultralytics first: python -m pip install ultralytics") from exc

    images = sorted(args.image_dir.glob("*.jpg"))[: args.limit]
    if not images:
        raise FileNotFoundError(f"No JPG files found in {args.image_dir}")
    model = YOLO(str(args.weights))
    numbered_dir = args.outdir / "numbered_visualization"
    graph_dir = args.outdir / "scene_graph"
    rule_dir = args.outdir / "rule_result"
    for directory in (numbered_dir, graph_dir, rule_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image_path in images:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
        result = model.predict(source=str(image_path), conf=args.conf, device=args.device, verbose=False)[0]
        names = result.names
        objects = []
        for index, box in enumerate(result.boxes):
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            objects.append({
                "object_id": f"obj_{index:03d}",
                "class": str(names[cls_id]),
                "class_id": cls_id,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                "confidence": round(float(box.conf.item()), 5),
                "source": "yolo_prediction",
            })
        scene_graph = build_scene_graph(objects, width, height)
        scene_graph["image_id"] = image_path.stem
        scene_graph["evidence_source"] = "yolo_prediction"
        rule_result = build_rule_result(scene_graph)
        numbered_path = numbered_dir / f"{image_path.stem}.png"
        graph_path = graph_dir / f"{image_path.stem}.json"
        rule_path = rule_dir / f"{image_path.stem}.json"
        draw_visualization(image, objects, numbered_path)
        graph_path.write_text(json.dumps(scene_graph, ensure_ascii=False, indent=2), encoding="utf-8")
        rule_path.write_text(json.dumps(rule_result, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({
            "id": image_path.stem,
            "original_image": str(image_path.resolve()),
            "numbered_visualization": str(numbered_path.resolve()),
            "scene_graph": str(graph_path.resolve()),
            "rule_result": str(rule_path.resolve()),
        })
    manifest_path = args.outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} prediction-based samples")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
