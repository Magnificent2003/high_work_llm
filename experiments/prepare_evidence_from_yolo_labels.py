"""Create four-source demo inputs from an existing YOLO dataset split.

This first-stage generator uses ground-truth boxes as an oracle detector. It is
useful for isolating the value of evidence presentation. It must be replaced
by predicted boxes after the YOLO detector is trained for an end-to-end claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ["mewp", "person", "harness", "helmet", "scaffolding", "ladder"]
COLORS = {
    "mewp": (230, 126, 34),
    "person": (52, 152, 219),
    "harness": (231, 76, 60),
    "helmet": (241, 196, 15),
    "scaffolding": (155, 89, 182),
    "ladder": (39, 174, 96),
}


def parse_yolo_label(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        if len(fields) != 5:
            continue
        cls, xc, yc, w, h = int(fields[0]), *(float(v) for v in fields[1:])
        x1 = max(0, int((xc - w / 2) * width))
        y1 = max(0, int((yc - h / 2) * height))
        x2 = min(width - 1, int((xc + w / 2) * width))
        y2 = min(height - 1, int((yc + h / 2) * height))
        objects.append({
            "object_id": f"obj_{index:03d}",
            "class": CLASS_NAMES[cls],
            "class_id": cls,
            "bbox_xyxy": [x1, y1, x2, y2],
            "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
            "source": "ground_truth_label",
        })
    return objects


def center_distance(a: dict[str, Any], b: dict[str, Any], width: int, height: int) -> float:
    ax, ay = a["center"]
    bx, by = b["center"]
    return math.hypot((ax - bx) / width, (ay - by) / height)


def contains_center(person: dict[str, Any], target: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = person["bbox_xyxy"]
    x, y = target["center"]
    return x1 <= x <= x2 and y1 <= y <= y2


def build_scene_graph(objects: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    persons = [obj for obj in objects if obj["class"] == "person"]
    for i, person in enumerate(persons):
        person["person_id"] = f"person_{i:03d}"
    person_records = []
    for person in persons:
        attached = []
        relations = []
        for target in objects:
            if target is person:
                continue
            near = contains_center(person, target) or center_distance(person, target, width, height) <= 0.22
            if near:
                attached.append(target["object_id"])
                relations.append({
                    "subject": person["person_id"],
                    "relation": "near",
                    "object": target["object_id"],
                    "distance_normalized": round(center_distance(person, target, width, height), 4),
                })
        person_records.append({
            "person_id": person["person_id"],
            "person_object_id": person["object_id"],
            "associated_object_ids": attached,
            "relations": relations,
        })
    return {
        "image_id": "",
        "representation_type": "person_centered_scene_graph",
        "evidence_source": "ground_truth_boxes_oracle_detector",
        "association_method": "containment_or_normalized_center_distance_0.22",
        "objects": objects,
        "persons": person_records,
    }


def build_rule_result(scene_graph: dict[str, Any]) -> dict[str, Any]:
    by_id = {obj["object_id"]: obj for obj in scene_graph["objects"]}
    rules = []
    for person in scene_graph["persons"]:
        classes = {by_id[obj_id]["class"] for obj_id in person["associated_object_ids"] if obj_id in by_id}
        rules.append({
            "person_id": person["person_id"],
            "candidate_work_at_height": bool(classes & {"mewp", "scaffolding", "ladder"}),
            "helmet_object_detected": "helmet" in classes,
            "harness_object_detected": "harness" in classes,
            "hook_connection": "unclear",
            "harness_compliance": "unclear",
            "needs_visual_review": True,
            "rule_note": "Object presence and proximity only; hook connection and compliance require visual evidence.",
        })
    return {
        "image_id": scene_graph["image_id"],
        "rule_type": "conservative_candidate_rules",
        "persons": rules,
    }


def draw_visualization(image: Image.Image, objects: list[dict[str, Any]], output: Path) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        label = f"{obj['object_id']}:{obj['class']}"
        color = COLORS.get(obj["class"], (255, 255, 255))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.rectangle((x1, max(0, y1 - 22), x1 + max(100, len(label) * 9), y1), fill=color)
        draw.text((x1 + 3, max(0, y1 - 20)), label, fill=(0, 0, 0), font=font)
    image.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-classes", type=int, default=2, help="至少包含多少个目标类别")
    parser.add_argument("--allow-no-person", action="store_true", help="允许没有 person 目标的图片")
    args = parser.parse_args()

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir}")
    candidates = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        class_ids = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                class_ids.add(int(fields[0]))
        if len(class_ids) < args.min_classes:
            continue
        if not args.allow_no_person and 1 not in class_ids:
            continue
        candidates.append(image_path)
        if len(candidates) >= args.limit:
            break
    images = candidates
    if not images:
        raise FileNotFoundError(f"No JPG files found in {image_dir}")

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
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label: {label_path}")
            objects = parse_yolo_label(label_path, width, height)
            scene_graph = build_scene_graph(objects, width, height)
            scene_graph["image_id"] = image_path.stem
            rule_result = build_rule_result(scene_graph)
            numbered_path = numbered_dir / f"{image_path.stem}.png"
            graph_path = graph_dir / f"{image_path.stem}.json"
            rule_path = rule_dir / f"{image_path.stem}.json"
            draw_visualization(image.copy(), objects, numbered_path)
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
    print(f"Generated {len(manifest)} samples")
    print(f"Manifest: {manifest_path.resolve()}")
    print("Evidence mode: ground-truth boxes (oracle detector), not YOLO predictions")


if __name__ == "__main__":
    main()
