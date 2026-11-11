"""Deterministic detection evidence shared by oracle and prediction workflows.

The association and conservative rule logic are preserved from the original
experiments. Proximity does not establish wearing or hook attachment.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

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
        if cls < 0 or cls >= len(CLASS_NAMES):
            raise ValueError(f"Unknown class id {cls} in {path}; expected {CLASS_NAMES}")
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
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        label = f"{obj['object_id']}:{obj['class']}"
        if obj.get("person_id"):
            label += f" ({obj['person_id']})"
        color = COLORS.get(obj["class"], (255, 255, 255))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.rectangle((x1, max(0, y1 - 22), x1 + max(100, len(label) * 9), y1), fill=color)
        draw.text((x1 + 3, max(0, y1 - 20)), label, fill=(0, 0, 0), font=font)
    image.save(output, quality=95)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(directory: Path) -> list[Path]:
    """Collect one dataset split and reject ambiguous manifest identifiers."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    images = sorted(path for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    stems: set[str] = set()
    for path in images:
        if path.stem in stems:
            raise ValueError(f"Duplicate image stem {path.stem!r} in {directory}; rename before preparing evidence")
        stems.add(path.stem)
    return images


def relative_path(path: Path, base: Path) -> str:
    """Serialize paths relative to the manifest, including images outside it."""
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError:  # Different Windows drives cannot share a relative path.
        return str(path.resolve())


def write_evidence(image_path: Path, image: Image.Image,
                   objects: list[dict[str, Any]], outdir: Path,
                   evidence_source: str) -> dict[str, str]:
    """Write all three derived sources and return their manifest entry."""
    width, height = image.size
    graph = build_scene_graph(objects, width, height)
    graph["image_id"] = image_path.stem
    graph["evidence_source"] = evidence_source
    rules = build_rule_result(graph)
    numbered_path = outdir / "numbered_visualization" / f"{image_path.stem}.png"
    graph_path = outdir / "scene_graph" / f"{image_path.stem}.json"
    rule_path = outdir / "rule_result" / f"{image_path.stem}.json"
    for path in (numbered_path, graph_path, rule_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    draw_visualization(image.copy(), objects, numbered_path)
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    rule_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "id": image_path.stem,
        "original_image": relative_path(image_path, outdir),
        "numbered_visualization": relative_path(numbered_path, outdir),
        "scene_graph": relative_path(graph_path, outdir),
        "rule_result": relative_path(rule_path, outdir),
    }


def write_manifest(entries: list[dict[str, str]], outdir: Path) -> Path:
    if not entries:
        raise ValueError("Cannot write an empty evidence manifest")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "manifest.json"
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
