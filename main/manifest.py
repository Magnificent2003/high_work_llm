"""Portable sample manifests and input checks performed before model calls."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .conditions import SOURCE_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_sample_id(sample_id: str) -> None:
    if not sample_id.strip() or sample_id in {".", ".."} or any(c in sample_id for c in ("/", "\\", "\x00")):
        raise ValueError(f"Invalid sample id: {sample_id!r}; use a filename without path separators")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    original_image: Path | None = None
    numbered_visualization: Path | None = None
    scene_graph: Path | None = None
    rule_result: Path | None = None
    gold: Path | None = None


def resolve_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_json(path: Path | None) -> Any:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_manifest(path: Path) -> list[Sample]:
    path = path.expanduser().resolve()
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("samples", [])
    if not isinstance(data, list) or not data:
        raise ValueError("Manifest must be a non-empty list or {'samples': [...]}.")
    samples = []
    seen = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {index} must be an object")
        sample_id = str(item.get("id", f"sample_{index:03d}"))
        validate_sample_id(sample_id)
        if not sample_id.strip() or sample_id in seen:
            raise ValueError(f"Empty or duplicate sample id: {sample_id!r}")
        seen.add(sample_id)
        samples.append(Sample(sample_id, **{
            key: resolve_path(item.get(key), path.parent) for key in (*SOURCE_NAMES, "gold")
        }))
    return samples


def samples_from_images(image_dir: Path, gold_dir: Path | None = None) -> list[Sample]:
    images = sorted(p.resolve() for p in image_dir.expanduser().rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images found under {image_dir}")
    seen = set()
    samples = []
    for image in images:
        validate_sample_id(image.stem)
        if image.stem in seen:
            raise ValueError(f"Duplicate image stem {image.stem!r}; use a manifest with unique IDs")
        seen.add(image.stem)
        # A requested gold directory must contain a label for every selected sample.
        gold = (gold_dir / f"{image.stem}.json").resolve() if gold_dir else None
        samples.append(Sample(image.stem, original_image=image, gold=gold))
    return samples


def validate_samples(samples: list[Sample], conditions: dict[str, frozenset[str]]) -> None:
    from .metrics import validate_schema

    if not samples or not conditions:
        raise ValueError("Samples and conditions must not be empty")
    seen = set()
    required = set().union(*conditions.values())
    if not required or required - set(SOURCE_NAMES) or any(not s for s in conditions.values()):
        raise ValueError("Each condition must contain known, non-empty sources")
    for sample in samples:
        validate_sample_id(sample.sample_id)
        if not sample.sample_id.strip() or sample.sample_id in seen:
            raise ValueError(f"Empty or duplicate sample id: {sample.sample_id!r}")
        seen.add(sample.sample_id)
        for name in required | ({"gold"} if sample.gold else set()):
            path = getattr(sample, name)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Sample {sample.sample_id}: missing {name}: {path}")
            if name in {"scene_graph", "rule_result", "gold"}:
                value = read_json(path)
                if not isinstance(value, dict):
                    raise ValueError(f"Sample {sample.sample_id}: {name} must be a JSON object")
                if name == "gold":
                    schema = validate_schema(value)
                    if not schema["valid"]:
                        raise ValueError(f"Sample {sample.sample_id}: invalid gold: {schema['errors']}")
