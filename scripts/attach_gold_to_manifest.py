"""Attach reviewed labels and rebase source paths into a new manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.config import ROOT
from main.conditions import SOURCE_NAMES
from main.evidence import relative_path
from main.manifest import load_manifest, read_json
from main.metrics import validate_schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "data/gold")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = load_manifest(args.manifest.resolve())
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = data["samples"] if isinstance(data, dict) else data
    for item, sample in zip(items, samples):
        gold_path = args.gold_dir / f"{sample.sample_id}.json"
        if not gold_path.is_file():
            raise FileNotFoundError(gold_path)
        schema = validate_schema(read_json(gold_path))
        if not schema["valid"]:
            raise ValueError(f"Invalid gold {gold_path}: {schema['errors']}")
        for source in SOURCE_NAMES:
            path = getattr(sample, source)
            if path is not None:
                item[source] = relative_path(path, args.out.parent)
        item["gold"] = relative_path(gold_path, args.out.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote gold-linked manifest: {args.out.resolve()}")


if __name__ == "__main__":
    main()
