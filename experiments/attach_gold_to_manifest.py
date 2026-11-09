"""Add reviewed gold paths to an existing experiment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    for item in data:
        gold = args.gold_dir / f"{item['id']}.json"
        if not gold.exists():
            raise FileNotFoundError(gold)
        item["gold"] = str(gold.resolve())
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote gold-linked manifest: {args.out.resolve()}")


if __name__ == "__main__":
    main()
