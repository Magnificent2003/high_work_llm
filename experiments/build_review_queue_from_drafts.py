"""Build the editable review CSV from completed per-image draft JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "work_at_height",
    "helmet_status",
    "harness_presence_status",
    "hook_attachment_status",
    "harness_usage_status",
    "final_risk_level",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.draft_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        draft = record.get("draft_output") or {}
        persons = draft.get("persons") or []
        for person in persons:
            rows.append({
                "sample_id": record["sample_id"],
                "image_path": record["image_path"],
                "review_image": record["numbered_visualization"],
                "person_id": str(person.get("person_id", "")),
                "draft_overall_risk_level": str(draft.get("overall_risk_level", "")),
                **{f"draft_{field}": str(person.get(field, "")) for field in FIELDS},
                "human_overall_risk_level": "",
                **{f"human_{field}": "" for field in FIELDS},
                "review_status": "pending",
                "review_comment": "Model-assisted draft; verify against original image.",
            })
    if not rows:
        raise RuntimeError("No completed draft persons found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
