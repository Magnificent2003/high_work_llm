"""Build or refill a human review queue from local evidence, without an API."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.manifest import load_manifest, read_json
from main.metrics import FIELD_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Existing review CSV; omitted to create a new queue")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = {sample.sample_id: sample for sample in load_manifest(args.manifest.resolve())}
    if args.csv:
        with args.csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
        for sample in samples.values():
            scene = read_json(sample.scene_graph) or {}
            for person in scene.get("persons", []):
                rows.append({
                    "sample_id": sample.sample_id,
                    "image_path": str(sample.original_image or ""),
                    "review_image": str(sample.numbered_visualization or ""),
                    "person_id": str(person["person_id"]),
                    **{f"draft_{field}": "" for field in FIELD_NAMES},
                    **{f"human_{field}": "" for field in FIELD_NAMES},
                    "review_status": "pending",
                    "review_comment": "",
                })
    if not rows:
        raise ValueError("No review rows found; check that scene graphs contain persons")
    rules = {}
    for sample_id, sample in samples.items():
        value = read_json(sample.rule_result) or {}
        rules[sample_id] = {str(person["person_id"]): person for person in value.get("persons", [])}
    for row in rows:
        rule = rules[row["sample_id"]].get(row["person_id"], {})
        row["draft_overall_risk_level"] = "unclear"
        row["draft_work_at_height"] = "yes" if rule.get("candidate_work_at_height") else "unclear"
        row["draft_helmet_status"] = "worn" if rule.get("helmet_object_detected") else "unclear"
        row["draft_harness_presence_status"] = "present" if rule.get("harness_object_detected") else "unclear"
        row["draft_hook_attachment_status"] = "unclear"
        row["draft_harness_usage_status"] = "unclear"
        row["draft_final_risk_level"] = "unclear"
        if not row.get("review_comment"):
            row["review_comment"] = "Local proximity draft only; inspect original image and numbered image."
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} local draft person rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
