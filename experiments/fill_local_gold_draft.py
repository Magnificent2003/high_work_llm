"""Fill a conservative local review draft from scene/rule evidence only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = {item["id"]: item for item in json.loads(args.manifest.read_text(encoding="utf-8"))}
    rows = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = manifest[row["sample_id"]]
            rule_path = Path(item["rule_result"])
            rules = json.loads(rule_path.read_text(encoding="utf-8"))
            rule_by_person = {str(p["person_id"]): p for p in rules.get("persons", [])}
            rule = rule_by_person.get(row["person_id"], {})
            row["draft_overall_risk_level"] = "unclear"
            row["draft_work_at_height"] = "yes" if rule.get("candidate_work_at_height") else "unclear"
            row["draft_helmet_status"] = "worn" if rule.get("helmet_object_detected") else "unclear"
            row["draft_harness_presence_status"] = "present" if rule.get("harness_object_detected") else "unclear"
            row["draft_hook_attachment_status"] = "unclear"
            row["draft_harness_usage_status"] = "unclear"
            row["draft_final_risk_level"] = "unclear"
            row["review_comment"] = "Local evidence draft only; inspect original image and numbered image."
            rows.append(row)
    if not rows:
        raise ValueError("No review rows found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} local draft person rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
