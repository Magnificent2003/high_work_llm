"""Convert the human-reviewed CSV into gold JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    # Excel on Chinese Windows commonly saves CSV as GBK/CP936.  Try UTF-8
    # first, then fall back to the local encoding so reviewed files can be
    # finalized without a manual re-save.
    try:
        text = args.csv.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = args.csv.read_text(encoding="cp936")
    for row in csv.DictReader(text.splitlines()):
        groups[row["sample_id"]].append(row)
    args.outdir.mkdir(parents=True, exist_ok=True)
    allowed = {
        "work_at_height": {"yes", "no", "unclear"},
        "helmet_status": {"worn", "not_worn", "unclear"},
        "harness_presence_status": {"present", "absent", "unclear"},
        "hook_attachment_status": {"attached", "not_attached", "unclear", "not_applicable"},
        "harness_usage_status": {"proper_use", "improper_use", "not_worn", "unclear"},
        "final_risk_level": {"low", "medium", "high", "unclear"},
    }
    for sample_id, rows in groups.items():
        persons = []
        overall_values = set()
        for row in rows:
            if row.get("review_status", "").strip().lower() not in {"checked", "approved", "done"}:
                raise ValueError(f"{sample_id}/{row['person_id']} is not marked checked")
            overall = row.get("human_overall_risk_level", "").strip()
            if overall:
                overall_values.add(overall)
            person = {"person_id": row["person_id"]}
            for field in FIELDS:
                value = row.get(f"human_{field}", "").strip()
                if value not in allowed[field]:
                    raise ValueError(f"Invalid or missing {field} in {sample_id}/{row['person_id']}: {value!r}")
                person[field] = value
            persons.append(person)
        invalid = overall_values - {"low", "medium", "high", "unclear"}
        if invalid or not overall_values:
            raise ValueError(f"{sample_id} has invalid human_overall_risk_level values: {sorted(overall_values)}")
        # Reviewers may assign risk per person.  Collapse those labels to an
        # image-level label using the highest severity present; this preserves
        # the conservative safety interpretation for multi-person scenes.
        if "high" in overall_values:
            overall_risk = "high"
        elif "medium" in overall_values:
            overall_risk = "medium"
        elif "low" in overall_values:
            overall_risk = "low"
        else:
            overall_risk = "unclear"
        output = {"overall_risk_level": overall_risk, "persons": persons}
        (args.outdir / f"{sample_id}.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(groups)} gold files to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
