"""Rescore saved responses after explicit alias repair; no model calls are made."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.config import ROOT
from main.metrics import FIELD_NAMES, exact_or_accuracy, summarize, validate_schema

KEYMAP = {'harness_body_status': 'harness_presence_status',
 'hook_connection': 'hook_attachment_status',
 'hook_connection_status': 'hook_attachment_status',
 'harness_compliance': 'harness_usage_status',
 'harness_compliance_status': 'harness_usage_status',
 'harness_overall_status': 'harness_usage_status',
 'harness_use_status': 'harness_usage_status',
 'harness_overall_usage_status': 'harness_usage_status',
 'harness_overall_use_status': 'harness_usage_status',
 'overall_harness_use_status': 'harness_usage_status',
 'work_at_height_status': 'work_at_height',
 'height_work_status': 'work_at_height',
 'high_altitude_work_status': 'work_at_height',
 'working_at_height_status': 'work_at_height',
 'safety_helmet_status': 'helmet_status',
 'risk_level': 'final_risk_level'}

VALMAP = {'harness_presence_status': {'worn': 'present',
                             'detected': 'present',
                             'visible': 'present',
                             'present': 'present',
                             'yes': 'present',
                             'unclear': 'unclear',
                             'absent': 'absent'},
 'hook_attachment_status': {'unclear': 'unclear',
                            'attached': 'attached',
                            'connected': 'attached',
                            'not_attached': 'not_attached',
                            'disconnected': 'not_attached',
                            'not_applicable': 'not_applicable'},
 'harness_usage_status': {'unclear': 'unclear',
                          'proper_use': 'proper_use',
                          'improper_use': 'improper_use',
                          'not_worn': 'not_worn',
                          'not_applicable': 'not_applicable'},
 'work_at_height': {'yes': 'yes',
                    'no': 'no',
                    'unclear': 'unclear',
                    'confirmed': 'yes',
                    True: 'yes',
                    False: 'no'},
 'helmet_status': {'worn': 'worn',
                   'wearing': 'worn',
                   'present': 'worn',
                   'detected': 'worn',
                   'yes': 'worn',
                   'not_worn': 'not_worn',
                   'unclear': 'unclear'},
 'final_risk_level': {'low': 'low', 'medium': 'medium', 'high': 'high', 'unclear': 'unclear'}}

def repair(obj):
    if not isinstance(obj, dict):
        return (None, [])
    out = {}
    notes = []
    if 'overall_risk_level' in obj:
        out['overall_risk_level'] = obj['overall_risk_level']
    elif 'overall_risk' in obj:
        out['overall_risk_level'] = obj['overall_risk']
        notes.append('key:overall_risk->overall_risk_level')
    if not isinstance(obj.get('persons'), list):
        return (out, notes)
    persons = []
    for p in obj['persons']:
        if not isinstance(p, dict):
            continue
        q = {'person_id': p['person_id']} if 'person_id' in p else {}
        for k, v in p.items():
            nk = KEYMAP.get(k, k)
            if nk not in FIELD_NAMES[1:]:
                continue
            if nk != k and nk in p:
                continue  # An explicit canonical field takes precedence over an alias.
            if nk != k:
                notes.append(f'key:{k}->{nk}')
            if nk in VALMAP:
                try:
                    nv = VALMAP[nk].get(v, VALMAP[nk].get(str(v)))
                except TypeError:
                    nv = None
                if nv is None:
                    continue
                if nv != v:
                    notes.append(f'value:{nk}:{v}->{nv}')
                q[nk] = nv
            else:
                q[nk] = v
        persons.append(q)
    out['persons'] = persons
    return (out, notes)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "data/gold")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.run.resolve() == args.out.resolve():
        parser.error("--out must differ from --run to preserve the original summary")
    rows = []
    scored_rows = []
    with (args.run / "raw_results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            if Path(sample_id).name != sample_id or sample_id in {".", ".."}:
                raise ValueError(f"Invalid sample id: {sample_id!r}")
            fixed, notes = repair(row.get("parsed_output"))
            gold_path = args.gold_dir / f"{sample_id}.json"
            gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.is_file() else None
            if gold is not None and not validate_schema(gold)["valid"]:
                raise ValueError(f"Invalid gold: {gold_path}")
            row["repaired_output"] = fixed
            row["repair_notes"] = notes
            row["repaired_schema"] = validate_schema(fixed)
            row["repaired_gold_metrics"] = exact_or_accuracy(fixed, gold)
            rows.append(row)
            scored_rows.append({
                **row,
                "parsed_output": fixed,
                "schema": row["repaired_schema"],
                "gold_metrics": row["repaired_gold_metrics"],
            })
    if not rows:
        raise ValueError("No saved responses found")
    conditions = {row["condition"]: frozenset(row.get("sources", [])) for row in rows}
    summary = summarize(scored_rows, conditions, args.seed)
    summary["mapping_policy"] = "Only explicit semantic key/value aliases; missing values are not filled."
    summary["gold_dir"] = str(args.gold_dir.resolve())
    summary["source_run"] = str(args.run.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rescored_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "valid_rate", "field_accuracy_mean", "exact_match_rate"])
        writer.writeheader()
        for condition, values in summary["conditions"].items():
            accuracy = [value for value in values["field_accuracy"].values() if value is not None]
            writer.writerow({
                "condition": condition,
                "valid_rate": values["schema_valid_rate"],
                "field_accuracy_mean": sum(accuracy) / len(accuracy) if accuracy else None,
                "exact_match_rate": values["exact_match_rate"],
            })
    print(f"Rescored {len(rows)} responses: {(args.out / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
