"""Create model-assisted gold-label drafts for human review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.config import ROOT, load_env_file, settings_from_env
from main.conditions import SOURCE_NAMES
from main.manifest import load_manifest, validate_samples
from main.metrics import parse_json, validate_schema
from main.prompting import build_prompt, image_paths

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/gold_draft")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-prompt-file", type=Path, default=ROOT / "prompts/gold_draft.txt")
    parser.add_argument("--system-prompt-file", type=Path, default=ROOT / "prompts/gold_draft_system.txt")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    task_prompt = args.task_prompt_file.read_text(encoding="utf-8")
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()

    load_env_file(args.env_file.resolve())
    samples = load_manifest(args.manifest.resolve())
    if args.limit:
        samples = samples[: args.limit]
    full_sources = frozenset(SOURCE_NAMES)
    validate_samples(samples, {"full": full_sources})
    try:
        from llm import VLMInvoker
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install requirements.txt before calling the API.") from exc

    invoker = VLMInvoker(settings=settings_from_env())
    args.outdir.mkdir(parents=True, exist_ok=True)
    drafts_dir = args.outdir / "draft_json"
    drafts_dir.mkdir(exist_ok=True)
    csv_path = args.outdir / "gold_review_queue.csv"
    rows: list[dict[str, str]] = []

    for index, sample in enumerate(samples, start=1):
        prompt = build_prompt(sample, full_sources, task_prompt)
        started = time.perf_counter()
        raw_output = ""
        error = ""
        try:
            raw_output = invoker.chat(
                system_prompt=system_prompt,
                user_prompt=prompt,
                image_paths=image_paths(sample, full_sources),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        parsed = parse_json(raw_output)
        schema = validate_schema(parsed)
        scene = json.loads(sample.scene_graph.read_text(encoding="utf-8")) if sample.scene_graph else {}
        persons = (parsed or {}).get("persons")
        persons = [person for person in persons if isinstance(person, dict)] if isinstance(persons, list) else []
        if not persons:
            persons = [{"person_id": p.get("person_id")} for p in scene.get("persons", [])]
        draft_record = {
            "sample_id": sample.sample_id,
            "status": "draft_pending_human_review",
            "image_path": str(sample.original_image),
            "numbered_visualization": str(sample.numbered_visualization),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error": error,
            "schema": schema,
            "draft_output": parsed,
            "raw_output": raw_output,
        }
        (drafts_dir / f"{sample.sample_id}.json").write_text(json.dumps(draft_record, ensure_ascii=False, indent=2), encoding="utf-8")
        draft_by_id = {str(p.get("person_id")): p for p in persons if isinstance(p, dict)}
        overall = str((parsed or {}).get("overall_risk_level", ""))
        for person in persons:
            person_id = str(person.get("person_id", ""))
            draft_person = draft_by_id.get(person_id, {})
            rows.append({
                "sample_id": sample.sample_id,
                "image_path": str(sample.original_image),
                "review_image": str(sample.numbered_visualization),
                "person_id": person_id,
                "draft_overall_risk_level": overall,
                **{f"draft_{field}": str(draft_person.get(field, "")) for field in FIELDS},
                "human_overall_risk_level": "",
                **{f"human_{field}": "" for field in FIELDS},
                "review_status": "pending",
                "review_comment": "",
            })
        print(f"[{index}/{len(samples)}] {sample.sample_id} valid={schema['valid']} error={error or 'none'}")

    if not rows:
        raise RuntimeError("No person rows were generated. Check the manifest and scene graphs.")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Draft JSON files: {drafts_dir.resolve()}")
    print(f"Review CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
