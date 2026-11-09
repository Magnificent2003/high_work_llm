"""Create model-assisted gold-label drafts for human review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from four_source_ablation_demo import (  # noqa: E402
    SOURCE_NAMES,
    build_prompt,
    image_paths,
    load_env_file,
    load_manifest,
    parse_json,
    settings_from_env,
    validate_schema,
)

FIELDS = (
    "work_at_height",
    "helmet_status",
    "harness_presence_status",
    "hook_attachment_status",
    "harness_usage_status",
    "final_risk_level",
)

TASK_PROMPT = """你正在协助制作施工安全数据集的人工标注草稿。请依据原始现场图像进行保守判断，场景图和规则结果只作为辅助证据，不是最终真值。对每一名作业人员输出 work_at_height、helmet_status、harness_presence_status、hook_attachment_status、harness_usage_status 和 final_risk_level。无法从图像确认时使用 unclear，不要把检测到安全带主体等同于安全带规范使用。只输出一个合法 JSON 对象，字段为 overall_risk_level 和 persons。persons 中每项包含 person_id 以及上述六个状态字段。枚举值必须严格使用：work_at_height=yes/no/unclear；helmet_status=worn/not_worn/unclear；harness_presence_status=present/absent/unclear；hook_attachment_status=attached/not_attached/unclear/not_applicable；harness_usage_status=proper_use/improper_use/not_worn/unclear；final_risk_level=low/medium/high/unclear。"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / "llm" / ".env")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_env_file(args.env_file.resolve())
    samples = load_manifest(args.manifest.resolve())
    if args.limit:
        samples = samples[: args.limit]
    try:
        from llm import VLMInvoker
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install llm/requirements.txt before calling the API.") from exc

    invoker = VLMInvoker(settings=settings_from_env(0.0, 1024))
    args.outdir.mkdir(parents=True, exist_ok=True)
    drafts_dir = args.outdir / "draft_json"
    drafts_dir.mkdir(exist_ok=True)
    csv_path = args.outdir / "gold_review_queue.csv"
    rows: list[dict[str, str]] = []
    full_sources = frozenset(SOURCE_NAMES)

    for index, sample in enumerate(samples, start=1):
        prompt = build_prompt(sample, full_sources, TASK_PROMPT)
        started = time.perf_counter()
        raw_output = ""
        error = ""
        try:
            raw_output = invoker.chat(
                system_prompt="你是施工安全人工标注辅助员。只能输出 JSON。",
                user_prompt=prompt,
                image_paths=image_paths(sample, full_sources),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        parsed = parse_json(raw_output)
        schema = validate_schema(parsed)
        scene = json.loads(sample.scene_graph.read_text(encoding="utf-8")) if sample.scene_graph else {}
        persons = (parsed or {}).get("persons") or []
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
        draft_by_id = {str(p.get("person_id")): p for p in (parsed or {}).get("persons", []) if isinstance(p, dict)}
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
