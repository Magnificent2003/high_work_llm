"""Four-source ablation demo for the construction-safety VLM pipeline.

The four sources are:
  A: original image
  B: numbered detection visualization
  C: scene graph JSON
  D: rule-result JSON

The script runs the complete input and leave-one-source-out conditions with
the same model, prompt, decoding settings, and sample order. It writes raw
responses, parsed outputs, per-sample metrics, and a summary JSON. Expert
labels in the manifest are optional for a smoke test, but required for a
claim about task accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LLM_ROOT = ROOT / "llm"
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

SOURCE_NAMES = ("original_image", "numbered_visualization", "scene_graph", "rule_result")
LEAVE_ONE_OUT = {
    "full": frozenset(SOURCE_NAMES),
    "minus_original_image": frozenset(SOURCE_NAMES[1:]),
    "minus_numbered_visualization": frozenset((SOURCE_NAMES[0], SOURCE_NAMES[2], SOURCE_NAMES[3])),
    "minus_scene_graph": frozenset((SOURCE_NAMES[0], SOURCE_NAMES[1], SOURCE_NAMES[3])),
    "minus_rule_result": frozenset((SOURCE_NAMES[0], SOURCE_NAMES[1], SOURCE_NAMES[2])),
    "original_image_only": frozenset((SOURCE_NAMES[0],)),
}

FIELD_NAMES = (
    "overall_risk_level",
    "work_at_height",
    "helmet_status",
    "harness_presence_status",
    "hook_attachment_status",
    "harness_usage_status",
    "final_risk_level",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    original_image: Path | None
    numbered_visualization: Path | None
    scene_graph: Path | None
    rule_result: Path | None
    gold: Path | None


def load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE env file without printing secrets."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def env_optional(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = env_optional(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_optional_int(name: str, default: int | None) -> int | None:
    value = env_optional(name)
    if value is None or value.lower() in {"none", "null"}:
        return default
    return int(value)


def env_optional_float(name: str, default: float | None) -> float | None:
    value = env_optional(name)
    if value is None or value.lower() in {"none", "null"}:
        return default
    return float(value)


def settings_from_env(temperature_override: float | None = None, max_tokens_override: int | None = None) -> VLMSettings:
    from llm import VLMSettings

    provider = env_optional("VLM_PROVIDER", env_optional("LLM_PROVIDER", "openai")).lower()
    if provider == "openai":
        api_key = env_optional("OPENAI_API_KEY")
        base_url = env_optional("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = env_optional("OPENAI_VLM_MODEL", "gpt-4o-mini")
    elif provider == "dashscope":
        api_key = env_optional("DASHSCOPE_API_KEY")
        base_url = env_optional("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = env_optional("DASHSCOPE_VLM_MODEL", "qwen-vl-plus")
    else:
        raise ValueError(f"Unsupported VLM_PROVIDER: {provider}")
    if not api_key:
        raise RuntimeError(f"No API key found for provider={provider!r}.")
    return VLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url or "",
        model=model or "",
        timeout_s=float(env_optional("VLM_TIMEOUT_S", env_optional("LLM_TIMEOUT_S", "120"))),
        temperature=(temperature_override if temperature_override is not None else env_optional_float("VLM_TEMPERATURE", 0.0)),
        max_tokens=(max_tokens_override if max_tokens_override is not None else env_optional_int("VLM_MAX_TOKENS", 1024)),
        trust_env=env_bool("VLM_TRUST_ENV", False),
        request_stream=env_bool("VLM_REQUEST_STREAM", False),
    )


def resolve_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> list[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples", [])
    if not isinstance(data, list) or not data:
        raise ValueError("Manifest must be a non-empty JSON list or {'samples': [...]}.")
    samples: list[Sample] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {index} is not an object")
        sample_id = str(item.get("id", f"sample_{index:03d}"))
        samples.append(Sample(
            sample_id=sample_id,
            original_image=resolve_path(item.get("original_image"), path.parent),
            numbered_visualization=resolve_path(item.get("numbered_visualization"), path.parent),
            scene_graph=resolve_path(item.get("scene_graph"), path.parent),
            rule_result=resolve_path(item.get("rule_result"), path.parent),
            gold=resolve_path(item.get("gold"), path.parent),
        ))
    return samples


def read_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_prompt(sample: Sample, sources: frozenset[str], task_prompt: str) -> str:
    parts = [task_prompt.strip()]
    labels = {
        "original_image": "图A为原始现场图像。",
        "numbered_visualization": "图B为目标检测编号可视化图。",
        "scene_graph": "文本C为以人员为中心的场景结构化表示。",
        "rule_result": "文本D为规则层生成的初步结果。",
    }
    for name in SOURCE_NAMES:
        if name not in sources:
            continue
        if name in {"original_image", "numbered_visualization"}:
            parts.append(labels[name])
        else:
            source_path = getattr(sample, name)
            if source_path is None:
                raise ValueError(f"sample {sample.sample_id} is missing source {name}")
            value = read_json(source_path)
            parts.append(f"{labels[name]}\n```json\n{compact_json(value)}\n```")
    missing = [name for name in SOURCE_NAMES if name not in sources]
    if missing:
        parts.append("以下输入源已被实验设计移除，不得依据其内容进行猜测：" + ", ".join(missing))
    parts.append("只输出一个合法 JSON 对象，不要输出 Markdown、解释或额外字段。")
    return "\n\n".join(parts)


def image_paths(sample: Sample, sources: frozenset[str]) -> list[Path]:
    paths: list[Path] = []
    for name in ("original_image", "numbered_visualization"):
        if name in sources:
            value = getattr(sample, name)
            if value is None:
                raise ValueError(f"sample {sample.sample_id} is missing source {name}")
            paths.append(value)
    return paths


def parse_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def validate_schema(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"valid": False, "errors": ["not_a_json_object"]}
    errors: list[str] = []
    allowed_top = {"overall_risk_level", "persons"}
    errors.extend(f"unexpected_top_level_{k}" for k in obj if k not in allowed_top)
    enums = {
        "overall_risk_level": {"low", "medium", "high", "unclear"},
        "work_at_height": {"yes", "no", "unclear"},
        "helmet_status": {"worn", "not_worn", "unclear"},
        "harness_presence_status": {"present", "absent", "unclear"},
        "hook_attachment_status": {"attached", "not_attached", "unclear", "not_applicable"},
        "harness_usage_status": {"proper_use", "improper_use", "not_worn", "unclear"},
        "final_risk_level": {"low", "medium", "high", "unclear"},
    }
    if obj.get("overall_risk_level") not in enums["overall_risk_level"]:
        errors.append("bad_overall_risk_level")
    persons = obj.get("persons")
    if not isinstance(persons, list):
        errors.append("persons_not_list")
        return {"valid": not errors, "errors": errors}
    allowed_person = {"person_id", *FIELD_NAMES[1:]}
    for i, person in enumerate(persons):
        if not isinstance(person, dict):
            errors.append(f"person_{i}_not_object"); continue
        errors.extend(f"person_{i}_unexpected_{k}" for k in person if k not in allowed_person)
        for key in FIELD_NAMES[1:]:
            if key not in person: errors.append(f"person_{i}_missing_{key}")
            elif person[key] not in enums[key]: errors.append(f"person_{i}_bad_{key}")
    return {"valid": not errors, "errors": errors}

def normalize_output(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    persons = obj.get("persons") or []
    normalized_persons = []
    for index, person in enumerate(persons):
        if not isinstance(person, dict):
            continue
        person_id = str(person.get("person_id", index))
        normalized_persons.append({
            "person_id": person_id,
            **{field: person.get(field) for field in FIELD_NAMES[1:]},
        })
    normalized_persons.sort(key=lambda item: item["person_id"])
    return {"overall_risk_level": obj.get("overall_risk_level"), "persons": normalized_persons}


def field_values(obj: dict[str, Any] | None) -> dict[str, list[Any]]:
    normalized = normalize_output(obj)
    values: dict[str, list[Any]] = {field: [] for field in FIELD_NAMES}
    if normalized is None:
        return values
    values["overall_risk_level"] = [normalized.get("overall_risk_level")]
    for person in normalized["persons"]:
        for field in FIELD_NAMES[1:]:
            values[field].append(person.get(field))
    return values


def exact_or_accuracy(pred: dict[str, Any] | None, gold: dict[str, Any] | None) -> dict[str, Any]:
    if gold is None:
        return {"gold_available": False}
    pred_n = normalize_output(pred)
    gold_n = normalize_output(gold)
    if pred_n is None or gold_n is None:
        return {"gold_available": True, "exact_match": False, "field_accuracy": {field: 0.0 for field in FIELD_NAMES}}
    result: dict[str, Any] = {"gold_available": True, "exact_match": pred_n == gold_n, "field_accuracy": {}}
    result["field_accuracy"]["overall_risk_level"] = float(pred_n["overall_risk_level"] == gold_n["overall_risk_level"])
    pred_by_id = {p["person_id"]: p for p in pred_n["persons"]}
    gold_by_id = {p["person_id"]: p for p in gold_n["persons"]}
    ids = sorted(set(pred_by_id) | set(gold_by_id))
    result["person_count_match"] = len(pred_n["persons"]) == len(gold_n["persons"])
    for field in FIELD_NAMES[1:]:
        if not ids:
            result["field_accuracy"][field] = 0.0
        else:
            result["field_accuracy"][field] = sum(
                pred_by_id.get(pid, {}).get(field) == gold_by_id.get(pid, {}).get(field)
                for pid in ids
            ) / len(ids)
    return result


def pairwise_agreement(outputs: list[dict[str, Any] | None]) -> float | None:
    if len(outputs) < 2:
        return None
    pairs = list(itertools.combinations(outputs, 2))
    return sum(normalize_output(a) == normalize_output(b) for a, b in pairs) / len(pairs)


def bootstrap_ci(values: list[float], seed: int = 20261026, repeats: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(repeats):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def make_conditions(mode: str) -> dict[str, frozenset[str]]:
    if mode == "leave_one_out":
        return dict(LEAVE_ONE_OUT)
    conditions: dict[str, frozenset[str]] = {}
    for size in range(1, len(SOURCE_NAMES) + 1):
        for subset in itertools.combinations(SOURCE_NAMES, size):
            key = "plus_" + "_".join(subset)
            conditions[key] = frozenset(subset)
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest; see experiments/README.md")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=LLM_ROOT / ".env")
    parser.add_argument("--conditions", choices=("leave_one_out", "all_subsets"), default="leave_one_out")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--only-condition", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--system-prompt", default="你是施工安全视觉核查助手。")
    parser.add_argument("--task-prompt", default=(
        "请基于当前提供的证据，识别每名作业人员的高处作业状态、安全帽状态、安全带存在状态、"
        "挂钩连接状态、安全带使用状态和最终风险等级。只能判断证据支持的内容；证据不足时使用 unclear。"
        "严格只输出一个 JSON 对象，顶层仅允许 overall_risk_level 和 persons。persons 中每项仅允许 person_id、"
        "work_at_height、helmet_status、harness_presence_status、hook_attachment_status、harness_usage_status、final_risk_level。"
        "枚举值严格为：overall_risk_level/final_risk_level=low|medium|high|unclear；work_at_height=yes|no|unclear；"
        "helmet_status=worn|not_worn|unclear；harness_presence_status=present|absent|unclear；"
        "hook_attachment_status=attached|not_attached|unclear|not_applicable；"
        "harness_usage_status=proper_use|improper_use|not_worn|unclear。不要输出别名字段、布尔值或额外字段。"
    ))
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    load_env_file(args.env_file.resolve())
    samples = load_manifest(args.manifest.resolve())
    if args.limit:
        samples = samples[: args.limit]
    conditions = make_conditions(args.conditions)
    if args.only_condition:
        if args.only_condition not in conditions: raise ValueError(f"unknown condition: {args.only_condition}")
        conditions = {args.only_condition: conditions[args.only_condition]}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "config.json").write_text(json.dumps({
        "manifest": str(args.manifest.resolve()),
        "conditions": {key: sorted(value) for key, value in conditions.items()},
        "repeats": args.repeats,
        "model": env_optional("OPENAI_VLM_MODEL", env_optional("DASHSCOPE_VLM_MODEL", "unknown")),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from llm import VLMInvoker
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The LLM library dependencies are missing. Install llm/requirements.txt "
            "in the environment used for the API demo."
        ) from exc
    invoker = VLMInvoker(settings=settings_from_env(args.temperature, args.max_tokens))
    raw_path = args.outdir / "raw_results.jsonl"
    rows: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for sample in samples:
            gold = read_json(sample.gold)
            for condition_name, sources in conditions.items():
                prompt = build_prompt(sample, sources, args.task_prompt)
                images = image_paths(sample, sources)
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    try:
                        raw_output = invoker.chat(
                            system_prompt=args.system_prompt,
                            user_prompt=prompt,
                            image_paths=images,
                            label_images=False,
                        )
                        error = None
                    except Exception as exc:  # preserve failed calls for audit
                        raw_output = ""
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - started
                    parsed = parse_json(raw_output)
                    row = {
                        "sample_id": sample.sample_id,
                        "condition": condition_name,
                        "sources": sorted(sources),
                        "repeat": repeat,
                        "elapsed_s": round(elapsed, 3),
                        "error": error,
                        "raw_output": raw_output,
                        "parsed_output": parsed,
                        "schema": validate_schema(parsed),
                        "gold_metrics": exact_or_accuracy(parsed, gold),
                    }
                    raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_file.flush()
                    rows.append(row)
                    print(f"{sample.sample_id}\t{condition_name}\trepeat={repeat}\t{elapsed:.1f}s\tvalid={row['schema']['valid']}")
    summary = summarize(rows, conditions, args.seed)
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize(rows: list[dict[str, Any]], conditions: dict[str, frozenset[str]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"n_rows": len(rows), "conditions": {}}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        valid = [row for row in subset if row["schema"]["valid"]]
        exact = [row["gold_metrics"]["exact_match"] for row in subset if row["gold_metrics"].get("gold_available")]
        field_acc: dict[str, float] = {}
        for field in FIELD_NAMES:
            values = [row["gold_metrics"].get("field_accuracy", {}).get(field)
                      for row in subset if row["gold_metrics"].get("gold_available")]
            values = [float(v) for v in values if v is not None]
            if values:
                field_acc[field] = sum(values) / len(values)
        grouped: dict[str, list[dict[str, Any] | None]] = {}
        for row in subset:
            grouped.setdefault(row["sample_id"], []).append(row["parsed_output"])
        stability = [pairwise_agreement(values) for values in grouped.values()]
        stability = [float(v) for v in stability if v is not None]
        result["conditions"][condition] = {
            "n_samples": len(grouped),
            "n_calls": len(subset),
            "json_parse_rate": sum(row["parsed_output"] is not None for row in subset) / len(subset) if subset else None,
            "schema_valid_rate": len(valid) / len(subset) if subset else None,
            "exact_match_rate": sum(exact) / len(exact) if exact else None,
            "field_accuracy": field_acc,
            "repeat_exact_agreement": sum(stability) / len(stability) if stability else None,
            "mean_latency_s": sum(row["elapsed_s"] for row in subset) / len(subset) if subset else None,
        }
    if "full" in result["conditions"]:
        full_rows = {row["sample_id"]: row for row in rows if row["condition"] == "full" and row["repeat"] == 0}
        deltas: dict[str, Any] = {}
        for condition in conditions:
            if condition == "full":
                continue
            other_rows = {row["sample_id"]: row for row in rows if row["condition"] == condition and row["repeat"] == 0}
            for field in FIELD_NAMES:
                paired: list[float] = []
                for sample_id, full_row in full_rows.items():
                    other = other_rows.get(sample_id)
                    if not other:
                        continue
                    full_acc = full_row["gold_metrics"].get("field_accuracy", {}).get(field)
                    other_acc = other["gold_metrics"].get("field_accuracy", {}).get(field)
                    if full_acc is not None and other_acc is not None:
                        paired.append(float(full_acc) - float(other_acc))
                if paired:
                    deltas.setdefault(condition, {})[field] = {
                        "mean_full_minus_ablated": sum(paired) / len(paired),
                        "bootstrap_95_ci": bootstrap_ci(paired, seed=seed),
                        "n_pairs": len(paired),
                    }
        result["paired_full_minus_ablated"] = deltas
    return result


if __name__ == "__main__":
    main()
