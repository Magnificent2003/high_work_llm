"""Output validation and image-level evaluation, independent of model providers."""
from __future__ import annotations

import itertools
import json
import random
import re
from typing import Any

FIELD_NAMES = (
    "overall_risk_level", "work_at_height", "helmet_status",
    "harness_presence_status", "hook_attachment_status",
    "harness_usage_status", "final_risk_level",
)

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
    if not isinstance(obj.get("overall_risk_level"), str) or obj["overall_risk_level"] not in enums["overall_risk_level"]:
        errors.append("bad_overall_risk_level")
    persons = obj.get("persons")
    if not isinstance(persons, list):
        errors.append("persons_not_list")
        return {"valid": not errors, "errors": errors}
    allowed_person = {"person_id", *FIELD_NAMES[1:]}
    seen_ids: set[str] = set()
    for i, person in enumerate(persons):
        if not isinstance(person, dict):
            errors.append(f"person_{i}_not_object")
            continue
        person_id = person.get("person_id")
        if not isinstance(person_id, str) or not person_id.strip():
            errors.append(f"person_{i}_bad_person_id")
        elif person_id in seen_ids:
            errors.append(f"person_{i}_duplicate_person_id")
        else:
            seen_ids.add(person_id)
        errors.extend(f"person_{i}_unexpected_{k}" for k in person if k not in allowed_person)
        for key in FIELD_NAMES[1:]:
            if key not in person:
                errors.append(f"person_{i}_missing_{key}")
            elif not isinstance(person[key], str) or person[key] not in enums[key]:
                errors.append(f"person_{i}_bad_{key}")
    return {"valid": not errors, "errors": errors}


def normalize_output(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    persons = obj.get("persons")
    if not isinstance(persons, list):
        return None
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
    if pred_n is None or gold_n is None or not validate_schema(pred)["valid"]:
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
    return sum(
        validate_schema(a)["valid"] and validate_schema(b)["valid"]
        and normalize_output(a) == normalize_output(b) for a, b in pairs
    ) / len(pairs)


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


def summarize(rows: list[dict[str, Any]], conditions: dict[str, frozenset[str]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_rows": len(rows),
        "n_errors": sum(bool(row.get("error")) for row in rows),
        "conditions": {},
    }
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
            "n_gold_calls": len(exact),
            "n_errors": sum(bool(row.get("error")) for row in subset),
            "json_parse_rate": sum(row["parsed_output"] is not None for row in subset) / len(subset) if subset else None,
            "schema_valid_rate": len(valid) / len(subset) if subset else None,
            "exact_match_rate": sum(exact) / len(exact) if exact else None,
            "field_accuracy": field_acc,
            "repeat_exact_agreement": sum(stability) / len(stability) if stability else None,
            "mean_latency_s": sum(row["elapsed_s"] for row in subset) / len(subset) if subset else None,
        }
    if "full" in result["conditions"]:
        # Treat each image as one unit: average its repeats before pairing and
        # bootstrapping so additional calls do not inflate the sample count.
        def sample_accuracy(condition: str, field: str) -> dict[str, float]:
            grouped: dict[str, list[float]] = {}
            for row in rows:
                if row["condition"] != condition:
                    continue
                accuracy = row["gold_metrics"].get("field_accuracy", {}).get(field)
                if accuracy is not None:
                    grouped.setdefault(row["sample_id"], []).append(float(accuracy))
            return {key: sum(values) / len(values) for key, values in grouped.items()}

        deltas: dict[str, Any] = {}
        for condition in conditions:
            if condition == "full":
                continue
            for field in FIELD_NAMES:
                full_acc = sample_accuracy("full", field)
                other_acc = sample_accuracy(condition, field)
                paired = [full_acc[key] - other_acc[key] for key in sorted(full_acc.keys() & other_acc.keys())]
                if paired:
                    deltas.setdefault(condition, {})[field] = {
                        "mean_full_minus_ablated": sum(paired) / len(paired),
                        "bootstrap_95_ci": bootstrap_ci(paired, seed=seed),
                        "n_pairs": len(paired),
                    }
        result["paired_full_minus_ablated"] = deltas
        result["paired_policy"] = "Mean repeats within each image, then pair and bootstrap images."
    return result
