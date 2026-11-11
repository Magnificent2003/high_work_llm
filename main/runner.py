"""Execute validated experiments with an injectable model client."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .manifest import Sample, read_json, validate_samples
from .metrics import exact_or_accuracy, parse_json, summarize, validate_schema
from .prompting import build_prompt, image_paths


def run_experiment(
    samples: list[Sample],
    conditions: dict[str, frozenset[str]],
    outdir: Path,
    system_prompt: str,
    task_prompt: str,
    repeats: int = 1,
    seed: int = 20261026,
    invoker: Any = None,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if invoker is None:
        raise ValueError("An invoker with a chat() method is required")
    validate_samples(samples, conditions)
    outdir.mkdir(parents=True, exist_ok=True)
    if any((outdir / name).exists() for name in ("raw_results.jsonl", "summary.json")):
        raise FileExistsError(f"Run outputs already exist in {outdir}; choose a new output directory")
    rows: list[dict[str, Any]] = []
    with (outdir / "raw_results.jsonl").open("x", encoding="utf-8") as handle:
        for sample in samples:
            gold = read_json(sample.gold)
            for name, sources in conditions.items():
                prompt = build_prompt(sample, sources, task_prompt)
                images = image_paths(sample, sources)
                for repeat in range(repeats):
                    started = time.perf_counter()
                    error = None
                    try:
                        raw_output = invoker.chat(
                            system_prompt=system_prompt,
                            user_prompt=prompt,
                            image_paths=images,
                            label_images=False,
                        )
                        if not isinstance(raw_output, str):
                            raise TypeError("Model client must return a string")
                    except Exception as exc:
                        # Persist failed calls so that they remain in metric denominators.
                        raw_output = ""
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - started
                    parsed = parse_json(raw_output)
                    row = {
                        "sample_id": sample.sample_id,
                        "condition": name,
                        "sources": sorted(sources),
                        "repeat": repeat,
                        "elapsed_s": round(elapsed, 3),
                        "error": error,
                        "raw_output": raw_output,
                        "parsed_output": parsed,
                        "schema": validate_schema(parsed),
                        "gold_metrics": exact_or_accuracy(parsed, gold),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(row)
                    print(f"{sample.sample_id}\t{name}\trepeat={repeat}\tvalid={row['schema']['valid']}")
    summary = summarize(rows, conditions, seed)
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
