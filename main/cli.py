"""Shared command line for the four-source application and ablation scripts."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

from .conditions import MODES, SOURCE_NAMES, make_conditions
from .config import PROMPTS_DIR, ROOT, load_env_file, settings_from_env
from .manifest import load_manifest, samples_from_images, validate_samples
from .prompting import build_prompt, image_paths
from .runner import run_experiment


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def optional_number(value: str, kind: type) -> int | float | None:
    if value.lower() in {"none", "null"}:
        return None
    number = kind(value)
    if not math.isfinite(number) or (kind is int and number < 1) or (kind is float and number < 0):
        raise ValueError(f"Invalid generation parameter: {value}")
    return number


def main(argv: list[str] | None = None, default_mode: str = "full") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path, help="Paths in JSON are relative to the manifest")
    inputs.add_argument("--image-dir", type=Path, help="Raw image directory; only for --mode base")
    parser.add_argument("--gold-dir", type=Path, help="Reviewed labels for --image-dir; stem.json")
    parser.add_argument("--mode", choices=MODES, default=default_mode)
    parser.add_argument("--drop", nargs="+", choices=SOURCE_NAMES, default=[])
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "run")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--temperature", help="Override env, e.g. 0 or none")
    parser.add_argument("--max-tokens", help="Override env, e.g. 1024 or none")
    parser.add_argument("--system-prompt-file", type=Path, default=PROMPTS_DIR / "system.txt")
    parser.add_argument("--task-prompt-file", type=Path, default=PROMPTS_DIR / "safety_task.txt")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and save requests without an API call")
    args = parser.parse_args(argv)
    try:
        if args.image_dir and args.mode != "base":
            raise ValueError("--image-dir requires --mode base")
        if args.gold_dir and not args.image_dir:
            raise ValueError("--gold-dir requires --image-dir; otherwise attach gold in the manifest")
        conditions = make_conditions(args.mode, args.drop)
        samples = load_manifest(args.manifest) if args.manifest else samples_from_images(args.image_dir, args.gold_dir)
        if args.limit:
            samples = samples[:args.limit]
        validate_samples(samples, conditions)
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
        task_prompt = args.task_prompt_file.read_text(encoding="utf-8").strip()
        if not task_prompt:
            raise ValueError("Task prompt must not be empty")
        overrides = {}
        for key, kind in (("temperature", float), ("max_tokens", int)):
            value = getattr(args, key)
            if value is not None:
                overrides[key] = optional_number(value, kind)
        outdir = args.outdir.expanduser().resolve()
        artifacts = ("config.json", "manifest.json", "raw_results.jsonl", "summary.json", "dry_run.json")
        if any((outdir / name).exists() for name in artifacts):
            raise FileExistsError(f"Run artifacts already exist in {outdir}; choose a new --outdir")

        config = {
            "mode": args.mode,
            "manifest": str(args.manifest.resolve()) if args.manifest else None,
            "image_dir": str(args.image_dir.resolve()) if args.image_dir else None,
            "sample_ids": [s.sample_id for s in samples],
            "conditions": {key: sorted(value) for key, value in conditions.items()},
            "repeats": args.repeats,
            "seed": args.seed,
            "planned_calls": len(samples) * len(conditions) * args.repeats,
            "dry_run": args.dry_run,
            "system_prompt": system_prompt,
            "task_prompt": task_prompt,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "task_prompt_sha256": hashlib.sha256(task_prompt.encode()).hexdigest(),
        }
        invoker = None
        if not args.dry_run:
            load_env_file(args.env_file.expanduser().resolve())
            settings = replace(settings_from_env(), **overrides)
            from llm import VLMInvoker

            invoker = VLMInvoker(settings=settings)
            # Store only reproducibility settings; never serialize credentials.
            config.update({key: getattr(settings, key) for key in (
                "provider", "model", "temperature", "max_tokens", "timeout_s", "request_stream", "trust_env",
            )})
        else:
            config["generation_overrides"] = overrides

        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        snapshot = [{"id": s.sample_id, **{
            name: str(getattr(s, name)) for name in (*SOURCE_NAMES, "gold") if getattr(s, name) is not None
        }} for s in samples]
        (outdir / "manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.dry_run:
            requests = [{
                "sample_id": sample.sample_id,
                "condition": name,
                "sources": sorted(sources),
                "image_paths": [str(p) for p in image_paths(sample, sources)],
                "user_prompt": build_prompt(sample, sources, task_prompt),
            } for sample in samples for name, sources in conditions.items()]
            (outdir / "dry_run.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Dry run passed: {len(samples)} samples, {len(conditions)} conditions, {config['planned_calls']} planned calls")
            print(f"Saved input plan: {outdir / 'dry_run.json'}")
            return
        summary = run_experiment(samples, conditions, outdir, system_prompt, task_prompt,
                                 repeats=args.repeats, seed=args.seed, invoker=invoker)
        print(f"Saved summary: {outdir / 'summary.json'}")
        if summary["n_errors"]:
            print(f"{summary['n_errors']} API calls failed; details saved in raw_results.jsonl", file=sys.stderr)
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    except ModuleNotFoundError as exc:
        parser.error(f"Missing dependency {exc.name!r}; install requirements.txt")
