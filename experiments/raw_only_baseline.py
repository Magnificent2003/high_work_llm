"""Run the raw-image baseline before rebuilding detector-derived evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_ROOT = ROOT / "llm"
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from four_source_ablation_demo import (  # noqa: E402
    exact_or_accuracy,
    load_env_file,
    parse_json,
    read_json,
    settings_from_env,
    validate_schema,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="原始图片目录")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=LLM_ROOT / ".env")
    parser.add_argument("--gold-dir", type=Path, default=None, help="可选；文件名为图片 stem + .json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_env_file(args.env_file.resolve())
    images = collect_images(args.image_dir.resolve())
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found under {args.image_dir.resolve()}")

    try:
        from llm import VLMInvoker
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The LLM library dependencies are missing. Install llm/requirements.txt "
            "in the environment used for the API demo."
        ) from exc

    invoker = VLMInvoker(settings=settings_from_env())
    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / "raw_only_results.jsonl"
    prompt = (
        "请对这张施工现场图片进行高空作业安全初筛。只依据图片中可见证据，"
        "看不清时使用 unclear，不要猜测。输出一个合法 JSON 对象，包含："
        "overall_risk_level，以及 persons 数组。每名人员包含 person_id、"
        "work_at_height、helmet_status、harness_presence_status、"
        "hook_attachment_status、harness_usage_status、final_risk_level。"
        "需要特别区分安全带主体可见和安全带是否规范使用。"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for image in images:
            gold_path = args.gold_dir / f"{image.stem}.json" if args.gold_dir else None
            gold = read_json(gold_path if gold_path and gold_path.exists() else None)
            started = time.perf_counter()
            error = None
            raw_output = ""
            try:
                raw_output = invoker.chat(
                    system_prompt="你是施工安全视觉核查助手，只输出 JSON。",
                    user_prompt=prompt,
                    image_paths=[image],
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            parsed = parse_json(raw_output)
            row = {
                "sample_id": image.stem,
                "image": str(image),
                "elapsed_s": round(time.perf_counter() - started, 3),
                "error": error,
                "raw_output": raw_output,
                "parsed_output": parsed,
                "schema": validate_schema(parsed),
                "gold_metrics": exact_or_accuracy(parsed, gold),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{image.name}\t{row['elapsed_s']:.1f}s\tvalid={row['schema']['valid']}")
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
