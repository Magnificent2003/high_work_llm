"""Compose only the selected evidence sources into the model request."""
from __future__ import annotations

from pathlib import Path
from .conditions import SOURCE_NAMES
from .manifest import Sample, read_json, compact_json

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
