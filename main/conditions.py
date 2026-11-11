"""Definitions shared by full, baseline and source-ablation experiments."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

SOURCE_NAMES = ("original_image", "numbered_visualization", "scene_graph", "rule_result")
MODES = ("full", "base", "ablation", "leave_one_out", "all_subsets")


def make_conditions(mode: str, drop: Iterable[str] = ()) -> dict[str, frozenset[str]]:
    removed = frozenset(drop)
    unknown = removed - set(SOURCE_NAMES)
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(sorted(unknown))}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if removed and mode != "ablation":
        raise ValueError("--drop is only valid with --mode ablation")
    full = frozenset(SOURCE_NAMES)
    base = frozenset(("original_image",))
    if mode == "full":
        return {"full": full}
    if mode == "base":
        return {"base": base}
    if mode == "ablation":
        if not removed or removed == full:
            raise ValueError("Ablation requires --drop and must retain at least one source")
        name = "minus_" + "_".join(s for s in SOURCE_NAMES if s in removed)
        return {name: full - removed}
    if mode == "leave_one_out":
        return {"full": full, **{f"minus_{s}": full - {s} for s in SOURCE_NAMES}, "base": base}
    conditions = {}
    for size in range(1, len(SOURCE_NAMES) + 1):
        for subset in combinations(SOURCE_NAMES, size):
            sources = frozenset(subset)
            name = "full" if sources == full else "base" if sources == base else "plus_" + "_".join(subset)
            conditions[name] = sources
    return conditions
