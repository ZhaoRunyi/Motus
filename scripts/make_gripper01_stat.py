#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


GRIPPER_DIMS = (6, 13)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def params_from_config(path: Path) -> dict:
    return dict(load_config(path)["dataset"]["params"])


def binarize_entry(entry: dict, *, threshold: float, full_width: float) -> tuple[np.ndarray, np.ndarray]:
    source_min = np.asarray(entry["min"], dtype=np.float32).copy()
    source_max = np.asarray(entry["max"], dtype=np.float32).copy()
    cutoff = threshold / full_width
    for dim in GRIPPER_DIMS:
        source_min[dim] = 1.0 if source_min[dim] >= cutoff else 0.0
        source_max[dim] = 1.0 if source_max[dim] >= cutoff else 0.0
    return source_min, source_max


def merged_gripper01_stats(stats: dict, names: list[str], *, threshold: float, full_width: float) -> dict:
    mins, maxs = [], []
    for name in names:
        if name not in stats:
            raise KeyError(f"{name!r} not found in stat.json")
        current_min, current_max = binarize_entry(stats[name], threshold=threshold, full_width=full_width)
        mins.append(current_min)
        maxs.append(current_max)

    merged_min = np.minimum.reduce(mins)
    merged_max = np.maximum.reduce(maxs)
    for dim in GRIPPER_DIMS:
        if (merged_min[dim], merged_max[dim]) != (0.0, 1.0):
            raise ValueError(f"gripper dim {dim} is not 0/1 after merge: {merged_min[dim]}..{merged_max[dim]}")
    return {
        "min": merged_min.astype(float).tolist(),
        "max": merged_max.astype(float).tolist(),
        "action_dim": int(merged_min.shape[0]),
        "gripper_type": "01",
        "source_embodiment_types": names,
        "gripper_threshold": threshold,
        "gripper_full_width": full_width,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stat-json", type=Path, default=root / "data/utils/stat.json")
    parser.add_argument("--source-config", type=Path, default=root / "configs/piper_multi_tasks_robotwin_like.yaml")
    parser.add_argument("--output-config", type=Path, default=root / "configs/piper_multi_tasks_gripper01.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_params = params_from_config(args.source_config)
    output_params = params_from_config(args.output_config)
    source_names = [str(name) for name in source_params["embodiment_types"]]
    output_name = str(output_params["embodiment_type"])
    with args.stat_json.open("r", encoding="utf-8") as file_obj:
        stats = json.load(file_obj)
    existing = dict(stats.get(output_name, {}))
    threshold = float(output_params.get("gripper_threshold", existing.get("gripper_threshold", 0.01)))
    full_width = float(output_params.get("gripper_full_width", existing.get("gripper_full_width", 0.10)))

    existing.update(merged_gripper01_stats(
        stats,
        source_names,
        threshold=threshold,
        full_width=full_width,
    ))
    stats[output_name] = existing

    with args.stat_json.open("w", encoding="utf-8") as file_obj:
        json.dump(stats, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")
    print(f"wrote {output_name} from {len(source_names)} source stats")


if __name__ == "__main__":
    main()
