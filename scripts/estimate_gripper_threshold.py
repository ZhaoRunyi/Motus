#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import yaml
ARMS = ("left", "right")

def roots_for(data: Path, tasks: list[str]) -> list[Path]:
    if (data / "meta/info.json").exists():
        return [data]
    roots = [data / task for task in tasks if (data / task / "meta/info.json").exists()]
    return roots or sorted(path.parent.parent for path in data.glob("*/meta/info.json"))

def episodes(root: Path) -> list[int]:
    lines = (root / "meta/episodes.jsonl").read_text().splitlines()
    return [json.loads(line)["episode_index"] for line in lines if line.strip()]

def parquet_path(root: Path, episode: int) -> Path:
    info = json.loads((root / "meta/info.json").read_text())
    chunk = int(episode) // int(info.get("chunks_size", 1000))
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return (root / template.format(episode_chunk=chunk, episode_index=int(episode))).resolve()

def read_grippers(root: Path, episode: int) -> dict:
    table = pq.read_table(parquet_path(root, episode), columns=["observation.state", "action"])
    out = {}
    for name, column in (("state", "observation.state"), ("action", "action")):
        values = np.asarray(table[column].to_pylist(), dtype=np.float32)
        left, right = (6, 22) if values.shape[1] >= 23 else (6, 13)
        out[name] = {"left": values[:, left], "right": values[:, right]}
    return out

def grasp_value(values: np.ndarray, skip_frac: float, window_frac: float, min_diff: float) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    start, window = max(1, int(len(values) * skip_frac)), max(8, int(len(values) * window_frac))
    if len(values) < 8 or start + window > len(values):
        return None
    base = float(np.median(values[:start]))
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    stable_tol, diff = max(0.005, 0.05 * spread), max(min_diff, 0.15 * spread)
    candidates = []
    for left in range(start, len(values) - window + 1, max(1, window // 4)):
        chunk = values[left : left + window]
        if float(np.std(chunk)) <= stable_tol and base - float(np.mean(chunk)) >= diff:
            candidates.append(float(np.mean(chunk)))
    if candidates: return min(candidates)
    low = float(np.percentile(values[start:], 10))
    return low if base - low >= diff else None

def plot_curves(task_data: dict, task_values: dict, output: Path) -> None:
    fig, axes = plt.subplots(len(task_data), 2, figsize=(12, 2.2 * len(task_data)), squeeze=False)
    for row, (task, episode_data) in enumerate(task_data.items()):
        for col, arm in enumerate(ARMS):
            ax = axes[row][col]
            for episode in episode_data:
                for source, style in (("state", "-"), ("action", "--")):
                    y = episode[source][arm]
                    ax.plot(np.linspace(0, 1, len(y)), y, style, alpha=0.12, linewidth=0.8)
            if task_values.get(task) is not None: ax.axhline(task_values[task], color="red", linewidth=1.2)
            ax.set_ylim(-0.03, 1.05)
            ax.set_title(f"{task} {arm}")
    output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output, dpi=180)

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--config", type=Path, default=root / "configs/piper_multi_tasks_robotwin_like.yaml")
    parser.add_argument("--stat-json", type=Path, default=root / "data/utils/stat.json")
    parser.add_argument("--target-stat", default="piper_multi_tasks_dual_14d_gripper01")
    parser.add_argument("--plot", type=Path, default=root / "logs/gripper_threshold/multi_task_grippers.png")
    for name, default in (("full-width", 0.10), ("skip-frac", 0.20), ("window-frac", 0.12), ("min-diff", 0.03)):
        parser.add_argument(f"--{name}", type=float, default=default)
    args = parser.parse_args()
    tasks = [str(name) for name in yaml.safe_load(args.config.read_text())["dataset"]["task_name"]]
    task_data, episode_values, task_values = {}, {}, {}
    for dataset in roots_for(args.data, tasks):
        task_data[dataset.name], episode_values[dataset.name] = [], {}
        for episode in episodes(dataset):
            grips = read_grippers(dataset, episode)
            vals = [grasp_value(grips[source][arm], args.skip_frac, args.window_frac, args.min_diff) for source in ("state", "action") for arm in ARMS]
            vals = [value for value in vals if value is not None]
            episode_values[dataset.name][str(episode)] = max(vals) if vals else None
            task_data[dataset.name].append(grips)
        valid = [value for value in episode_values[dataset.name].values() if value is not None]
        task_values[dataset.name] = max(valid) if valid else None
    global_raw = max(value for value in task_values.values() if value is not None)
    plot_curves(task_data, task_values, args.plot)
    stats = json.loads(args.stat_json.read_text())
    entry = stats.setdefault(args.target_stat, {})
    entry["gripper_threshold"] = global_raw * args.full_width
    entry["gripper_grasp_values"] = {"episode": episode_values, "task": task_values, "global_raw_ratio": global_raw, "plot": str(args.plot)}
    args.stat_json.write_text(json.dumps(stats, indent=4, ensure_ascii=False) + "\n")
    print(f"threshold={entry['gripper_threshold']:.6f} raw_ratio={global_raw:.6f} plot={args.plot}")

if __name__ == "__main__":
    main()
