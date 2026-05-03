#!/usr/bin/env python3
"""Prepare 14d sim LeRobot datasets for Motus training.

This script keeps the public Motus dataset loader unchanged by converting sim
datasets into the LeRobot shape that Motus already consumes:

- observation.qpos -> observation.state
- action stays action
- cam_*.color video keys -> observation.images.cam_* video keys

Unused sim-only columns such as qvel, qf, and ee poses are intentionally left in
the source dataset and are not copied into the Motus mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


MOTUS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from scripts.collect_motus_prompt_t5_cache import collect_dataset_prompt_cache  # noqa: E402

REQUIRED_META_FILES = ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl")
MOTUS_TARGET_PREFIX = "Motus_"
MOTUS_VALID_EPISODES_FILE = "motus_valid_episodes.json"
STATE_KEY = "observation.qpos"
TARGET_STATE_KEY = "observation.state"
ACTION_KEY = "action"
SCALAR_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
ROOT_FILES_TO_COPY = ("README.md", ".gitattributes", "episode_mapping.json")
VIDEO_KEY_MAP = {
    "cam_high.color": "observation.images.cam_high",
    "cam_left_wrist.color": "observation.images.cam_left_wrist",
    "cam_right_wrist.color": "observation.images.cam_right_wrist",
}
CANONICAL_14D_NAMES = [
    "left_joint_waist", "left_joint_shoulder", "left_joint_elbow", "left_joint_forearm_roll",
    "left_joint_wrist_angle", "left_joint_wrist_rotate", "left_gripper",
    "right_joint_waist", "right_joint_shoulder", "right_joint_elbow", "right_joint_forearm_roll",
    "right_joint_wrist_angle", "right_joint_wrist_rotate", "right_gripper",
]
GRIPPER_INDICES = (6, 13)
SIM_TO_REAL_RAW_GRIPPER_SCALE = 0.05 / 0.07
DEFAULT_VIDEO_COPY_WORKERS = 16


def progress(iterable: Any, **kwargs: Any) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_jsonlines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonlines_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def is_lerobot_dataset_root(path: Path) -> bool:
    return path.is_dir() and all((path / relative).exists() for relative in REQUIRED_META_FILES)


def discover_lerobot_datasets(input_path: Path) -> list[Path]:
    input_path = input_path.resolve()
    if is_lerobot_dataset_root(input_path):
        return [input_path]

    datasets: list[Path] = []
    for info_path in sorted(input_path.rglob("meta/info.json")):
        candidate = info_path.parent.parent
        if candidate.name.startswith(MOTUS_TARGET_PREFIX):
            continue
        if is_lerobot_dataset_root(candidate):
            datasets.append(candidate.resolve())

    seen: set[Path] = set()
    deduped: list[Path] = []
    for dataset_root in datasets:
        if dataset_root not in seen:
            deduped.append(dataset_root)
            seen.add(dataset_root)
    return deduped


def default_output_root_for_input(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if is_lerobot_dataset_root(input_path):
        return input_path.parent
    return input_path


def target_root_for_dataset(source_root: Path, input_path: Path, output_root: Path) -> Path:
    source_root = source_root.resolve()
    input_path = input_path.resolve()
    if is_lerobot_dataset_root(input_path):
        relative = Path(f"{MOTUS_TARGET_PREFIX}{source_root.name}")
    else:
        relative_source = source_root.relative_to(input_path)
        relative = relative_source.parent / f"{MOTUS_TARGET_PREFIX}{source_root.name}"
    return output_root / relative


def sanitized_dataset_key(dataset_name: str, action_dim: int) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", dataset_name).strip("_").lower()
    return f"{normalized}_{action_dim}d"


def feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...] | None:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if not isinstance(shape, list):
        return None
    return tuple(int(dim) for dim in shape)


def is_sim_motus_compatible(info: dict[str, Any]) -> tuple[bool, str]:
    state_shape = feature_shape(info, STATE_KEY)
    action_shape = feature_shape(info, ACTION_KEY)
    if state_shape != (14,):
        return False, f"{STATE_KEY} shape is {state_shape}, expected (14,)"
    if action_shape != (14,):
        return False, f"{ACTION_KEY} shape is {action_shape}, expected (14,)"
    missing_videos = [key for key in VIDEO_KEY_MAP if key not in info.get("features", {})]
    if missing_videos:
        return False, f"missing sim video features: {missing_videos}"
    return True, "ok"


def format_episode_path(path_template: str, episode_index: int, chunks_size: int) -> str:
    episode_chunk = episode_index // chunks_size
    return path_template.format(episode_chunk=episode_chunk, episode_index=episode_index)


def copy_small_file(source_path: Path, target_path: Path, overwrite: bool) -> None:
    if not source_path.exists():
        return
    if target_path.exists() and not overwrite:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def normalize_episodes_for_motus(target_root: Path, overwrite: bool) -> int:
    tasks = load_jsonlines(target_root / "meta" / "tasks.jsonl")
    episodes = load_jsonlines(target_root / "meta" / "episodes.jsonl")
    task_by_index = {
        int(task["task_index"]): task["task"]
        for task in tasks
        if "task_index" in task and isinstance(task.get("task"), str)
    }
    single_task = next(iter(task_by_index.values())) if len(task_by_index) == 1 else None

    updated = 0
    for episode in episodes:
        tasks_value = episode.get("tasks")
        has_tasks = (
            isinstance(tasks_value, list)
            and tasks_value
            and isinstance(tasks_value[0], str)
            and tasks_value[0].strip()
        )
        if has_tasks and not overwrite:
            continue

        prompt = ""
        if has_tasks:
            prompt = tasks_value[0].strip()
        elif "task_index" in episode and int(episode["task_index"]) in task_by_index:
            prompt = task_by_index[int(episode["task_index"])]
        elif single_task:
            prompt = single_task

        if prompt:
            episode["tasks"] = [prompt]
            updated += 1

    write_jsonlines_atomic(target_root / "meta" / "episodes.jsonl", episodes)
    return updated


def episode_prompt(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks")
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], str) and tasks[0].strip():
        return tasks[0].strip()
    task = episode.get("task")
    if isinstance(task, str) and task.strip() and not task.strip().isdigit():
        return task.strip()
    return ""


def transform_info(source_info: dict[str, Any]) -> dict[str, Any]:
    source_features = source_info["features"]
    target_features: dict[str, Any] = {}

    state_feature = dict(source_features[STATE_KEY])
    state_feature["shape"] = [14]
    state_feature["names"] = CANONICAL_14D_NAMES
    target_features[TARGET_STATE_KEY] = state_feature

    action_feature = dict(source_features[ACTION_KEY])
    action_feature["shape"] = [14]
    action_feature["names"] = CANONICAL_14D_NAMES
    target_features[ACTION_KEY] = action_feature

    for source_key, target_key in VIDEO_KEY_MAP.items():
        target_features[target_key] = dict(source_features[source_key])

    for key in SCALAR_KEYS:
        if key in source_features:
            target_features[key] = source_features[key]

    target_info = dict(source_info)
    target_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    target_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    target_info["features"] = target_features
    target_info["total_videos"] = int(source_info.get("total_episodes", 0)) * len(VIDEO_KEY_MAP)
    return target_info


def scale_gripper_stats(stats: dict[str, Any]) -> dict[str, Any]:
    scaled = dict(stats)
    for stat_name, values in stats.items():
        if not isinstance(values, list) or len(values) <= max(GRIPPER_INDICES):
            continue
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        updated = list(values)
        for gripper_index in GRIPPER_INDICES:
            updated[gripper_index] = float(updated[gripper_index]) * SIM_TO_REAL_RAW_GRIPPER_SCALE
        scaled[stat_name] = updated
    return scaled


def transform_stats_mapping(stats: dict[str, Any]) -> dict[str, Any]:
    transformed: dict[str, Any] = {}
    if STATE_KEY in stats:
        transformed[TARGET_STATE_KEY] = scale_gripper_stats(stats[STATE_KEY])
    if ACTION_KEY in stats:
        transformed[ACTION_KEY] = scale_gripper_stats(stats[ACTION_KEY])
    for source_key, target_key in VIDEO_KEY_MAP.items():
        if source_key in stats:
            transformed[target_key] = stats[source_key]
    for key in SCALAR_KEYS:
        if key in stats:
            transformed[key] = stats[key]
    return transformed


def write_transformed_meta(source_root: Path, target_root: Path, overwrite: bool) -> None:
    source_meta = source_root / "meta"
    target_meta = target_root / "meta"
    target_meta.mkdir(parents=True, exist_ok=True)

    info = transform_info(load_json(source_meta / "info.json"))
    with open(target_meta / "info.json", "w", encoding="utf-8") as file_obj:
        json.dump(info, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n")

    for filename in ("tasks.jsonl", "episodes.jsonl"):
        copy_small_file(source_meta / filename, target_meta / filename, overwrite=overwrite)

    stats_path = source_meta / "stats.json"
    if stats_path.exists():
        with open(target_meta / "stats.json", "w", encoding="utf-8") as file_obj:
            json.dump(transform_stats_mapping(load_json(stats_path)), file_obj, ensure_ascii=False, indent=4)
            file_obj.write("\n")

    episodes_stats_path = source_meta / "episodes_stats.jsonl"
    if episodes_stats_path.exists() and (overwrite or not (target_meta / "episodes_stats.jsonl").exists()):
        rows = []
        for row in load_jsonlines(episodes_stats_path):
            if isinstance(row.get("stats"), dict):
                row = dict(row)
                row["stats"] = transform_stats_mapping(row["stats"])
            rows.append(row)
        write_jsonlines_atomic(target_meta / "episodes_stats.jsonl", rows)


def files_match_size(source_path: Path, target_path: Path) -> bool:
    return target_path.is_file() and source_path.stat().st_size == target_path.stat().st_size


def link_or_copy_file(source_file: Path, target_file: Path, overwrite: bool) -> str:
    if target_file.exists():
        if files_match_size(source_file, target_file):
            return "skipped"
        if not overwrite:
            return "skipped"
        target_file.unlink()
    elif target_file.is_symlink():
        if not overwrite:
            return "skipped"
        target_file.unlink()

    target_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_file, target_file)
        return "linked"
    except OSError:
        shutil.copy2(source_file, target_file)
        return "copied"


def copy_directory_tree(source_path: Path, target_path: Path, overwrite: bool, workers: int) -> dict[str, int]:
    if not source_path.exists():
        return {"linked": 0, "copied": 0, "skipped": 0}
    if target_path.is_symlink():
        if overwrite:
            target_path.unlink()
        else:
            raise FileExistsError(f"Refusing to replace existing symlink: {target_path}")
    if target_path.exists() and not target_path.is_dir():
        if overwrite:
            target_path.unlink()
        else:
            raise FileExistsError(f"Refusing to replace existing non-directory path: {target_path}")

    files = [path for path in source_path.rglob("*") if path.is_file()]
    for source_dir in [path for path in source_path.rglob("*") if path.is_dir()]:
        (target_path / source_dir.relative_to(source_path)).mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)
    stats = {"linked": 0, "copied": 0, "skipped": 0}
    pending: list[tuple[Path, Path]] = []

    for source_file in files:
        target_file = target_path / source_file.relative_to(source_path)
        if target_file.exists():
            if files_match_size(source_file, target_file) or not overwrite:
                stats["skipped"] += 1
                continue
        pending.append((source_file, target_file))

    if not pending:
        print(f"Video {source_path.name}: skipped {stats['skipped']} existing files")
        return stats

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [
            executor.submit(
                link_or_copy_file,
                source_file,
                target_file,
                overwrite,
            )
            for source_file, target_file in pending
        ]
        for future in progress(
            as_completed(futures),
            total=len(futures),
            desc=f"Video {source_path.name} link/copy",
            unit="file",
        ):
            stats[future.result()] += 1
    return stats


def copy_video_directories(source_root: Path, target_root: Path, overwrite: bool, workers: int) -> dict[str, int]:
    info = load_json(source_root / "meta" / "info.json")
    total_episodes = int(info.get("total_episodes", 0))
    chunks_size = int(info.get("chunks_size", 1000))
    total_chunks = max(1, (total_episodes + chunks_size - 1) // chunks_size)
    stats = {"linked": 0, "copied": 0, "skipped": 0}
    for chunk_index in range(total_chunks):
        for source_key, target_key in VIDEO_KEY_MAP.items():
            source_path = source_root / "videos" / f"chunk-{chunk_index:03d}" / source_key
            target_path = target_root / "videos" / f"chunk-{chunk_index:03d}" / target_key
            current_stats = copy_directory_tree(source_path, target_path, overwrite=overwrite, workers=workers)
            for key, value in current_stats.items():
                stats[key] += value
    return stats


def source_episode_parquet_path(source_root: Path, episode_index: int) -> Path:
    info = load_json(source_root / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    return source_root / format_episode_path(str(info["data_path"]), episode_index, chunks_size)


def target_episode_parquet_path(target_root: Path, episode_index: int) -> Path:
    info = load_json(target_root / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    return target_root / format_episode_path(str(info["data_path"]), episode_index, chunks_size)


def fixed_size_float_array(values: list[Any], width: int) -> pa.Array:
    return pa.array(values, type=pa.list_(pa.float32(), list_size=width))


def map_sim_gripper_to_real_raw(values: list[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).copy()
    array[..., GRIPPER_INDICES] *= SIM_TO_REAL_RAW_GRIPPER_SCALE
    return array


def convert_episode_parquet(source_path: Path, target_path: Path, overwrite: bool) -> np.ndarray:
    if target_path.exists() and not overwrite:
        table = pq.read_table(target_path, columns=[ACTION_KEY])
        return np.asarray(table.column(ACTION_KEY).to_pylist(), dtype=np.float32)

    columns = [STATE_KEY, ACTION_KEY, *SCALAR_KEYS]
    table = pq.read_table(source_path, columns=columns)
    state_values = map_sim_gripper_to_real_raw(table.column(STATE_KEY).to_pylist())
    action_values = map_sim_gripper_to_real_raw(table.column(ACTION_KEY).to_pylist())

    target_arrays: dict[str, pa.Array] = {
        TARGET_STATE_KEY: fixed_size_float_array(state_values.tolist(), 14),
        ACTION_KEY: fixed_size_float_array(action_values.tolist(), 14),
    }
    for key in SCALAR_KEYS:
        target_arrays[key] = table.column(key)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(target_arrays), target_path)
    return action_values


def write_motus_valid_episodes(
    target_root: Path,
    valid_episode_indices: list[int],
    invalid_episodes: list[dict[str, Any]],
) -> None:
    payload = {
        "valid_episode_indices": valid_episode_indices,
        "invalid_episodes": invalid_episodes,
    }
    with open(target_root / "meta" / MOTUS_VALID_EPISODES_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def convert_data_and_compute_stats(
    source_root: Path,
    target_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    episodes = load_jsonlines(target_root / "meta" / "episodes.jsonl")
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    valid_episode_indices: list[int] = []
    invalid_episodes: list[dict[str, Any]] = []
    start_time = time.time()

    for episode in progress(episodes, desc=f"Convert {source_root.name}", unit="episode"):
        episode_index = int(episode["episode_index"])
        source_path = source_episode_parquet_path(source_root, episode_index)
        target_path = target_episode_parquet_path(target_root, episode_index)
        try:
            actions = convert_episode_parquet(source_path, target_path, overwrite=overwrite)
            current_min = actions.min(axis=0)
            current_max = actions.max(axis=0)
            action_min = current_min if action_min is None else np.minimum(action_min, current_min)
            action_max = current_max if action_max is None else np.maximum(action_max, current_max)
            valid_episode_indices.append(episode_index)
        except Exception as error:
            invalid_episodes.append(
                {
                    "episode_index": episode_index,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "error": str(error),
                }
            )

    write_motus_valid_episodes(target_root, valid_episode_indices, invalid_episodes)

    if action_min is None or action_max is None:
        raise ValueError(f"No readable sim action data found for {source_root}")

    return {
        "min": action_min.astype(float).tolist(),
        "max": action_max.astype(float).tolist(),
        "file_count": len(valid_episode_indices),
        "total_files_scanned": len(episodes),
        "invalid_file_count": len(invalid_episodes),
        "invalid_episodes": invalid_episodes,
        "valid_episode_indices": valid_episode_indices,
        "action_dim": 14,
        "processing_time_seconds": time.time() - start_time,
        "num_processes_used": 1,
    }


def t5_cache_is_complete(dataset_root: Path, t5_folder_name: str) -> bool:
    episodes = load_jsonlines(dataset_root / "meta" / "episodes.jsonl")
    if not episodes:
        return False
    for episode in episodes:
        rel_path = episode.get("t5_embedding_path")
        if not isinstance(rel_path, str) or not (dataset_root / rel_path).exists():
            return False
    return True


def init_wan_t5_encoder(wan_path: str, device: str, text_len: int) -> Any:
    try:
        from Motus.bak.wan.modules.t5 import T5EncoderModel  # type: ignore
    except Exception:
        bak_root = str((MOTUS_ROOT / "bak").resolve())
        if bak_root not in sys.path:
            sys.path.insert(0, bak_root)
        from wan.modules.t5 import T5EncoderModel  # type: ignore

    ckpt = os.path.join(wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=int(text_len),
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )


def encode_prompt_t5(encoder: Any, prompt: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        encoded = encoder([prompt], device)
    if isinstance(encoded, list):
        emb = encoded[0]
    elif isinstance(encoded, torch.Tensor):
        emb = encoded
    else:
        raise ValueError(f"Unexpected T5 encoder output type: {type(encoded)}")
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.detach().cpu()


def ensure_t5_cache(
    dataset_root: Path,
    *,
    t5_folder_name: str,
    wan_path: str | None,
    device: str | None,
    text_len: int,
    overwrite: bool,
) -> str:
    episodes = load_jsonlines(dataset_root / "meta" / "episodes.jsonl")
    updated = 0
    out_dir = dataset_root / t5_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    complete_files = True
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        rel_path = f"{t5_folder_name}/episode_{episode_index:06d}.pt"
        if (dataset_root / rel_path).exists():
            episode["t5_embedding_path"] = rel_path
        else:
            complete_files = False
    if complete_files:
        write_jsonlines_atomic(dataset_root / "meta" / "episodes.jsonl", episodes)
        return "complete"

    if t5_cache_is_complete(dataset_root, t5_folder_name):
        return "complete"

    resolved_wan_path = wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
    if not resolved_wan_path:
        raise ValueError(
            f"T5 cache is incomplete for {dataset_root}, but no WAN path was provided. "
            "Use --wan-path or set WAN_PATH/WAN_ROOT."
        )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None
    prompt_cache: dict[str, torch.Tensor] = {}
    for episode in progress(episodes, desc=f"T5 cache {dataset_root.name}", unit="episode"):
        episode_index = int(episode["episode_index"])
        rel_path = f"{t5_folder_name}/episode_{episode_index:06d}.pt"
        abs_path = dataset_root / rel_path
        if abs_path.exists():
            episode["t5_embedding_path"] = rel_path
            continue

        prompt = episode_prompt(episode)
        if prompt not in prompt_cache:
            if encoder is None:
                print(f"Loading WAN T5 encoder from {resolved_wan_path} on {resolved_device} ...")
                encoder = init_wan_t5_encoder(resolved_wan_path, resolved_device, text_len)
            prompt_cache[prompt] = encode_prompt_t5(encoder, prompt, resolved_device)

        torch.save(prompt_cache[prompt], abs_path)
        episode["t5_embedding_path"] = rel_path
        updated += 1

    write_jsonlines_atomic(dataset_root / "meta" / "episodes.jsonl", episodes)
    return "generated" if updated > 0 else "pointers_fixed"


def motus_dataset_is_complete(
    target_root: Path,
    stat_json_path: Path,
    embodiment_type: str,
    t5_folder_name: str,
) -> tuple[bool, str]:
    if not is_lerobot_dataset_root(target_root):
        return False, "missing LeRobot metadata"
    if not (target_root / "meta" / MOTUS_VALID_EPISODES_FILE).exists():
        return False, f"missing meta/{MOTUS_VALID_EPISODES_FILE}"
    if not t5_cache_is_complete(target_root, t5_folder_name):
        return False, "missing complete T5 cache"
    if not stat_json_path.exists():
        return False, f"missing stat.json: {stat_json_path}"
    stats = load_json(stat_json_path)
    entry = stats.get(embodiment_type)
    if not isinstance(entry, dict) or len(entry.get("min", [])) != 14 or len(entry.get("max", [])) != 14:
        return False, f"missing stat.json entry {embodiment_type}"
    return True, "complete"


def update_motus_stat_json(stat_json_path: Path, results: list[dict[str, Any]]) -> None:
    stat_json_path.parent.mkdir(parents=True, exist_ok=True)
    stats = load_json(stat_json_path) if stat_json_path.exists() else {}
    for result in results:
        if result.get("stat_entry") is not None and result.get("embodiment_type") is not None:
            stats[result["embodiment_type"]] = result["stat_entry"]
    temp_path = stat_json_path.with_suffix(stat_json_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file_obj:
        json.dump(stats, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n")
    temp_path.replace(stat_json_path)


def write_manifest(output_root: Path, results: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "sim_motus_lerobot_manifest.json"
    stat_patch_path = output_root / "sim_motus_stat_patch.json"
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump({"datasets": results}, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    stat_patch = {
        result["embodiment_type"]: result["stat_entry"]
        for result in results
        if result.get("stat_entry") is not None and result.get("embodiment_type") is not None
    }
    with open(stat_patch_path, "w", encoding="utf-8") as file_obj:
        json.dump(stat_patch, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def prepare_dataset(
    source_root: Path,
    target_root: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    t5_folder_name: str,
    wan_path: str | None,
    device: str | None,
    t5_text_len: int,
    stat_json_path: Path,
    video_copy_workers: int,
) -> dict[str, Any]:
    source_info = load_json(source_root / "meta" / "info.json")
    compatible, reason = is_sim_motus_compatible(source_info)
    embodiment_type = sanitized_dataset_key(source_root.name, action_dim=14)
    result: dict[str, Any] = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "dataset_name": source_root.name,
        "repo_id": target_root.name,
        "embodiment_type": embodiment_type,
        "compatible": compatible,
        "compatibility_reason": reason,
        "status": "planned" if dry_run else "prepared",
    }

    if not compatible:
        result["status"] = "skipped"
        return result

    if not dry_run and not overwrite:
        complete, complete_reason = motus_dataset_is_complete(
            target_root=target_root,
            stat_json_path=stat_json_path,
            embodiment_type=embodiment_type,
            t5_folder_name=t5_folder_name,
        )
        if complete:
            result["status"] = "complete"
            result["complete_reason"] = complete_reason
            result["prompt_t5_cache"] = collect_dataset_prompt_cache(
                target_root, MOTUS_ROOT / "t5_prompt_cache", overwrite=overwrite
            )
            return result

    if dry_run:
        return result

    target_root.mkdir(parents=True, exist_ok=True)
    write_transformed_meta(source_root, target_root, overwrite=overwrite)
    for filename in ROOT_FILES_TO_COPY:
        copy_small_file(source_root / filename, target_root / filename, overwrite=overwrite)
    normalize_episodes_for_motus(target_root, overwrite=overwrite)
    result["video_files"] = copy_video_directories(
        source_root,
        target_root,
        overwrite=overwrite,
        workers=video_copy_workers,
    )

    result["stat_entry"] = convert_data_and_compute_stats(source_root, target_root, overwrite=overwrite)
    result["t5_status"] = ensure_t5_cache(
        target_root,
        t5_folder_name=t5_folder_name,
        wan_path=wan_path,
        device=device,
        text_len=t5_text_len,
        overwrite=overwrite,
    )
    result["prompt_t5_cache"] = collect_dataset_prompt_cache(
        target_root, MOTUS_ROOT / "t5_prompt_cache", overwrite=overwrite
    )
    result["motus_dataset_params"] = {
        "repo_id": target_root.name,
        "root": str(target_root),
        "embodiment_type": embodiment_type,
        "enable_t5_fallback": False,
        "t5_folder_name": t5_folder_name,
        "video_backend": "pyav",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Motus-facing mirrors for 14d sim LeRobot datasets without modifying the source datasets."
        )
    )
    parser.add_argument("input", type=Path, help="Sim LeRobot dataset root, or a parent directory to scan.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where Motus mirror datasets will be created. Defaults to the source parent directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Refresh transformed metadata/data/cache files.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan; do not write anything.")
    parser.add_argument(
        "--stat-json-path",
        type=Path,
        default=MOTUS_ROOT / "data/utils/stat.json",
        help="Target Motus stat.json path. Computed stats are merged into this file by default.",
    )
    parser.add_argument(
        "--wan-path",
        type=str,
        default=None,
        help="Base path containing Wan2.2-TI2V-5B/. Required only when T5 cache is missing.",
    )
    parser.add_argument("--device", type=str, default=None, help="T5 device, e.g. cuda:0 or cpu.")
    parser.add_argument("--t5-folder-name", type=str, default="t5_embedding", help="T5 cache folder in each mirror.")
    parser.add_argument("--t5-text-len", type=int, default=512, help="WAN T5 text length.")
    parser.add_argument(
        "--video-copy-workers",
        type=int,
        default=DEFAULT_VIDEO_COPY_WORKERS,
        help="Parallel workers for video hardlink/copy. Existing same-size files are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output_root.resolve() if args.output_root is not None else default_output_root_for_input(input_path)
    dataset_roots = discover_lerobot_datasets(input_path)
    if not dataset_roots:
        raise FileNotFoundError(f"No LeRobot dataset roots found under {input_path}")

    results: list[dict[str, Any]] = []
    for source_root in dataset_roots:
        target_root = target_root_for_dataset(source_root, input_path, output_root)
        result = prepare_dataset(
            source_root=source_root,
            target_root=target_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            t5_folder_name=args.t5_folder_name,
            wan_path=args.wan_path,
            device=args.device,
            t5_text_len=args.t5_text_len,
            stat_json_path=args.stat_json_path.resolve(),
            video_copy_workers=args.video_copy_workers,
        )
        results.append(result)
        print(f"{result['status']}: {source_root} -> {target_root} ({result['compatibility_reason']})")
        if "video_files" in result:
            print(f"video files: {result['video_files']}")

    if not args.dry_run:
        write_manifest(output_root, results)
        print(f"Wrote manifest: {output_root / 'sim_motus_lerobot_manifest.json'}")
        print(f"Wrote stats patch: {output_root / 'sim_motus_stat_patch.json'}")
        update_motus_stat_json(args.stat_json_path.resolve(), results)
        print(f"Updated Motus stats: {args.stat_json_path.resolve()}")


if __name__ == "__main__":
    main()
