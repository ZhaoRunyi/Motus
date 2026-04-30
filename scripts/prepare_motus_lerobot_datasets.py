#!/usr/bin/env python3
"""Prepare read-only LeRobot datasets for Motus training.

The script creates a Motus-facing mirror for each source LeRobot dataset:

- metadata is copied into the mirror and normalized for Motus prompt lookup;
- heavy data/media paths in meta/info.json are rewritten to point back to the source dataset;
- original source datasets are never modified;
- WAN T5 cache and Motus normalization stats are prepared by default because
  Motus training needs them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


MOTUS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

REQUIRED_META_FILES = (
    "meta/info.json",
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
)
CACHE_DIRS_TO_COPY = ("t5_embedding",)
ROOT_FILES_TO_COPY = ("README.md", ".gitattributes", "episode_mapping.json")
MOTUS_VALID_EPISODES_FILE = "motus_valid_episodes.json"
PIPER_IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
MOTUS_TARGET_PREFIX = "Motus_"


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
    with open(temp_path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def is_lerobot_dataset_root(path: Path) -> bool:
    return path.is_dir() and all((path / relative).exists() for relative in REQUIRED_META_FILES)


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_lerobot_datasets(input_path: Path, ignored_roots: tuple[Path, ...] = ()) -> list[Path]:
    input_path = input_path.resolve()
    if is_lerobot_dataset_root(input_path):
        return [input_path]

    datasets: list[Path] = []
    for info_path in sorted(input_path.rglob("meta/info.json")):
        candidate = info_path.parent.parent
        if candidate.name.startswith(MOTUS_TARGET_PREFIX):
            continue
        if any(path_is_relative_to(candidate.resolve(), ignored_root.resolve()) for ignored_root in ignored_roots):
            continue
        if is_lerobot_dataset_root(candidate):
            datasets.append(candidate.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for dataset_root in datasets:
        if dataset_root not in seen:
            deduped.append(dataset_root)
            seen.add(dataset_root)
    return deduped


def discover_motus_mirror_datasets(input_path: Path) -> list[Path]:
    input_path = input_path.resolve()
    if is_lerobot_dataset_root(input_path) and input_path.name.startswith(MOTUS_TARGET_PREFIX):
        return [input_path]

    datasets: list[Path] = []
    for info_path in sorted(input_path.rglob("meta/info.json")):
        candidate = info_path.parent.parent
        if not candidate.name.startswith(MOTUS_TARGET_PREFIX):
            continue
        if is_lerobot_dataset_root(candidate):
            datasets.append(candidate.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for dataset_root in datasets:
        if dataset_root not in seen:
            deduped.append(dataset_root)
            seen.add(dataset_root)
    return deduped


def target_root_for_dataset(source_root: Path, input_path: Path, output_root: Path) -> Path:
    source_root = source_root.resolve()
    input_path = input_path.resolve()
    if is_lerobot_dataset_root(input_path):
        relative = Path(f"{MOTUS_TARGET_PREFIX}{source_root.name}")
    else:
        relative_source = source_root.relative_to(input_path)
        relative = relative_source.parent / f"{MOTUS_TARGET_PREFIX}{source_root.name}"
    return output_root / relative


def feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...] | None:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if not isinstance(shape, list):
        return None
    return tuple(int(dim) for dim in shape)


def is_piper_motus_compatible(info: dict[str, Any]) -> tuple[bool, str]:
    state_shape = feature_shape(info, "observation.state")
    action_shape = feature_shape(info, "action")
    if state_shape != (32,):
        return False, f"observation.state shape is {state_shape}, expected (32,)"
    if action_shape != (32,):
        return False, f"action shape is {action_shape}, expected (32,)"
    missing_images = [key for key in PIPER_IMAGE_KEYS if key not in info.get("features", {})]
    if missing_images:
        return False, f"missing Piper image features: {missing_images}"
    return True, "ok"


def sanitized_dataset_key(dataset_name: str, arms: str, action_dim: int) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", dataset_name).strip("_").lower()
    return f"{normalized}_{arms}_{action_dim}d"


def default_output_root_for_input(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if is_lerobot_dataset_root(input_path):
        return input_path.parent
    return input_path


def format_episode_path(path_template: str, episode_index: int, chunks_size: int) -> str:
    episode_chunk = episode_index // chunks_size
    return path_template.format(episode_chunk=episode_chunk, episode_index=episode_index)


def point_info_paths_to_source(source_root: Path, target_root: Path) -> None:
    info_path = target_root / "meta" / "info.json"
    info = load_json(info_path)
    for key in ("data_path", "video_path"):
        template = info.get(key)
        if isinstance(template, str) and template:
            source_template_path = source_root / template
            info[key] = os.path.relpath(source_template_path, start=target_root)
    with open(info_path, "w", encoding="utf-8") as file_obj:
        json.dump(info, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n")


def episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    info = load_json(dataset_root / "meta" / "info.json")
    data_path = info.get("data_path")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError(f"meta/info.json has no valid data_path: {dataset_root}")
    chunks_size = int(info.get("chunks_size", 1000))
    return dataset_root / format_episode_path(data_path, episode_index, chunks_size)


def write_motus_valid_episodes(
    dataset_root: Path,
    valid_episode_indices: list[int],
    invalid_episodes: list[dict[str, Any]],
) -> None:
    sidecar_path = dataset_root / "meta" / MOTUS_VALID_EPISODES_FILE
    payload = {
        "valid_episode_indices": valid_episode_indices,
        "invalid_episodes": invalid_episodes,
    }
    with open(sidecar_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def motus_valid_episodes_file_exists(dataset_root: Path) -> bool:
    return (dataset_root / "meta" / MOTUS_VALID_EPISODES_FILE).exists()


def get_action_dim(state_action_space: str, state_action_arms: str) -> int:
    from data.lerobot.slai_piper_policy import StateSpaceConfig, get_space_dim

    return get_space_dim(StateSpaceConfig(ids=state_action_space, arms=state_action_arms))


def get_embodiment_type(dataset_name: str, state_action_space: str, state_action_arms: str) -> str:
    action_dim = get_action_dim(state_action_space, state_action_arms)
    return sanitized_dataset_key(dataset_name, state_action_arms, action_dim)


def copy_small_file(source_path: Path, target_path: Path, overwrite: bool) -> None:
    if not source_path.exists():
        return
    if target_path.exists() and not overwrite:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def copy_meta_directory(source_root: Path, target_root: Path, overwrite: bool) -> None:
    source_meta = source_root / "meta"
    target_meta = target_root / "meta"
    target_meta.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_meta.iterdir()):
        if source_path.is_file():
            copy_small_file(source_path, target_meta / source_path.name, overwrite=overwrite)


def copy_directory_tree(source_path: Path, target_path: Path, overwrite: bool) -> None:
    if not source_path.exists():
        return
    if target_path.is_symlink():
        if overwrite:
            target_path.unlink()
        else:
            raise FileExistsError(f"Refusing to replace existing symlink: {target_path}")
    if target_path.exists() and not target_path.is_dir():
        raise FileExistsError(f"Refusing to replace existing non-directory path: {target_path}")
    if target_path.exists() and not overwrite:
        return
    shutil.copytree(source_path, target_path, dirs_exist_ok=overwrite)


def resolve_matching_motus_dataset_root(target_root: Path, motus_t5_source_root: Path) -> Path | None:
    motus_t5_source_root = motus_t5_source_root.resolve()
    target_name = target_root.name

    if (
        motus_t5_source_root.name == target_name
        and motus_t5_source_root.name.startswith(MOTUS_TARGET_PREFIX)
        and is_lerobot_dataset_root(motus_t5_source_root)
    ):
        return motus_t5_source_root

    direct_candidate = motus_t5_source_root / target_name
    if direct_candidate.name.startswith(MOTUS_TARGET_PREFIX) and is_lerobot_dataset_root(direct_candidate):
        return direct_candidate.resolve()

    for candidate in discover_motus_mirror_datasets(motus_t5_source_root):
        if candidate.name == target_name:
            return candidate
    return None


def seed_t5_cache_from_matching_motus_dataset(
    target_root: Path,
    *,
    motus_t5_source_root: Path | None,
    t5_folder_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_requested",
        "source_root": None,
        "copied_files": 0,
        "skipped_existing_files": 0,
    }
    if motus_t5_source_root is None:
        return result

    source_dataset_root = resolve_matching_motus_dataset_root(target_root, motus_t5_source_root)
    if source_dataset_root is None:
        result["status"] = "source_not_found"
        return result

    source_t5_root = source_dataset_root / t5_folder_name
    result["source_root"] = str(source_dataset_root)
    if not source_t5_root.is_dir():
        result["status"] = "source_cache_missing"
        return result

    target_t5_root = target_root / t5_folder_name
    target_t5_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for source_path in sorted(source_t5_root.rglob("*.pt")):
        relative_path = source_path.relative_to(source_t5_root)
        target_path = target_t5_root / relative_path
        if target_path.exists() and not overwrite:
            skipped += 1
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1

    result["copied_files"] = copied
    result["skipped_existing_files"] = skipped
    result["status"] = "seeded" if copied > 0 else "already_present"
    return result


def normalize_episodes_for_motus(dataset_root: Path, overwrite: bool) -> int:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks = load_jsonlines(tasks_path)
    episodes = load_jsonlines(episodes_path)

    task_by_index: dict[int, str] = {}
    for task in tasks:
        if "task_index" in task and isinstance(task.get("task"), str):
            task_by_index[int(task["task_index"])] = task["task"]

    single_task = next(iter(task_by_index.values())) if len(task_by_index) == 1 else None
    updated = 0
    for episode in episodes:
        tasks_value = episode.get("tasks")
        has_tasks = (
            isinstance(tasks_value, list)
            and len(tasks_value) > 0
            and isinstance(tasks_value[0], str)
            and tasks_value[0].strip()
        )
        if has_tasks and not overwrite:
            continue

        prompt = ""
        if has_tasks:
            prompt = tasks_value[0].strip()
        elif isinstance(episode.get("task"), str) and not episode["task"].strip().isdigit():
            prompt = episode["task"].strip()
        elif "task_index" in episode and int(episode["task_index"]) in task_by_index:
            prompt = task_by_index[int(episode["task_index"])]
        elif single_task:
            prompt = single_task

        if prompt:
            episode["tasks"] = [prompt]
            updated += 1

    write_jsonlines_atomic(episodes_path, episodes)
    return updated


def episodes_have_motus_tasks(dataset_root: Path) -> bool:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return False
    episodes = load_jsonlines(episodes_path)
    if not episodes:
        return False
    for episode in episodes:
        tasks_value = episode.get("tasks")
        if not (
            isinstance(tasks_value, list)
            and len(tasks_value) > 0
            and isinstance(tasks_value[0], str)
            and tasks_value[0].strip()
        ):
            return False
    return True


def compute_action_stat_entry(
    dataset_root: Path,
    state_action_space: str,
    state_action_arms: str,
    action_key: str = "action",
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    from data.lerobot.slai_piper_policy import StateSpaceConfig, get_space_dim, select_state_action_vector

    state_config = StateSpaceConfig(ids=state_action_space, arms=state_action_arms)
    action_dim = get_space_dim(state_config)
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    file_count = 0
    start_time = time.time()

    valid_episode_indices: list[int] = []
    invalid_episodes: list[dict[str, Any]] = []
    episodes = load_jsonlines(dataset_root / "meta" / "episodes.jsonl")

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        parquet_path = episode_parquet_path(dataset_root, episode_index)
        try:
            table = pq.read_table(parquet_path, columns=[action_key])
        except Exception as error:
            invalid_episodes.append(
                {
                    "episode_index": episode_index,
                    "path": str(parquet_path),
                    "error": str(error),
                }
            )
            continue

        actions = np.asarray(table.column(action_key).to_pylist(), dtype=np.float32)
        selected_actions = select_state_action_vector(actions, state_config).numpy()
        current_min = selected_actions.min(axis=0)
        current_max = selected_actions.max(axis=0)
        action_min = current_min if action_min is None else np.minimum(action_min, current_min)
        action_max = current_max if action_max is None else np.maximum(action_max, current_max)
        valid_episode_indices.append(episode_index)
        file_count += 1

    write_motus_valid_episodes(dataset_root, valid_episode_indices, invalid_episodes)

    if action_min is None or action_max is None:
        raise ValueError(f"No readable parquet action data found for {dataset_root}")

    return {
        "min": action_min.astype(float).tolist(),
        "max": action_max.astype(float).tolist(),
        "file_count": file_count,
        "total_files_scanned": len(episodes),
        "invalid_file_count": len(invalid_episodes),
        "invalid_episodes": invalid_episodes,
        "valid_episode_indices": valid_episode_indices,
        "action_dim": action_dim,
        "processing_time_seconds": time.time() - start_time,
        "num_processes_used": 1,
    }


def generate_t5_cache(
    target_root: Path,
    repo_id: str,
    t5_folder_name: str,
    wan_path: str | None,
    device: str | None,
    text_len: int,
) -> None:
    command = [
        sys.executable,
        str(MOTUS_ROOT / "data/lerobot/add_t5_cache_to_lerobot_dataset.py"),
        "--repo_id",
        repo_id,
        "--root",
        str(target_root),
        "--t5_folder_name",
        t5_folder_name,
        "--text_len",
        str(text_len),
    ]
    if wan_path:
        command.extend(["--wan_path", wan_path])
    if device:
        command.extend(["--device", device])
    subprocess.run(command, check=True)


def t5_cache_is_complete(dataset_root: Path, t5_folder_name: str) -> bool:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return False
    episodes = load_jsonlines(episodes_path)
    if not episodes:
        return False
    for episode in episodes:
        rel_path = episode.get("t5_embedding_path")
        if not isinstance(rel_path, str) or not (dataset_root / rel_path).exists():
            return False
    return True


def patch_existing_t5_pointers(dataset_root: Path, t5_folder_name: str) -> int:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = load_jsonlines(episodes_path)
    updated = 0
    for episode in episodes:
        current_path = episode.get("t5_embedding_path")
        if isinstance(current_path, str) and (dataset_root / current_path).exists():
            continue
        episode_index = int(episode["episode_index"])
        rel_path = f"{t5_folder_name}/episode_{episode_index:06d}.pt"
        if (dataset_root / rel_path).exists():
            episode["t5_embedding_path"] = rel_path
            updated += 1
    if updated:
        write_jsonlines_atomic(episodes_path, episodes)
    return updated


def ensure_t5_cache_for_motus(
    target_root: Path,
    repo_id: str,
    t5_folder_name: str,
    wan_path: str | None,
    device: str | None,
    text_len: int,
) -> str:
    patched = patch_existing_t5_pointers(target_root, t5_folder_name)
    if t5_cache_is_complete(target_root, t5_folder_name):
        return "complete" if patched == 0 else "pointers_fixed"

    resolved_wan_path = wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
    if not resolved_wan_path:
        raise ValueError(
            f"T5 cache is incomplete for {target_root}, but no WAN path was provided. "
            "Use --wan-path or set WAN_PATH/WAN_ROOT."
        )

    generate_t5_cache(
        target_root=target_root,
        repo_id=repo_id,
        t5_folder_name=t5_folder_name,
        wan_path=resolved_wan_path,
        device=device,
        text_len=text_len,
    )
    if not t5_cache_is_complete(target_root, t5_folder_name):
        raise RuntimeError(f"T5 generation finished but cache is still incomplete: {target_root}")
    return "generated"


def stat_json_has_entry(stat_json_path: Path, embodiment_type: str, action_dim: int) -> bool:
    if not stat_json_path.exists():
        return False
    stats = load_json(stat_json_path)
    entry = stats.get(embodiment_type)
    if not isinstance(entry, dict):
        return False
    return (
        isinstance(entry.get("min"), list)
        and isinstance(entry.get("max"), list)
        and len(entry["min"]) == action_dim
        and len(entry["max"]) == action_dim
    )


def motus_dataset_is_complete(
    target_root: Path,
    stat_json_path: Path,
    embodiment_type: str,
    action_dim: int,
    t5_folder_name: str,
) -> tuple[bool, str]:
    if not is_lerobot_dataset_root(target_root):
        return False, "missing LeRobot metadata"
    if not motus_valid_episodes_file_exists(target_root):
        return False, f"missing meta/{MOTUS_VALID_EPISODES_FILE}"
    if not episodes_have_motus_tasks(target_root):
        return False, "episodes.jsonl missing tasks prompts"
    if not t5_cache_is_complete(target_root, t5_folder_name):
        return False, "missing complete T5 cache"
    if not stat_json_has_entry(stat_json_path, embodiment_type, action_dim):
        return False, f"missing stat.json entry {embodiment_type}"
    return True, "complete"


def prepare_dataset(
    source_root: Path,
    target_root: Path,
    *,
    overwrite: bool,
    allow_incompatible: bool,
    dry_run: bool,
    state_action_space: str,
    state_action_arms: str,
    t5_folder_name: str,
    motus_t5_source_root: Path | None,
    wan_path: str | None,
    device: str | None,
    t5_text_len: int,
    stat_json_path: Path,
) -> dict[str, Any]:
    info = load_json(source_root / "meta/info.json")
    compatible, reason = is_piper_motus_compatible(info)
    action_dim = get_action_dim(state_action_space, state_action_arms)
    embodiment_type = sanitized_dataset_key(source_root.name, state_action_arms, action_dim)
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

    if not compatible and not allow_incompatible:
        result["status"] = "skipped"
        return result
    if not dry_run and not overwrite:
        complete, complete_reason = motus_dataset_is_complete(
            target_root=target_root,
            stat_json_path=stat_json_path,
            embodiment_type=embodiment_type,
            action_dim=action_dim,
            t5_folder_name=t5_folder_name,
        )
        if complete:
            result["status"] = "complete"
            result["complete_reason"] = complete_reason
            result["motus_dataset_params"] = {
                "repo_id": target_root.name,
                "root": str(target_root),
                "embodiment_type": embodiment_type,
                "state_action_space": state_action_space,
                "state_action_arms": state_action_arms,
                "enable_t5_fallback": False,
                "t5_folder_name": t5_folder_name,
                "video_backend": "pyav",
            }
            return result
    if dry_run:
        return result

    target_root.mkdir(parents=True, exist_ok=True)
    copy_meta_directory(source_root, target_root, overwrite=overwrite)
    for filename in ROOT_FILES_TO_COPY:
        copy_small_file(source_root / filename, target_root / filename, overwrite=overwrite)
    point_info_paths_to_source(source_root, target_root)
    for dirname in CACHE_DIRS_TO_COPY:
        copy_directory_tree(source_root / dirname, target_root / dirname, overwrite=overwrite)

    updated_episodes = normalize_episodes_for_motus(target_root, overwrite=overwrite)
    result["episodes_updated_for_tasks"] = updated_episodes

    if compatible:
        result["stat_entry"] = compute_action_stat_entry(
            target_root,
            state_action_space=state_action_space,
            state_action_arms=state_action_arms,
        )

    result["t5_seed"] = seed_t5_cache_from_matching_motus_dataset(
        target_root,
        motus_t5_source_root=motus_t5_source_root,
        t5_folder_name=t5_folder_name,
        overwrite=overwrite,
    )

    result["t5_status"] = ensure_t5_cache_for_motus(
        target_root=target_root,
        repo_id=target_root.name,
        t5_folder_name=t5_folder_name,
        wan_path=wan_path,
        device=device,
        text_len=t5_text_len,
    )

    result["motus_dataset_params"] = {
        "repo_id": target_root.name,
        "root": str(target_root),
        "embodiment_type": embodiment_type,
        "state_action_space": state_action_space,
        "state_action_arms": state_action_arms,
        "enable_t5_fallback": False,
        "t5_folder_name": t5_folder_name,
        "video_backend": "pyav",
    }
    return result


def write_manifest(output_root: Path, results: list[dict[str, Any]]) -> None:
    manifest_path = output_root / "motus_lerobot_manifest.json"
    stat_patch_path = output_root / "motus_stat_patch.json"
    output_root.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump({"datasets": results}, file_obj, ensure_ascii=False, indent=2)

    stat_patch = {
        result["embodiment_type"]: result["stat_entry"]
        for result in results
        if result.get("stat_entry") is not None and result.get("embodiment_type") is not None
    }
    with open(stat_patch_path, "w", encoding="utf-8") as file_obj:
        json.dump(stat_patch, file_obj, ensure_ascii=False, indent=2)


def update_motus_stat_json(stat_json_path: Path, results: list[dict[str, Any]]) -> None:
    stat_json_path.parent.mkdir(parents=True, exist_ok=True)
    if stat_json_path.exists():
        with open(stat_json_path, "r", encoding="utf-8") as file_obj:
            stats = json.load(file_obj)
    else:
        stats = {}

    for result in results:
        if result.get("stat_entry") is not None and result.get("embodiment_type") is not None:
            stats[result["embodiment_type"]] = result["stat_entry"]

    temp_path = stat_json_path.with_suffix(stat_json_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file_obj:
        json.dump(stats, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n")
    temp_path.replace(stat_json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Motus-facing mirrors for LeRobot datasets without modifying the source datasets. "
            "The input may be one dataset root or a parent directory containing many datasets."
        )
    )
    parser.add_argument("input", type=Path, help="LeRobot dataset root, or a parent directory to scan recursively.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Where Motus mirror datasets will be created. Defaults to the source parent directory; "
            "each mirror is named Motus_<source_folder_name>."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Refresh copied metadata and copied cache files.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan; do not write anything.")
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Mirror datasets even if they do not match the current Piper 32d Motus schema.",
    )
    parser.add_argument("--state-action-space", type=str, default="joints", help="Motus Piper state/action preset.")
    parser.add_argument("--state-action-arms", type=str, default="dual", help="Motus Piper arm preset.")
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
        help="Base path containing Wan2.2-TI2V-5B/. Required only when a mirror has missing T5 cache.",
    )
    parser.add_argument("--device", type=str, default=None, help="T5 device, e.g. cuda:0 or cpu.")
    parser.add_argument("--t5-folder-name", type=str, default="t5_embedding", help="T5 cache folder in each mirror.")
    parser.add_argument("--t5-text-len", type=int, default=512, help="WAN T5 text length.")
    parser.add_argument(
        "--motus-t5-source-root",
        type=Path,
        default=None,
        help=(
            "Optional existing Motus mirror root containing precomputed t5_embedding caches. "
            "Matching Motus_<dataset_name> caches are copied before generating missing files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = (args.output_root.resolve() if args.output_root is not None else default_output_root_for_input(input_path))
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
            allow_incompatible=args.allow_incompatible,
            dry_run=args.dry_run,
            state_action_space=args.state_action_space,
            state_action_arms=args.state_action_arms,
            t5_folder_name=args.t5_folder_name,
            motus_t5_source_root=(args.motus_t5_source_root.resolve() if args.motus_t5_source_root is not None else None),
            wan_path=args.wan_path,
            device=args.device,
            t5_text_len=args.t5_text_len,
            stat_json_path=args.stat_json_path.resolve(),
        )
        results.append(result)
        print(f"{result['status']}: {source_root} -> {target_root} ({result['compatibility_reason']})")

    if not args.dry_run:
        write_manifest(output_root, results)
        print(f"Wrote manifest: {output_root / 'motus_lerobot_manifest.json'}")
        print(f"Wrote stats patch: {output_root / 'motus_stat_patch.json'}")
        update_motus_stat_json(args.stat_json_path.resolve(), results)
        print(f"Updated Motus stats: {args.stat_json_path.resolve()}")


if __name__ == "__main__":
    main()
