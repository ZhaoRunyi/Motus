#!/usr/bin/env python3
"""Stage video-backed Motus Piper datasets without replacing existing mirrors.

The current Motus Piper mirrors are lightweight LeRobot mirrors named like:

    Motus_Piper_click_bell_0403

Some of them point at image-backed source datasets, where each parquet row
contains three image columns. This script creates a staged video-backed copy
that matches the structure used by Motus_Piper_pour_dual_0427:

    Piper_click_bell_0403_video/          # new source dataset with videos
      data/.../episode_XXXXXX.parquet     # image columns removed
      videos/.../{video_key}/episode.mp4  # one mp4 per camera per episode

    MotusVideo_Piper_click_bell_0403/     # new Motus mirror
      meta/info.json                      # points to Piper_*_video data/videos
      t5_embedding/...                    # hardlinked/copied from old mirror

Existing Motus_* directories are never modified by default.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


MOTUS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from lerobot.datasets.video_utils import encode_video_frames  # noqa: E402


IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

DEFAULT_LATEST_MOTUS_TASKS = (
    "Motus_Piper_beaker_mixer_0421",
    "Motus_Piper_carry_basket_0426",
    "Motus_Piper_click_bell_0403",
    "Motus_Piper_depress_pipette_0421",
    "Motus_Piper_dock_tubes_0421",
    "Motus_Piper_insert_test_tube_0430",
    "Motus_Piper_items_hand_over_place_0421",
    "Motus_Piper_open_drawer_0421",
    "Motus_Piper_open_pan_0421",
    "Motus_Piper_pour_dual_0427",
    "Motus_Piper_rearr_0421",
)

ROOT_FILES_TO_COPY = ("README.md", ".gitattributes", "episode_mapping.json")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n")


def load_jsonlines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonlines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_episode_path(path_template: str, episode_index: int, chunks_size: int) -> str:
    episode_chunk = episode_index // chunks_size
    return path_template.format(episode_chunk=episode_chunk, episode_index=episode_index)


def codec_name_for_info(vcodec: str) -> str:
    if vcodec == "libsvtav1":
        return "av1"
    return vcodec


def image_feature_to_video(feature: dict[str, Any], *, fps: int, vcodec: str, pix_fmt: str) -> dict[str, Any]:
    shape = list(feature.get("shape", [3, 480, 640]))
    if len(shape) != 3:
        raise ValueError(f"Expected CHW image shape, got {shape}")
    return {
        "dtype": "video",
        "shape": shape,
        "names": list(feature.get("names", ["channels", "height", "width"])),
        "info": {
            "video.height": int(shape[1]),
            "video.width": int(shape[2]),
            "video.codec": codec_name_for_info(vcodec),
            "video.pix_fmt": pix_fmt,
            "video.is_depth_map": False,
            "video.fps": int(fps),
            "video.channels": int(shape[0]),
            "has_audio": False,
        },
    }


def update_info_for_video_backend(
    info: dict[str, Any],
    *,
    data_path: str,
    video_path: str,
    fps: int,
    vcodec: str,
    pix_fmt: str,
) -> dict[str, Any]:
    updated = dict(info)
    updated["data_path"] = data_path
    updated["video_path"] = video_path
    updated["fps"] = int(fps)
    features = dict(updated.get("features", {}))
    for key in IMAGE_KEYS:
        if key not in features:
            raise KeyError(f"Missing image feature {key} in meta/info.json")
        features[key] = image_feature_to_video(features[key], fps=fps, vcodec=vcodec, pix_fmt=pix_fmt)
    updated["features"] = features
    if isinstance(updated.get("total_episodes"), int):
        updated["total_videos"] = int(updated["total_episodes"]) * len(IMAGE_KEYS)
    return updated


def restrict_metadata_to_episodes(dataset_root: Path, episode_indices: list[int]) -> None:
    """Make a staged debug subset self-consistent when --max-episodes is used."""
    wanted = set(int(index) for index in episode_indices)
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = [row for row in load_jsonlines(episodes_path) if int(row["episode_index"]) in wanted]
    write_jsonlines(episodes_path, episodes)

    episodes_stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if episodes_stats_path.exists():
        rows = [
            row
            for row in load_jsonlines(episodes_stats_path)
            if int(row.get("episode_index", -1)) in wanted
        ]
        write_jsonlines(episodes_stats_path, rows)

    valid_path = dataset_root / "meta" / "motus_valid_episodes.json"
    if valid_path.exists():
        valid = load_json(valid_path)
        valid["valid_episode_indices"] = [
            int(index)
            for index in valid.get("valid_episode_indices", [])
            if int(index) in wanted
        ]
        invalid = valid.get("invalid_episodes", [])
        if isinstance(invalid, list):
            valid["invalid_episodes"] = [
                row for row in invalid if int(row.get("episode_index", -1)) in wanted
            ]
        write_json(valid_path, valid)

    info_path = dataset_root / "meta" / "info.json"
    info = load_json(info_path)
    info["total_episodes"] = len(episodes)
    if episodes and all("length" in row for row in episodes):
        info["total_frames"] = int(sum(int(row["length"]) for row in episodes))
    if info.get("video_path"):
        info["total_videos"] = len(episodes) * len(IMAGE_KEYS)
    info["splits"] = {"train": f"0:{len(episodes)}"}
    write_json(info_path, info)


def resolve_source_root_from_mirror(mirror_root: Path, info: dict[str, Any]) -> Path:
    data_path = info.get("data_path")
    if not isinstance(data_path, str) or "/data/" not in data_path:
        if not mirror_root.name.startswith("Motus_"):
            raise ValueError(f"Cannot infer source root for {mirror_root}")
        return mirror_root.parent / mirror_root.name.removeprefix("Motus_")
    prefix = data_path.split("/data/", 1)[0]
    return (mirror_root / prefix).resolve()


def copy_file_if_exists(source: Path, target: Path, *, overwrite: bool) -> None:
    if not source.exists():
        return
    if target.exists() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_directory(
    source: Path,
    target: Path,
    *,
    overwrite: bool,
    hardlink_files: bool = False,
) -> None:
    if not source.exists():
        return
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        rel = source_file.relative_to(source)
        target_file = target / rel
        if target_file.exists() and not overwrite:
            continue
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if target_file.exists():
            target_file.unlink()
        if hardlink_files:
            try:
                os.link(source_file, target_file)
                continue
            except OSError:
                pass
        shutil.copy2(source_file, target_file)


def copy_metadata_and_small_files(source_root: Path, target_root: Path, *, overwrite: bool) -> None:
    copy_directory(source_root / "meta", target_root / "meta", overwrite=overwrite)
    for filename in ROOT_FILES_TO_COPY:
        copy_file_if_exists(source_root / filename, target_root / filename, overwrite=overwrite)


def image_cell_to_pil(cell: Any, dataset_root: Path) -> Image.Image:
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = cell.get("path")
        if path:
            return Image.open(dataset_root / str(path)).convert("RGB")
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    raise TypeError(f"Unsupported image cell type: {type(cell)!r}")


def save_column_as_video(
    image_cells: list[Any],
    *,
    source_root: Path,
    video_path: Path,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    crf: int,
    overwrite: bool,
) -> None:
    if video_path.exists() and not overwrite:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="motus_video_frames_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for frame_idx, cell in enumerate(image_cells):
            image = image_cell_to_pil(cell, source_root)
            image.save(tmp_path / f"frame_{frame_idx:06d}.png")
        encode_video_frames(
            imgs_dir=tmp_path,
            video_path=video_path,
            fps=int(fps),
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            crf=int(crf),
            overwrite=True,
        )


def convert_episode_to_video_backend(
    *,
    source_root: str,
    target_source_root: str,
    source_data_template: str,
    target_data_template: str,
    target_video_template: str,
    chunks_size: int,
    episode_index: int,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    crf: int,
    overwrite: bool,
) -> dict[str, Any]:
    source_root_path = Path(source_root)
    target_source_root_path = Path(target_source_root)
    source_parquet = source_root_path / format_episode_path(source_data_template, episode_index, chunks_size)
    target_parquet = target_source_root_path / format_episode_path(target_data_template, episode_index, chunks_size)
    table = pq.read_table(source_parquet)

    missing = [key for key in IMAGE_KEYS if key not in table.column_names]
    if missing:
        raise KeyError(f"{source_parquet} is missing image columns: {missing}")

    for key in IMAGE_KEYS:
        video_rel = target_video_template.format(
            episode_chunk=episode_index // chunks_size,
            video_key=key,
            episode_index=episode_index,
        )
        video_path = target_source_root_path / video_rel
        save_column_as_video(
            table.column(key).to_pylist(),
            source_root=source_root_path,
            video_path=video_path,
            fps=fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            crf=crf,
            overwrite=overwrite,
        )

    if overwrite or not target_parquet.exists():
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        stripped = table.drop([key for key in IMAGE_KEYS if key in table.column_names])
        pq.write_table(stripped.replace_schema_metadata(None), target_parquet)

    return {
        "episode_index": episode_index,
        "source_parquet": str(source_parquet),
        "target_parquet": str(target_parquet),
    }


def source_is_video_backed(info: dict[str, Any]) -> bool:
    if not info.get("video_path"):
        return False
    features = info.get("features", {})
    return all(features.get(key, {}).get("dtype") == "video" for key in IMAGE_KEYS)


def video_codec_from_info(info: dict[str, Any], default: str) -> str:
    feature = info.get("features", {}).get(IMAGE_KEYS[0], {})
    codec = feature.get("info", {}).get("video.codec")
    return str(codec) if codec else default


def video_pix_fmt_from_info(info: dict[str, Any], default: str) -> str:
    feature = info.get("features", {}).get(IMAGE_KEYS[0], {})
    pix_fmt = feature.get("info", {}).get("video.pix_fmt")
    return str(pix_fmt) if pix_fmt else default


def collect_episode_indices(meta_root: Path, max_episodes: int | None) -> list[int]:
    episodes = load_jsonlines(meta_root / "episodes.jsonl")
    indices = [int(row["episode_index"]) for row in episodes]
    if max_episodes is not None:
        indices = indices[:max_episodes]
    return indices


def stage_mirror(
    *,
    old_mirror_root: Path,
    new_mirror_root: Path,
    data_source_root: Path,
    source_info: dict[str, Any],
    fps: int,
    vcodec: str,
    pix_fmt: str,
    overwrite: bool,
    link_t5: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    copy_metadata_and_small_files(old_mirror_root, new_mirror_root, overwrite=overwrite)
    copy_directory(
        old_mirror_root / "t5_embedding",
        new_mirror_root / "t5_embedding",
        overwrite=overwrite,
        hardlink_files=link_t5,
    )

    info_path = new_mirror_root / "meta" / "info.json"
    base_info = load_json(info_path)
    rel_data_template = os.path.relpath(
        data_source_root / "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        start=new_mirror_root,
    )
    rel_video_template = os.path.relpath(
        data_source_root
        / "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        start=new_mirror_root,
    )
    updated = update_info_for_video_backend(
        base_info,
        data_path=rel_data_template,
        video_path=rel_video_template,
        fps=fps,
        vcodec=vcodec,
        pix_fmt=pix_fmt,
    )
    write_json(info_path, updated)


def stage_one_dataset(args: argparse.Namespace, old_mirror_root: Path) -> dict[str, Any]:
    old_info = load_json(old_mirror_root / "meta" / "info.json")
    source_root = resolve_source_root_from_mirror(old_mirror_root, old_info)
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found for {old_mirror_root}: {source_root}")

    source_info = load_json(source_root / "meta" / "info.json")
    output_root = Path(args.output_root).resolve()
    new_mirror_root = output_root / f"{args.mirror_prefix}{source_root.name}"
    target_source_root = output_root / f"{source_root.name}{args.source_suffix}"

    fps = int(args.fps or source_info.get("fps") or old_info.get("fps") or 10)
    chunks_size = int(source_info.get("chunks_size", 1000))
    source_data_template = str(source_info.get("data_path", old_info["data_path"]))
    target_data_template = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    target_video_template = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    already_video = source_is_video_backed(source_info)
    data_source_root = source_root if already_video else target_source_root

    result: dict[str, Any] = {
        "old_mirror_root": str(old_mirror_root),
        "source_root": str(source_root),
        "new_mirror_root": str(new_mirror_root),
        "target_source_root": str(data_source_root),
        "already_video_backed": already_video,
        "converted_episodes": 0,
    }

    print(f"\n==> {old_mirror_root.name}")
    print(f"    source: {source_root}")
    print(f"    staged mirror: {new_mirror_root}")
    if already_video:
        print("    source is already video-backed; only staging a new mirror")
    else:
        print(f"    staged source: {target_source_root}")

    if args.dry_run:
        result["status"] = "dry_run"
        return result

    if new_mirror_root.exists() and not args.overwrite:
        raise FileExistsError(f"Staged mirror exists. Use --overwrite to update: {new_mirror_root}")
    if target_source_root.exists() and not args.overwrite and not already_video:
        raise FileExistsError(f"Staged source exists. Use --overwrite to update: {target_source_root}")

    if not already_video:
        copy_metadata_and_small_files(source_root, target_source_root, overwrite=args.overwrite)
        target_info = update_info_for_video_backend(
            source_info,
            data_path=target_data_template,
            video_path=target_video_template,
            fps=fps,
            vcodec=args.vcodec,
            pix_fmt=args.pix_fmt,
        )
        write_json(target_source_root / "meta" / "info.json", target_info)

        episode_indices = collect_episode_indices(source_root / "meta", args.max_episodes)
        if args.max_episodes is not None:
            restrict_metadata_to_episodes(target_source_root, episode_indices)
        jobs = [
            {
                "source_root": str(source_root),
                "target_source_root": str(target_source_root),
                "source_data_template": source_data_template,
                "target_data_template": target_data_template,
                "target_video_template": target_video_template,
                "chunks_size": chunks_size,
                "episode_index": episode_index,
                "fps": fps,
                "vcodec": args.vcodec,
                "pix_fmt": args.pix_fmt,
                "crf": args.crf,
                "overwrite": args.overwrite,
            }
            for episode_index in episode_indices
        ]

        if args.num_workers <= 1:
            for job in jobs:
                convert_episode_to_video_backend(**job)
                result["converted_episodes"] += 1
                print(f"    converted episode {job['episode_index']:06d}")
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(convert_episode_to_video_backend, **job) for job in jobs]
                for future in as_completed(futures):
                    item = future.result()
                    result["converted_episodes"] += 1
                print(f"    converted episode {int(item['episode_index']):06d}")
    else:
        episode_indices = collect_episode_indices(source_root / "meta", args.max_episodes)

    stage_mirror(
        old_mirror_root=old_mirror_root,
        new_mirror_root=new_mirror_root,
        data_source_root=data_source_root,
        source_info=source_info,
        fps=fps,
        vcodec=args.vcodec if not already_video else video_codec_from_info(source_info, args.vcodec),
        pix_fmt=args.pix_fmt if not already_video else video_pix_fmt_from_info(source_info, args.pix_fmt),
        overwrite=args.overwrite,
        link_t5=args.link_t5,
        dry_run=False,
    )
    if args.max_episodes is not None:
        restrict_metadata_to_episodes(new_mirror_root, episode_indices)

    result["status"] = "staged"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage video-backed Motus Piper mirrors while preserving existing Motus_* directories."
    )
    parser.add_argument("--root", type=Path, default=Path("/aoss/data/ZhaoRunyi"), help="Root containing Motus_* mirrors and Piper_* sources.")
    parser.add_argument("--output-root", type=Path, default=None, help="Where staged datasets are written. Defaults to --root.")
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_LATEST_MOTUS_TASKS), help="Motus mirror directory names to stage.")
    parser.add_argument("--mirror-prefix", default="MotusVideo_", help="Prefix for staged Motus mirror directories.")
    parser.add_argument("--source-suffix", default="_video", help="Suffix for converted source dataset directories.")
    parser.add_argument("--vcodec", default="libsvtav1", choices=["libsvtav1", "h264", "hevc"], help="Video codec for newly encoded datasets.")
    parser.add_argument("--pix-fmt", default="yuv420p", help="Pixel format for encoded videos.")
    parser.add_argument("--crf", type=int, default=30, help="Encoder CRF. Lower is higher quality/larger.")
    parser.add_argument("--fps", type=int, default=None, help="Override dataset fps. Default: read from meta/info.json.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug limit per dataset.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel episode conversions per dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing staged outputs.")
    parser.add_argument("--no-link-t5", dest="link_t5", action="store_false", help="Copy T5 cache files instead of hardlinking when possible.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing files.")
    parser.set_defaults(link_t5=True)
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = args.root
    return args


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output_root = args.output_root.resolve()

    manifest: list[dict[str, Any]] = []
    for task_name in args.tasks:
        old_mirror_root = args.root / task_name
        if not old_mirror_root.exists():
            raise FileNotFoundError(f"Motus mirror not found: {old_mirror_root}")
        manifest.append(stage_one_dataset(args, old_mirror_root))

    if not args.dry_run:
        manifest_path = args.output_root / "motus_piper_video_staging_manifest.json"
        write_json(manifest_path, {"datasets": manifest})
        print(f"\nWrote manifest: {manifest_path}")

    print("\nStaged mirror names:")
    for item in manifest:
        print(f"  - {Path(item['new_mirror_root']).name}")


if __name__ == "__main__":
    main()
