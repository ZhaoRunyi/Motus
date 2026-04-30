#!/usr/bin/env python3
"""Collect Motus episode-level T5 embeddings into deploy prompt cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


MOTUS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from scripts.prepare_motus_lerobot_datasets import discover_motus_mirror_datasets  # noqa: E402
from scripts.prepare_motus_lerobot_datasets import load_jsonlines  # noqa: E402


def prompt_cache_filename(prompt: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", prompt).strip("_").lower()
    if not normalized:
        normalized = "prompt"
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{normalized[:80]}_{digest}.pt"


def episode_prompt(episode: dict[str, Any]) -> str | None:
    tasks = episode.get("tasks")
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], str) and tasks[0].strip():
        return tasks[0].strip()
    task = episode.get("task")
    if isinstance(task, str) and task.strip() and not task.strip().isdigit():
        return task.strip()
    return None


def collect_dataset_prompt_cache(
    dataset_root: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "copied": 0,
        "skipped_existing": 0,
        "skipped_missing_prompt": 0,
        "skipped_missing_embedding": 0,
        "prompts": [],
    }
    if not episodes_path.exists():
        return result

    seen_prompts: set[str] = set()
    for episode in load_jsonlines(episodes_path):
        prompt = episode_prompt(episode)
        if prompt is None:
            result["skipped_missing_prompt"] += 1
            continue
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        rel_embedding = episode.get("t5_embedding_path")
        if not isinstance(rel_embedding, str):
            result["skipped_missing_embedding"] += 1
            continue
        source_path = dataset_root / rel_embedding
        if not source_path.exists():
            result["skipped_missing_embedding"] += 1
            continue

        target_path = output_dir / prompt_cache_filename(prompt)
        prompt_result = {
            "prompt": prompt,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }
        if target_path.exists() and not overwrite:
            result["skipped_existing"] += 1
            prompt_result["status"] = "already_present"
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            result["copied"] += 1
            prompt_result["status"] = "copied"
        result["prompts"].append(prompt_result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect t5_embedding/*.pt from Motus_* datasets into Motus deploy prompt cache."
    )
    parser.add_argument("input", type=Path, help="Motus dataset root, or parent directory containing Motus_* datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MOTUS_ROOT / "t5_prompt_cache",
        help="Deploy prompt cache directory used by challenge_deploy/serve_policy.py.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prompt cache files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to <output-dir>/prompt_cache_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest is not None else output_dir / "prompt_cache_manifest.json"

    dataset_roots = discover_motus_mirror_datasets(input_path)
    if not dataset_roots:
        raise FileNotFoundError(f"No Motus_* dataset roots found under {input_path}")

    results = [
        collect_dataset_prompt_cache(dataset_root, output_dir, overwrite=args.overwrite)
        for dataset_root in dataset_roots
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump({"datasets": results}, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")

    copied = sum(int(result["copied"]) for result in results)
    skipped_existing = sum(int(result["skipped_existing"]) for result in results)
    print(
        f"Collected prompt T5 cache into {output_dir} "
        f"(copied={copied}, skipped_existing={skipped_existing}, manifest={manifest_path})"
    )


if __name__ == "__main__":
    main()
