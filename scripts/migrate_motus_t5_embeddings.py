#!/usr/bin/env python3
"""Copy precomputed T5 caches from existing Motus mirrors into local Motus mirrors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


MOTUS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTUS_ROOT))

from scripts.prepare_motus_lerobot_datasets import discover_motus_mirror_datasets  # noqa: E402
from scripts.prepare_motus_lerobot_datasets import patch_existing_t5_pointers  # noqa: E402
from scripts.prepare_motus_lerobot_datasets import seed_t5_cache_from_matching_motus_dataset  # noqa: E402
from scripts.prepare_motus_lerobot_datasets import t5_cache_is_complete  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan one Motus mirror dataset or a parent directory containing many Motus_* datasets, "
            "then copy matching t5_embedding caches from another Motus root."
        )
    )
    parser.add_argument("input", type=Path, help="Motus dataset root, or a parent directory to scan recursively.")
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Existing Motus mirror root containing precomputed t5_embedding caches, e.g. /aoss/data/ZhaoRunyi.",
    )
    parser.add_argument("--t5-folder-name", type=str, default="t5_embedding", help="T5 cache folder name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local .pt files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    source_root = args.source_root.resolve()

    dataset_roots = discover_motus_mirror_datasets(input_path)
    if not dataset_roots:
        raise FileNotFoundError(f"No Motus_* dataset roots found under {input_path}")

    for dataset_root in dataset_roots:
        seed_result = seed_t5_cache_from_matching_motus_dataset(
            dataset_root,
            motus_t5_source_root=source_root,
            t5_folder_name=args.t5_folder_name,
            overwrite=args.overwrite,
        )
        pointer_updates = patch_existing_t5_pointers(dataset_root, args.t5_folder_name)
        complete = t5_cache_is_complete(dataset_root, args.t5_folder_name)
        print(
            f"{seed_result['status']}: {dataset_root} "
            f"(source={seed_result['source_root']}, copied={seed_result['copied_files']}, "
            f"skipped={seed_result['skipped_existing_files']}, pointers_fixed={pointer_updates}, complete={complete})"
        )


if __name__ == "__main__":
    main()
