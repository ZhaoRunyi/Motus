import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_motus_lerobot_datasets import (  # noqa: E402
    default_output_root_for_input,
    discover_lerobot_datasets,
    discover_motus_mirror_datasets,
    prepare_dataset,
    seed_t5_cache_from_matching_motus_dataset,
    target_root_for_dataset,
    update_motus_stat_json,
    write_manifest,
)
from data.lerobot.repair_parquet_hf_metadata import repair_dataset_parquet_hf_metadata  # noqa: E402


def write_jsonlines(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


class PrepareMotusLeRobotDatasetsTest(unittest.TestCase):
    def test_prepare_mirror_keeps_source_readonly_and_computes_14d_stats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_parent = tmp_path / "source"
            source_root = source_parent / "Piper_unit_task_0424"
            meta_root = source_root / "meta"
            data_root = source_root / "data" / "chunk-000"
            meta_root.mkdir(parents=True)
            data_root.mkdir(parents=True)

            info = {
                "codebase_version": "v2.1",
                "robot_type": "aloha",
                "total_episodes": 2,
                "total_frames": 6,
                "total_tasks": 1,
                "total_videos": 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": None,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [32], "names": None},
                    "action": {"dtype": "float32", "shape": [32], "names": None},
                    "observation.images.cam_high": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "observation.images.cam_left_wrist": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "observation.images.cam_right_wrist": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "task_index": {"dtype": "int64", "shape": [1], "names": None},
                },
            }
            (meta_root / "info.json").write_text(json.dumps(info), encoding="utf-8")
            write_jsonlines(meta_root / "tasks.jsonl", [{"task_index": 0, "task": "Unit test task"}])
            write_jsonlines(
                meta_root / "episodes.jsonl",
                [
                    {"episode_index": 0, "length": 3, "t5_embedding_path": "t5_embedding/episode_000000.pt"},
                    {"episode_index": 1, "length": 3, "t5_embedding_path": "t5_embedding/episode_000001.pt"},
                ],
            )
            t5_root = source_root / "t5_embedding"
            t5_root.mkdir(parents=True)
            (t5_root / "episode_000000.pt").write_bytes(b"dummy t5 cache")
            (t5_root / "episode_000001.pt").write_bytes(b"dummy t5 cache")
            source_episodes_before = (meta_root / "episodes.jsonl").read_text(encoding="utf-8")

            states = np.stack([np.arange(32, dtype=np.float32) + offset for offset in [0, 1, 2]])
            actions = np.stack([np.arange(32, dtype=np.float32) + offset for offset in [10, 20, 30]])
            table = pa.table(
                {
                    "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                    "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                    "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
                    "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                    "episode_index": pa.array([0, 0, 0], type=pa.int64()),
                    "index": pa.array([0, 1, 2], type=pa.int64()),
                    "task_index": pa.array([0, 0, 0], type=pa.int64()),
                }
            )
            pq.write_table(table, data_root / "episode_000000.parquet")
            (data_root / "episode_000001.parquet").write_bytes(b"not a parquet file")

            output_root = tmp_path / "motus"
            discovered = discover_lerobot_datasets(source_parent)
            self.assertEqual(discovered, [source_root.resolve()])
            self.assertEqual(default_output_root_for_input(source_parent), source_parent.resolve())

            target_root = target_root_for_dataset(source_root, source_parent, output_root)
            self.assertEqual(target_root, output_root / "Motus_Piper_unit_task_0424")
            stat_json_path = tmp_path / "stat.json"
            result = prepare_dataset(
                source_root=source_root,
                target_root=target_root,
                overwrite=False,
                allow_incompatible=False,
                dry_run=False,
                state_action_space="joints",
                state_action_arms="dual",
                t5_folder_name="t5_embedding",
                motus_t5_source_root=None,
                repair_parquet_hf_metadata=False,
                wan_path=None,
                device=None,
                t5_text_len=512,
                stat_json_path=stat_json_path,
            )
            write_manifest(output_root, [result])

            self.assertEqual((meta_root / "episodes.jsonl").read_text(encoding="utf-8"), source_episodes_before)
            target_info = json.loads((target_root / "meta" / "info.json").read_text(encoding="utf-8"))
            expected_data_path = os.path.relpath(
                source_root / "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                start=target_root,
            )
            self.assertEqual(target_info["data_path"], expected_data_path)
            self.assertFalse((target_root / "data").exists())

            target_episodes = [
                json.loads(line)
                for line in (target_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(target_episodes[0]["tasks"], ["Unit test task"])

            self.assertEqual(result["embodiment_type"], "piper_unit_task_0424_dual_14d")
            self.assertEqual(result["stat_entry"]["action_dim"], 14)
            self.assertEqual(result["stat_entry"]["file_count"], 1)
            self.assertEqual(result["stat_entry"]["total_files_scanned"], 2)
            self.assertEqual(result["stat_entry"]["invalid_file_count"], 1)
            self.assertEqual(result["stat_entry"]["invalid_episodes"][0]["episode_index"], 1)
            self.assertEqual(result["stat_entry"]["valid_episode_indices"], [0])
            expected_indices = list(range(0, 7)) + list(range(16, 23))
            np.testing.assert_allclose(result["stat_entry"]["min"], actions[:, expected_indices].min(axis=0))
            np.testing.assert_allclose(result["stat_entry"]["max"], actions[:, expected_indices].max(axis=0))
            sidecar = json.loads((target_root / "meta" / "motus_valid_episodes.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["valid_episode_indices"], [0])
            self.assertEqual(sidecar["invalid_episodes"][0]["episode_index"], 1)
            self.assertTrue((output_root / "motus_lerobot_manifest.json").exists())
            self.assertTrue((output_root / "motus_stat_patch.json").exists())
            self.assertFalse((target_root / "t5_embedding").is_symlink())
            self.assertTrue((target_root / "t5_embedding" / "episode_000000.pt").exists())

            update_motus_stat_json(stat_json_path, [result])
            merged_stats = json.loads(stat_json_path.read_text(encoding="utf-8"))
            self.assertIn("piper_unit_task_0424_dual_14d", merged_stats)

            complete_result = prepare_dataset(
                source_root=source_root,
                target_root=target_root,
                overwrite=False,
                allow_incompatible=False,
                dry_run=False,
                state_action_space="joints",
                state_action_arms="dual",
                t5_folder_name="t5_embedding",
                motus_t5_source_root=None,
                repair_parquet_hf_metadata=False,
                wan_path=None,
                device=None,
                t5_text_len=512,
                stat_json_path=stat_json_path,
            )
            self.assertEqual(complete_result["status"], "complete")

    def test_prepare_dataset_can_seed_t5_from_existing_motus_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_parent = tmp_path / "source"
            source_root = source_parent / "Piper_click_bell_0403"
            meta_root = source_root / "meta"
            data_root = source_root / "data" / "chunk-000"
            meta_root.mkdir(parents=True)
            data_root.mkdir(parents=True)

            info = {
                "codebase_version": "v2.1",
                "robot_type": "aloha",
                "total_episodes": 2,
                "total_frames": 6,
                "total_tasks": 1,
                "total_videos": 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": None,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [32], "names": None},
                    "action": {"dtype": "float32", "shape": [32], "names": None},
                    "observation.images.cam_high": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "observation.images.cam_left_wrist": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "observation.images.cam_right_wrist": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "task_index": {"dtype": "int64", "shape": [1], "names": None},
                },
            }
            (meta_root / "info.json").write_text(json.dumps(info), encoding="utf-8")
            write_jsonlines(meta_root / "tasks.jsonl", [{"task_index": 0, "task": "Click the bell"}])
            write_jsonlines(
                meta_root / "episodes.jsonl",
                [
                    {"episode_index": 0, "length": 3},
                    {"episode_index": 1, "length": 3},
                ],
            )

            for episode_index in range(2):
                actions = np.stack(
                    [np.arange(32, dtype=np.float32) + offset for offset in [10 + episode_index, 20 + episode_index, 30 + episode_index]]
                )
                states = np.stack(
                    [np.arange(32, dtype=np.float32) + offset for offset in [0 + episode_index, 1 + episode_index, 2 + episode_index]]
                )
                table = pa.table(
                    {
                        "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                        "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                        "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
                        "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                        "episode_index": pa.array([episode_index, episode_index, episode_index], type=pa.int64()),
                        "index": pa.array([0, 1, 2], type=pa.int64()),
                        "task_index": pa.array([0, 0, 0], type=pa.int64()),
                    }
                )
                pq.write_table(table, data_root / f"episode_{episode_index:06d}.parquet")

            existing_motus_root = tmp_path / "existing"
            existing_dataset_root = existing_motus_root / "Motus_Piper_click_bell_0403"
            existing_meta_root = existing_dataset_root / "meta"
            existing_t5_root = existing_dataset_root / "t5_embedding"
            existing_meta_root.mkdir(parents=True)
            existing_t5_root.mkdir(parents=True)
            (existing_meta_root / "info.json").write_text(json.dumps(info), encoding="utf-8")
            write_jsonlines(
                existing_meta_root / "tasks.jsonl",
                [{"task_index": 0, "task": "Click the bell"}],
            )
            write_jsonlines(
                existing_meta_root / "episodes.jsonl",
                [
                    {"episode_index": 0, "length": 3, "t5_embedding_path": "t5_embedding/episode_000000.pt"},
                    {"episode_index": 1, "length": 3, "t5_embedding_path": "t5_embedding/episode_000001.pt"},
                ],
            )
            (existing_t5_root / "episode_000000.pt").write_bytes(b"seed0")
            (existing_t5_root / "episode_000001.pt").write_bytes(b"seed1")

            discovered = discover_motus_mirror_datasets(existing_motus_root)
            self.assertEqual(discovered, [existing_dataset_root.resolve()])

            target_root = tmp_path / "prepared" / "Motus_Piper_click_bell_0403"
            stat_json_path = tmp_path / "stat.json"

            with mock.patch("scripts.prepare_motus_lerobot_datasets.generate_t5_cache") as generate_t5_cache:
                result = prepare_dataset(
                    source_root=source_root,
                    target_root=target_root,
                    overwrite=False,
                    allow_incompatible=False,
                    dry_run=False,
                    state_action_space="joints",
                    state_action_arms="dual",
                    t5_folder_name="t5_embedding",
                    motus_t5_source_root=existing_motus_root,
                    repair_parquet_hf_metadata=False,
                    wan_path=None,
                    device=None,
                    t5_text_len=512,
                    stat_json_path=stat_json_path,
                )

            generate_t5_cache.assert_not_called()
            self.assertEqual(result["t5_seed"]["status"], "seeded")
            self.assertEqual(result["t5_seed"]["copied_files"], 2)
            self.assertEqual(result["t5_status"], "pointers_fixed")
            self.assertEqual((target_root / "t5_embedding" / "episode_000000.pt").read_bytes(), b"seed0")
            self.assertEqual((target_root / "t5_embedding" / "episode_000001.pt").read_bytes(), b"seed1")
            target_episodes = [
                json.loads(line)
                for line in (target_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(target_episodes[0]["t5_embedding_path"], "t5_embedding/episode_000000.pt")
            self.assertEqual(target_episodes[1]["t5_embedding_path"], "t5_embedding/episode_000001.pt")

    def test_seed_t5_cache_skips_existing_files_when_not_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_root = tmp_path / "prepared" / "Motus_Piper_click_bell_0403"
            target_meta = target_root / "meta"
            target_t5 = target_root / "t5_embedding"
            target_meta.mkdir(parents=True)
            target_t5.mkdir(parents=True)
            info = {
                "features": {},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": None,
            }
            (target_meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
            write_jsonlines(target_meta / "episodes.jsonl", [{"episode_index": 0, "length": 1}])
            write_jsonlines(target_meta / "tasks.jsonl", [{"task_index": 0, "task": "Click the bell"}])
            (target_t5 / "episode_000000.pt").write_bytes(b"local")

            source_root = tmp_path / "existing" / "Motus_Piper_click_bell_0403"
            source_meta = source_root / "meta"
            source_t5 = source_root / "t5_embedding"
            source_meta.mkdir(parents=True)
            source_t5.mkdir(parents=True)
            (source_meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
            write_jsonlines(source_meta / "episodes.jsonl", [{"episode_index": 0, "length": 1}])
            write_jsonlines(source_meta / "tasks.jsonl", [{"task_index": 0, "task": "Click the bell"}])
            (source_t5 / "episode_000000.pt").write_bytes(b"remote")

            result = seed_t5_cache_from_matching_motus_dataset(
                target_root,
                motus_t5_source_root=tmp_path / "existing",
                t5_folder_name="t5_embedding",
                overwrite=False,
            )

            self.assertEqual(result["status"], "already_present")
            self.assertEqual(result["copied_files"], 0)
            self.assertEqual(result["skipped_existing_files"], 1)
            self.assertEqual((target_t5 / "episode_000000.pt").read_bytes(), b"local")

    def test_repair_dataset_parquet_hf_metadata_strips_incompatible_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "Piper_bad_metadata"
            data_root = dataset_root / "data" / "chunk-000"
            data_root.mkdir(parents=True)

            states = np.stack([np.arange(32, dtype=np.float32) + offset for offset in [0, 1, 2]])
            actions = np.stack([np.arange(32, dtype=np.float32) + offset for offset in [10, 20, 30]])
            table = pa.table(
                {
                    "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                    "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                    "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
                    "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                    "episode_index": pa.array([0, 0, 0], type=pa.int64()),
                    "index": pa.array([0, 1, 2], type=pa.int64()),
                    "task_index": pa.array([0, 0, 0], type=pa.int64()),
                }
            )
            bad_hf_metadata = {
                "info": {
                    "features": {
                        "observation.state": {
                            "feature": {"dtype": "float32", "_type": "Value"},
                            "length": 32,
                            "_type": "List",
                        },
                        "action": {
                            "feature": {"dtype": "float32", "_type": "Value"},
                            "length": 32,
                            "_type": "List",
                        },
                        "timestamp": {"dtype": "float32", "_type": "Value"},
                        "frame_index": {"dtype": "int64", "_type": "Value"},
                        "episode_index": {"dtype": "int64", "_type": "Value"},
                        "index": {"dtype": "int64", "_type": "Value"},
                        "task_index": {"dtype": "int64", "_type": "Value"},
                    }
                }
            }
            parquet_path = data_root / "episode_000000.parquet"
            pq.write_table(
                table.replace_schema_metadata(
                    {b"huggingface": json.dumps(bad_hf_metadata, ensure_ascii=False).encode("utf-8")}
                ),
                parquet_path,
            )

            result = repair_dataset_parquet_hf_metadata(dataset_root)

            self.assertEqual(result["status"], "repaired")
            self.assertEqual(result["repaired_files"], 1)
            repaired_metadata = pq.read_schema(parquet_path).metadata or {}
            self.assertNotIn(b"huggingface", repaired_metadata)

    def test_prepare_dataset_can_repair_source_parquet_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_root = tmp_path / "source" / "Piper_repair_me"
            meta_root = source_root / "meta"
            data_root = source_root / "data" / "chunk-000"
            t5_root = source_root / "t5_embedding"
            meta_root.mkdir(parents=True)
            data_root.mkdir(parents=True)
            t5_root.mkdir(parents=True)

            info = {
                "codebase_version": "v2.1",
                "robot_type": "aloha",
                "total_episodes": 1,
                "total_frames": 3,
                "total_tasks": 1,
                "total_videos": 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": None,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [32], "names": None},
                    "action": {"dtype": "float32", "shape": [32], "names": None},
                    "observation.images.cam_high": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "observation.images.cam_left_wrist": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "observation.images.cam_right_wrist": {"dtype": "image", "shape": [3, 8, 8], "names": None},
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                    "index": {"dtype": "int64", "shape": [1], "names": None},
                    "task_index": {"dtype": "int64", "shape": [1], "names": None},
                },
            }
            (meta_root / "info.json").write_text(json.dumps(info), encoding="utf-8")
            write_jsonlines(meta_root / "tasks.jsonl", [{"task_index": 0, "task": "Repair metadata"}])
            write_jsonlines(
                meta_root / "episodes.jsonl",
                [{"episode_index": 0, "length": 3, "t5_embedding_path": "t5_embedding/episode_000000.pt"}],
            )
            (t5_root / "episode_000000.pt").write_bytes(b"seed0")

            states = np.stack([np.arange(32, dtype=np.float32) + offset for offset in [0, 1, 2]])
            actions = np.stack([np.arange(32, dtype=np.float32) + offset for offset in [10, 20, 30]])
            table = pa.table(
                {
                    "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                    "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), list_size=32)),
                    "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
                    "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                    "episode_index": pa.array([0, 0, 0], type=pa.int64()),
                    "index": pa.array([0, 1, 2], type=pa.int64()),
                    "task_index": pa.array([0, 0, 0], type=pa.int64()),
                }
            )
            bad_hf_metadata = {
                "info": {
                    "features": {
                        "observation.state": {
                            "feature": {"dtype": "float32", "_type": "Value"},
                            "length": 32,
                            "_type": "List",
                        },
                        "action": {
                            "feature": {"dtype": "float32", "_type": "Value"},
                            "length": 32,
                            "_type": "List",
                        },
                        "timestamp": {"dtype": "float32", "_type": "Value"},
                        "frame_index": {"dtype": "int64", "_type": "Value"},
                        "episode_index": {"dtype": "int64", "_type": "Value"},
                        "index": {"dtype": "int64", "_type": "Value"},
                        "task_index": {"dtype": "int64", "_type": "Value"},
                    }
                }
            }
            parquet_path = data_root / "episode_000000.parquet"
            pq.write_table(
                table.replace_schema_metadata(
                    {b"huggingface": json.dumps(bad_hf_metadata, ensure_ascii=False).encode("utf-8")}
                ),
                parquet_path,
            )

            result = prepare_dataset(
                source_root=source_root,
                target_root=tmp_path / "prepared" / "Motus_Piper_repair_me",
                overwrite=False,
                allow_incompatible=False,
                dry_run=False,
                state_action_space="joints",
                state_action_arms="dual",
                t5_folder_name="t5_embedding",
                motus_t5_source_root=None,
                repair_parquet_hf_metadata=True,
                wan_path=None,
                device=None,
                t5_text_len=512,
                stat_json_path=tmp_path / "stat.json",
            )

            self.assertEqual(result["parquet_hf_metadata_repair"]["status"], "repaired")
            self.assertEqual(result["parquet_hf_metadata_repair"]["repaired_files"], 1)
            self.assertNotIn(b"huggingface", pq.read_schema(parquet_path).metadata or {})


if __name__ == "__main__":
    unittest.main()
