import importlib
import io
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.lerobot.slai_piper_policy import StateSpaceConfig, get_space_indices, select_state_action_vector

PIPER_SPACE = StateSpaceConfig(ids="joints", arms="dual")
PIPER_14_INDICES = get_space_indices(PIPER_SPACE)


def _import_lerobot_dataset_module():
    for module_name in [
        "data.lerobot.lerobot_dataset",
        "lerobot",
        "lerobot.datasets",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.datasets.video_utils",
        "utils",
        "utils.vlm_utils",
    ]:
        sys.modules.pop(module_name, None)

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []  # mark as package
    vlm_utils_mod = types.ModuleType("utils.vlm_utils")
    vlm_utils_mod.preprocess_vlm_messages = lambda text, image, processor: {
        "text": text,
        "size": image.size,
    }

    lerobot_pkg = types.ModuleType("lerobot")
    datasets_pkg = types.ModuleType("lerobot.datasets")
    lerobot_dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    video_utils_mod = types.ModuleType("lerobot.datasets.video_utils")

    class DummyLeRobotDataset:
        pass

    class DummyLeRobotDatasetMetadata:
        pass

    class DummyMultiLeRobotDataset:
        pass

    class DummyVideoFrame:
        pass

    def _unexpected_decode(*args, **kwargs):
        raise AssertionError("decode_video_frames should be monkeypatched in this test")

    lerobot_dataset_mod.LeRobotDataset = DummyLeRobotDataset
    lerobot_dataset_mod.LeRobotDatasetMetadata = DummyLeRobotDatasetMetadata
    lerobot_dataset_mod.MultiLeRobotDataset = DummyMultiLeRobotDataset
    video_utils_mod.decode_video_frames = _unexpected_decode
    video_utils_mod.VideoFrame = DummyVideoFrame

    sys.modules["utils"] = utils_pkg
    sys.modules["utils.vlm_utils"] = vlm_utils_mod
    sys.modules["lerobot"] = lerobot_pkg
    sys.modules["lerobot.datasets"] = datasets_pkg
    sys.modules["lerobot.datasets.lerobot_dataset"] = lerobot_dataset_mod
    sys.modules["lerobot.datasets.video_utils"] = video_utils_mod

    return importlib.import_module("data.lerobot.lerobot_dataset")


def _png_bytes(color, size):
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeHFDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys())

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.rows[index]
        return {
            key: [self.rows[i][key] for i in index]
            for key in self.column_names
        }


class FakeMeta:
    def __init__(self, root, episodes, video_paths=None):
        self.root = str(root)
        self.episodes = episodes
        self._video_paths = video_paths or {}

    def get_video_file_path(self, episode_index, video_key):
        key = (int(episode_index), video_key)
        if key not in self._video_paths:
            raise KeyError(key)
        return self._video_paths[key]


class FakeDataset:
    def __init__(self, root, rows, episodes, video_paths=None):
        self.root = str(root)
        self.hf_dataset = FakeHFDataset(rows)
        self.meta = FakeMeta(root=root, episodes=episodes, video_paths=video_paths)
        self.episode_data_index = {"from": [0], "to": [len(rows)]}
        self.num_episodes = 1
        self.tolerance_s = 0.1
        self.video_backend = "pyav"


def _build_adapter(module, fake_dataset, *, has_concat=False, has_three_cam=True):
    adapter = module.LeRobotMotusDataset.__new__(module.LeRobotMotusDataset)
    adapter.task_mode = "single"
    adapter.task_name = None
    adapter.state_action_config = PIPER_SPACE
    adapter.global_downsample_rate = 1
    adapter.video_action_freq_ratio = 2
    adapter.num_video_frames = 2
    adapter.action_chunk_size = adapter.num_video_frames * adapter.video_action_freq_ratio
    adapter.video_size = (24, 24)
    adapter.has_concat = has_concat
    adapter.has_three_cam = has_three_cam
    adapter.single_view_candidates = ["observation.images.main", "observation.image", "image"]
    adapter.lerobot_dataset = fake_dataset
    adapter._episode_embedding_cache = {}
    adapter.enable_t5_fallback = False
    adapter.vlm_processor = None
    adapter.action_min = np.zeros(len(PIPER_14_INDICES), dtype=np.float32)
    adapter.action_max = np.full(len(PIPER_14_INDICES), 100.0, dtype=np.float32)
    adapter.task_prompts = {0: "Click the bell"}
    return adapter


def test_resolve_instruction_text_uses_episode_tasks_and_task_map():
    module = _import_lerobot_dataset_module()
    adapter = module.LeRobotMotusDataset.__new__(module.LeRobotMotusDataset)
    adapter.task_mode = "single"
    adapter.task_prompts = {0: "Click the bell"}

    assert (
        adapter.get_prompt(
            item_cond={"task_index": 0},
            episode_meta={"tasks": ["Click the bell"]},
        )
        == "Click the bell"
    )
    assert (
        adapter.get_prompt(
            item_cond={"task_index": 0},
            episode_meta={"task": "0"},
        )
        == "Click the bell"
    )


def test_slai_piper_policy_joints_projects_robotwin_aligned_14d():
    vector = torch.arange(32, dtype=torch.float32)
    projected = select_state_action_vector(vector, PIPER_SPACE)

    assert projected.tolist() == [vector[i].item() for i in PIPER_14_INDICES]


def test_getitem_image_backed_three_camera_path_returns_expected_shapes(tmp_path):
    module = _import_lerobot_dataset_module()

    rows = []
    for frame_idx in range(8):
        rows.append(
            {
                "observation.state": (torch.arange(32, dtype=torch.float32) + frame_idx).tolist(),
                "action": (torch.arange(32, dtype=torch.float32) + frame_idx + 0.5).tolist(),
                "observation.images.cam_high": Image.open(io.BytesIO(_png_bytes((255, 0, 0), (12, 8)))).convert("RGB"),
                "observation.images.cam_left_wrist": Image.open(io.BytesIO(_png_bytes((0, 255, 0), (6, 4)))).convert("RGB"),
                "observation.images.cam_right_wrist": Image.open(io.BytesIO(_png_bytes((0, 0, 255), (6, 4)))).convert("RGB"),
                "timestamp": float(frame_idx) / 30.0,
                "episode_index": 0,
                "index": frame_idx,
                "frame_index": frame_idx,
                "task_index": 0,
                "language_embedding": torch.ones(3, 4) * (frame_idx + 1),
            }
        )

    fake_dataset = FakeDataset(
        root=tmp_path,
        rows=rows,
        episodes={0: {"episode_index": 0, "tasks": ["Click the bell"]}},
    )
    adapter = _build_adapter(module, fake_dataset, has_concat=False, has_three_cam=True)

    with mock.patch.object(module.random, "randint", side_effect=[0, 0]):
        sample = adapter[0]

    assert sample["first_frame"].shape == (3, 24, 24)
    assert sample["video_frames"].shape == (2, 3, 24, 24)
    assert sample["initial_state"].shape == (14,)
    assert sample["action_sequence"].shape == (4, 14)
    assert sample["language_embedding"].shape == (3, 4)
    expected_state = torch.tensor(PIPER_14_INDICES, dtype=torch.float32) / 100.0
    assert torch.allclose(sample["initial_state"], expected_state, atol=1e-6)


def test_getitem_video_backed_prefers_decode_video_frames(tmp_path, monkeypatch):
    module = _import_lerobot_dataset_module()

    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_file = video_dir / "episode_000000.mp4"
    video_file.write_bytes(b"not-a-real-video")

    rows = []
    for frame_idx in range(8):
        rows.append(
            {
                "observation.state": (torch.arange(32, dtype=torch.float32) + frame_idx).tolist(),
                "action": (torch.arange(32, dtype=torch.float32) + frame_idx).tolist(),
                "timestamp": float(frame_idx) / 30.0,
                "episode_index": 0,
                "index": frame_idx,
                "frame_index": frame_idx,
                "task_index": 0,
                "language_embedding": torch.ones(2, 5),
            }
        )

    fake_dataset = FakeDataset(
        root=tmp_path,
        rows=rows,
        episodes={0: {"episode_index": 0, "task": "0"}},
        video_paths={(0, "observation.images.cam_concatenated"): "videos/episode_000000.mp4"},
    )
    adapter = _build_adapter(module, fake_dataset, has_concat=True, has_three_cam=False)

    decode_calls = []

    def fake_decode_video_frames(video_path, timestamps, tolerance_s, video_backend):
        decode_calls.append((str(video_path), list(timestamps), tolerance_s, video_backend))
        frame_count = len(timestamps)
        frames = torch.linspace(
            0.0,
            1.0,
            steps=frame_count * 3 * 6 * 6,
            dtype=torch.float32,
        ).reshape(1, frame_count, 3, 6, 6)
        return frames

    monkeypatch.setattr(module, "decode_video_frames", fake_decode_video_frames)

    with mock.patch.object(module.random, "randint", side_effect=[0, 0]):
        sample = adapter[0]

    assert len(decode_calls) == 1
    assert decode_calls[0][0].endswith("videos/episode_000000.mp4")
    assert sample["first_frame"].shape == (3, 24, 24)
    assert sample["video_frames"].shape == (2, 3, 24, 24)
    assert sample["action_sequence"].shape == (4, 14)
