import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.lerobot.slai_piper_policy import StateSpaceConfig, get_space_dim


def _load_jsonlines(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_piper_dataset_metadata_and_motus_config_are_aligned():
    dataset_root = Path("/workspace/data/ZhaoRunyi/Piper_click_bell_0403")
    config_path = Path("/workspace/Motus/configs/piper_click_bell_0403_cpu_smoke.yaml")
    stats_path = Path("/workspace/Motus/data/utils/stat.json")
    checkpoint_path = Path("/workspace/ckpts/Motus/mp_rank_00_model_states.pt")

    tasks = _load_jsonlines(dataset_root / "meta" / "tasks.jsonl")
    episodes = _load_jsonlines(dataset_root / "meta" / "episodes.jsonl")
    config = yaml.safe_load(config_path.read_text())
    stats = json.loads(stats_path.read_text())

    assert tasks == [{"task_index": 0, "task": "Click the bell"}]
    assert len(episodes) == 43
    assert all(ep["tasks"] == ["Click the bell"] for ep in episodes)

    assert config["common"]["action_dim"] == 14
    assert config["common"]["state_dim"] == 14
    assert config["common"]["global_downsample_rate"] == 3
    assert config["dataset"]["params"]["state_action_space"] == "joints"
    assert config["dataset"]["params"]["state_action_arms"] == "dual"
    assert config["dataset"]["params"]["embodiment_type"] == "piper_click_bell_0403_dual_14d"
    assert (
        get_space_dim(
            StateSpaceConfig(
                ids=config["dataset"]["params"]["state_action_space"],
                arms=config["dataset"]["params"]["state_action_arms"],
            )
        )
        == 14
    )

    stats_entry = stats["piper_click_bell_0403_dual_14d"]
    assert len(stats_entry["min"]) == 14
    assert len(stats_entry["max"]) == 14
    assert stats_entry["action_dim"] == 14

    assert checkpoint_path.exists()
