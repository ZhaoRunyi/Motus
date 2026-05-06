from __future__ import annotations

import dataclasses
from typing import List, Literal, Union

import numpy as np
import torch

FIELD_NAMES = ("joint", "gripper", "ee_pos", "ee_rot")

IDS_MAP = {
    "all": ["joint", "gripper", "ee_pos", "ee_rot"],
    "joints": ["joint", "gripper"],
    "joint_gripper": ["joint", "gripper"],
    "joint_only": ["joint"],
    "ee_gripper": ["gripper", "ee_pos", "ee_rot"],
    "ee_only": ["ee_pos", "ee_rot"],
}

ARM_IDS_MAP = {
    "dual": ["left", "right"],
    "left": ["left"],
    "right": ["right"],
}

IMAGE_IDS_MAP = {
    "all": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "main": ["cam_high"],
}

ROTATION_FORMAT_ALIASES = {
    "rpy": "rpy",
    "euler": "rpy",
    "quat": "quat",
    "rot6d": "rot6d",
}

GRIPPER_FULL_WIDTH = 0.10

JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

EE_POS_NAMES = ["x", "y", "z"]
EE_ROTATION_NAMES = {
    "rpy": ["roll", "pitch", "yaw"],
    "quat": ["quat_x", "quat_y", "quat_z", "quat_w"],
    "rot6d": [f"rot6d_{idx}" for idx in range(6)],
}

DATASET_IMAGE_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}

MODEL_IMAGE_KEYS = {
    "cam_high": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}

FULL_DATA_IDS = IDS_MAP["all"]
FULL_DATA_ARMS = ARM_IDS_MAP["dual"]


def _resolve_ids(ids: Union[str, List[str]]) -> List[str]:
    if isinstance(ids, str):
        fields = IDS_MAP.get(ids)
        if fields is None:
            raise ValueError(f"Unsupported ids preset: {ids}")
        return list(fields)
    return list(ids)


def _validate_fields(fields: List[str]) -> None:
    invalid = [field for field in fields if field not in FIELD_NAMES]
    if invalid:
        raise ValueError(f"Invalid field names {invalid}; allowed: {list(FIELD_NAMES)}")


def _resolve_arms(arms: str) -> List[str]:
    resolved = ARM_IDS_MAP.get(arms)
    if resolved is None:
        raise ValueError(f"Unsupported arm preset: {arms}")
    return list(resolved)


def _resolve_image_ids(ids: Union[str, List[str]]) -> List[str]:
    if isinstance(ids, str):
        resolved = IMAGE_IDS_MAP.get(ids)
        if resolved is None:
            raise ValueError(f"Unsupported image preset: {ids}")
        return list(resolved)

    valid = set(DATASET_IMAGE_KEYS)
    invalid = [image_id for image_id in ids if image_id not in valid]
    if invalid:
        raise ValueError(f"Invalid image ids {invalid}; allowed: {sorted(valid)}")
    return list(ids)


def _resolve_rotation_format(rotation: str) -> str:
    resolved = ROTATION_FORMAT_ALIASES.get(rotation)
    if resolved is None:
        raise ValueError(f"Unsupported ee rotation format: {rotation}")
    return resolved


@dataclasses.dataclass(frozen=True)
class GripperConfig:
    type: Literal["raw", "01"] = "raw"
    threshold: float = 0.01
    full_width: float = GRIPPER_FULL_WIDTH


@dataclasses.dataclass(frozen=True)
class StateSpaceConfig:
    ids: Union[str, List[str]] = "joints"
    arms: Literal["dual", "left", "right"] = "dual"
    ee_rotation: Literal["rpy", "euler", "quat", "rot6d"] = "rot6d"
    gripper: GripperConfig | None = dataclasses.field(default_factory=GripperConfig)

    def __post_init__(self) -> None:
        fields = _resolve_ids(self.ids)
        _validate_fields(fields)
        _resolve_arms(self.arms)
        _resolve_rotation_format(self.ee_rotation)
        if "gripper" not in fields and self.gripper is not None:
            object.__setattr__(self, "gripper", None)


@dataclasses.dataclass(frozen=True)
class ActionSpaceConfig:
    ids: Union[str, List[str]] = "joints"
    arms: Literal["dual", "left", "right"] = "dual"
    ee_rotation: Literal["rpy", "euler", "quat", "rot6d"] = "rot6d"
    gripper: GripperConfig | None = dataclasses.field(default_factory=GripperConfig)

    def __post_init__(self) -> None:
        fields = _resolve_ids(self.ids)
        _validate_fields(fields)
        _resolve_arms(self.arms)
        _resolve_rotation_format(self.ee_rotation)
        if "gripper" not in fields and self.gripper is not None:
            object.__setattr__(self, "gripper", None)


@dataclasses.dataclass(frozen=True)
class ImageSpaceConfig:
    ids: Union[str, List[str]] = "all"

    def __post_init__(self) -> None:
        _resolve_image_ids(self.ids)


def _fields_from_state_config(config: StateSpaceConfig) -> List[str]:
    fields = _resolve_ids(config.ids)
    _validate_fields(fields)
    return fields


def _fields_from_action_config(config: ActionSpaceConfig) -> List[str]:
    fields = _resolve_ids(config.ids)
    _validate_fields(fields)
    return fields


def _image_ids_from_config(config: ImageSpaceConfig) -> List[str]:
    return _resolve_image_ids(config.ids)


def _space_from_state_config(config: StateSpaceConfig) -> dict[str, object]:
    return {
        "ids": _fields_from_state_config(config),
        "arms": _resolve_arms(config.arms),
        "ee_rotation": _resolve_rotation_format(config.ee_rotation),
    }


def _space_from_action_config(config: ActionSpaceConfig) -> dict[str, object]:
    return {
        "ids": _fields_from_action_config(config),
        "arms": _resolve_arms(config.arms),
        "ee_rotation": _resolve_rotation_format(config.ee_rotation),
    }


def _full_data_space(ee_rotation: str) -> dict[str, object]:
    return {
        "ids": list(FULL_DATA_IDS),
        "arms": list(FULL_DATA_ARMS),
        "ee_rotation": ee_rotation,
    }


def _field_slices_from_space(space: dict[str, object]) -> dict[str, slice]:
    field_slices: dict[str, slice] = {}
    cursor = 0

    for arm in space["arms"]:
        if "joint" in space["ids"]:
            next_cursor = cursor + len(JOINT_NAMES)
            field_slices[f"{arm}_joint"] = slice(cursor, next_cursor)
            cursor = next_cursor
        if "gripper" in space["ids"]:
            next_cursor = cursor + 1
            field_slices[f"{arm}_gripper"] = slice(cursor, next_cursor)
            cursor = next_cursor
        if "ee_pos" in space["ids"]:
            next_cursor = cursor + len(EE_POS_NAMES)
            field_slices[f"{arm}_ee_pos"] = slice(cursor, next_cursor)
            cursor = next_cursor
        if "ee_rot" in space["ids"]:
            next_cursor = cursor + len(EE_ROTATION_NAMES[space["ee_rotation"]])
            field_slices[f"{arm}_ee_rot"] = slice(cursor, next_cursor)
            cursor = next_cursor

    return field_slices


def _indices_from_space(space: dict[str, object]) -> List[int]:
    full_field_slices = _field_slices_from_space(_full_data_space(space["ee_rotation"]))
    indices: List[int] = []
    for arm in space["arms"]:
        for field in space["ids"]:
            field_slice = full_field_slices[f"{arm}_{field}"]
            indices.extend(range(field_slice.start, field_slice.stop))
    return indices


def _names_from_space(space: dict[str, object]) -> List[str]:
    names: List[str] = []
    rotation_names = EE_ROTATION_NAMES[space["ee_rotation"]]

    for arm in space["arms"]:
        if "joint" in space["ids"]:
            names.extend(f"{arm}_joint_{joint_name}" for joint_name in JOINT_NAMES)
        if "gripper" in space["ids"]:
            names.append(f"{arm}_gripper")
        if "ee_pos" in space["ids"]:
            names.extend(f"{arm}_ee_pos_{axis}" for axis in EE_POS_NAMES)
        if "ee_rot" in space["ids"]:
            names.extend(f"{arm}_ee_{axis}" for axis in rotation_names)

    return names


def _apply_gripper_01(value: np.ndarray, gripper_config: GripperConfig) -> np.ndarray:
    if gripper_config.full_width <= 0:
        raise ValueError(f"Gripper full width must be positive, got {gripper_config.full_width}")

    value = np.asarray(value)
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    width_like = value * gripper_config.full_width if max_abs <= 1.0 + 1e-6 else value
    return (width_like >= gripper_config.threshold).astype(value.dtype)


def _extract_vec(
    full: np.ndarray,
    space: dict[str, object],
    gripper_config: GripperConfig | None,
) -> np.ndarray:
    full = np.asarray(full)
    expected_dim = len(_indices_from_space(_full_data_space(space["ee_rotation"])))
    if full.shape[-1] != expected_dim:
        raise ValueError(f"Expected full Piper vector dim {expected_dim}, got {full.shape[-1]}.")
    indices = _indices_from_space(space)
    vec = full[..., indices]
    if "gripper" in space["ids"] and gripper_config is not None and gripper_config.type == "01":
        field_slices = _field_slices_from_space(space)
        vec = vec.copy()
        for arm in space["arms"]:
            gripper_slice = field_slices[f"{arm}_gripper"]
            vec[..., gripper_slice] = _apply_gripper_01(vec[..., gripper_slice], gripper_config)
    return vec


def get_space_dim(config: StateSpaceConfig | ActionSpaceConfig) -> int:
    if isinstance(config, StateSpaceConfig):
        space = _space_from_state_config(config)
    else:
        space = _space_from_action_config(config)
    return len(_indices_from_space(space))


def get_space_indices(config: StateSpaceConfig | ActionSpaceConfig) -> List[int]:
    if isinstance(config, StateSpaceConfig):
        space = _space_from_state_config(config)
    else:
        space = _space_from_action_config(config)
    return _indices_from_space(space)


def get_vector_names(config: StateSpaceConfig | ActionSpaceConfig) -> List[str]:
    if isinstance(config, StateSpaceConfig):
        space = _space_from_state_config(config)
    else:
        space = _space_from_action_config(config)
    return _names_from_space(space)


def get_image_ids(config: ImageSpaceConfig) -> List[str]:
    return _image_ids_from_config(config)


def get_image_key_map(config: ImageSpaceConfig) -> dict[str, str]:
    return {image_id: DATASET_IMAGE_KEYS[image_id] for image_id in _image_ids_from_config(config)}


def get_model_image_key_map(config: ImageSpaceConfig) -> dict[str, str]:
    return {image_id: MODEL_IMAGE_KEYS[image_id] for image_id in _image_ids_from_config(config)}


def extract_state_action_inputs(
    full_state: np.ndarray,
    actions: np.ndarray | None = None,
    *,
    state_space: StateSpaceConfig | None = None,
    action_space: ActionSpaceConfig | None = None,
) -> dict[str, np.ndarray]:
    state_space = state_space or StateSpaceConfig()
    action_space = action_space or ActionSpaceConfig()

    inputs = {
        "state": _extract_vec(
            np.asarray(full_state),
            _space_from_state_config(state_space),
            state_space.gripper,
        )
    }
    if actions is not None:
        inputs["actions"] = _extract_vec(
            np.asarray(actions),
            _space_from_action_config(action_space),
            action_space.gripper,
        )
    return inputs


def select_state_action_vector(
    value,
    config: StateSpaceConfig | ActionSpaceConfig | None = None,
) -> torch.Tensor:
    config = config or StateSpaceConfig()
    array = np.asarray(value)
    if isinstance(config, StateSpaceConfig):
        space = _space_from_state_config(config)
        selected = _extract_vec(array, space, config.gripper)
    else:
        space = _space_from_action_config(config)
        selected = _extract_vec(array, space, config.gripper)
    return torch.as_tensor(selected).float()
