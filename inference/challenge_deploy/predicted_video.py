from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import torch


def session_video_filename(session_id: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_.-]+", "_", session_id).strip("._")
    return f"{(normalized or 'session')}_predicted_video.mp4"


def predicted_frames_to_bgr(predicted_frames: torch.Tensor) -> list[np.ndarray]:
    frames = predicted_frames.detach().cpu().float()
    if frames.dim() != 5:
        raise ValueError(f"Expected predicted_frames with 5 dims, got {tuple(frames.shape)}")
    if frames.shape[1] == 3:
        frames = frames.permute(0, 2, 1, 3, 4)
    frames = frames.squeeze(0).clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy()
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]


class PredictedVideoStore:
    def __init__(self, output_dir: str | Path, *, fps: float, frame_size: tuple[int, int]) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._frame_size = frame_size
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._paths: dict[str, Path] = {}

    def append(self, session_id: str, predicted_frames: torch.Tensor) -> None:
        writer = self._writers.get(session_id)
        if writer is None:
            path = self._output_dir / session_video_filename(session_id)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, self._frame_size)
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open predicted video writer: {path}")
            self._writers[session_id] = writer
            self._paths[session_id] = path
        for frame in predicted_frames_to_bgr(predicted_frames):
            writer.write(frame)

    def fetch(self, session_id: str) -> dict[str, Any]:
        path = self.close(session_id)
        if path is None or not path.exists():
            return {"predicted_video_name": None, "predicted_video_remote_path": None, "predicted_video_bytes": None}
        return {
            "predicted_video_name": path.name,
            "predicted_video_remote_path": str(path),
            "predicted_video_bytes": path.read_bytes(),
        }

    def close_many(self, session_ids: set[str]) -> None:
        for session_id in session_ids:
            self.close(session_id)

    def close(self, session_id: str) -> Path | None:
        writer = self._writers.pop(session_id, None)
        if writer is not None:
            writer.release()
        return self._paths.get(session_id)
