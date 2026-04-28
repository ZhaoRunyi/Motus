#!/usr/bin/env python3
"""Serve Motus over websocket with an openpi-style client protocol."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http
import logging
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
import websockets
import websockets.asyncio.server as _server
import websockets.frames
import yaml

try:
    from .predicted_video import PredictedVideoStore
    from .websocket_client_policy import Packer, unpackb
except ImportError:
    from predicted_video import PredictedVideoStore
    from websocket_client_policy import Packer, unpackb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BAK_ROOT = PROJECT_ROOT / "bak"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BAK_ROOT) not in sys.path:
    sys.path.insert(0, str(BAK_ROOT))

from models.motus import Motus  # noqa: E402
from models.motus import MotusConfig  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # noqa: E402


logger = logging.getLogger(__name__)


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def build_vlm_inputs(
    processor: AutoProcessor,
    prompt: str,
    image: Image.Image,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    encoded = processor(text=[text], images=[image], return_tensors="pt")

    vlm_inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "pixel_values": encoded["pixel_values"].to(device),
        "image_grid_thw": encoded.get("image_grid_thw"),
    }
    if vlm_inputs["image_grid_thw"] is not None:
        vlm_inputs["image_grid_thw"] = vlm_inputs["image_grid_thw"].to(device)
    return vlm_inputs


def create_motus_from_yaml(config_dict: dict[str, Any], device: torch.device) -> Motus:
    common = config_dict["common"]
    model_cfg = config_dict["model"]
    und_cfg = model_cfg.get("und_expert", {})
    und_vlm_cfg = und_cfg.get("vlm", {})
    action_cfg = model_cfg["action_expert"]
    loss_cfg = model_cfg["loss_weights"]
    config = MotusConfig(
        wan_checkpoint_path=model_cfg["wan"]["checkpoint_path"],
        vae_path=model_cfg["wan"]["vae_path"],
        wan_config_path=model_cfg["wan"]["config_path"],
        video_precision=model_cfg["wan"]["precision"],
        vlm_checkpoint_path=model_cfg["vlm"]["checkpoint_path"],
        und_expert_hidden_size=int(und_cfg.get("hidden_size", 512)),
        und_expert_ffn_dim_multiplier=int(und_cfg.get("ffn_dim_multiplier", 4)),
        und_expert_norm_eps=float(und_cfg.get("norm_eps", 1e-5)),
        vlm_adapter_input_dim=int(und_vlm_cfg.get("input_dim", 2048)),
        vlm_adapter_projector_type=str(und_vlm_cfg.get("projector_type", "mlp3x_silu")),
        num_layers=30,
        action_state_dim=int(common["state_dim"]),
        action_dim=int(common["action_dim"]),
        action_expert_dim=int(action_cfg["hidden_size"]),
        action_expert_ffn_dim_multiplier=int(action_cfg["ffn_dim_multiplier"]),
        action_expert_norm_eps=float(action_cfg.get("norm_eps", 1e-6)),
        global_downsample_rate=int(common["global_downsample_rate"]),
        video_action_freq_ratio=int(common["video_action_freq_ratio"]),
        num_video_frames=int(common["num_video_frames"]),
        video_height=int(common["video_height"]),
        video_width=int(common["video_width"]),
        batch_size=1,
        video_loss_weight=float(loss_cfg["video_loss_weight"]),
        action_loss_weight=float(loss_cfg["action_loss_weight"]),
        training_mode="finetune",
        load_pretrained_backbones=False,
    )
    return Motus(config).to(device)


def prompt_cache_filename(prompt: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", prompt).strip("_").lower()
    if not normalized:
        normalized = "prompt"
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{normalized[:80]}_{digest}.pt"


class MotusRemotePolicy:
    """Thin policy wrapper that mirrors openpi's remote serving flow."""

    def __init__(
        self,
        *,
        model_config: str,
        ckpt_dir: str,
        wan_path: str | None,
        default_prompt: str | None,
        t5_embeds: str | None,
        device: str,
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._config_dict = load_yaml_config(model_config)
        self._default_prompt = default_prompt
        self._wan_path = wan_path
        self._state_dim = int(self._config_dict["common"]["state_dim"])
        self._action_dim = int(self._config_dict["common"]["action_dim"])
        self._video_height = int(self._config_dict["common"]["video_height"])
        self._video_width = int(self._config_dict["common"]["video_width"])
        self._num_inference_steps = int(self._config_dict["model"]["inference"]["num_inference_timesteps"])
        self._action_chunk_size = (
            int(self._config_dict["common"]["num_video_frames"])
            * int(self._config_dict["common"]["video_action_freq_ratio"])
        )

        logger.info("Loading Motus model")
        self._model = create_motus_from_yaml(self._config_dict, self._device)
        self._model.eval()
        self._model.load_checkpoint(ckpt_dir, strict=False)

        vlm_ckpt = self._config_dict["model"]["vlm"]["checkpoint_path"]
        self._processor = AutoProcessor.from_pretrained(vlm_ckpt, trust_remote_code=True)

        self._t5_encoder = None
        self._prompt_cache: dict[str, list[torch.Tensor]] = {}
        self._fixed_language_embeddings = None
        self._prompt_t5_cache_dir = Path(
            os.environ.get("MOTUS_T5_CACHE_DIR", str(PROJECT_ROOT / "t5_prompt_cache"))
        )
        self._prompt_t5_cache_dir.mkdir(parents=True, exist_ok=True)
        self._predicted_videos = PredictedVideoStore(
            PROJECT_ROOT / "logs" / "predicted_videos",
            fps=4.0,
            frame_size=(self._video_width, self._video_height),
        )

        if t5_embeds is not None:
            if self._default_prompt is None:
                raise ValueError("--default_prompt is required when --t5_embeds is used")
            self._fixed_language_embeddings = self._normalize_language_embeddings(
                torch.load(t5_embeds, map_location=self._device)
            )

        prompt_mode = "request"
        if self._fixed_language_embeddings is not None:
            prompt_mode = "fixed"
        elif self._wan_path is not None:
            prompt_mode = "request_cached_or_generate"
        else:
            prompt_mode = "request_cached"

        self.metadata = {
            "model": "Motus",
            "device": str(self._device),
            "default_prompt": self._default_prompt,
            "prompt_mode": prompt_mode,
            "prompt_t5_cache_dir": str(self._prompt_t5_cache_dir),
            "image_size": [self._video_height, self._video_width],
            "state_dim": self._state_dim,
            "action_dim": self._action_dim,
            "action_chunk_size": self._action_chunk_size,
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = self._resolve_prompt(obs)
        session_id = str(obs.get("session_id") or "default")
        first_frame, first_frame_pil = self._extract_image(obs)
        state = self._extract_state(obs)
        language_embeddings = self._resolve_language_embeddings(obs, prompt)
        vlm_inputs = build_vlm_inputs(self._processor, prompt, first_frame_pil, self._device)

        with torch.inference_mode():
            predicted_frames, predicted_actions = self._model.inference_step(
                first_frame=first_frame,
                state=state,
                num_inference_steps=self._num_inference_steps,
                language_embeddings=language_embeddings,
                vlm_inputs=[vlm_inputs],
            )
        self._predicted_videos.append(session_id, predicted_frames)
        return {"actions": predicted_actions.squeeze(0).float().cpu().numpy()}

    def get_predicted_video(self, session_id: str) -> dict[str, Any]:
        return self._predicted_videos.fetch(session_id)

    def close_predicted_videos(self, session_ids: set[str]) -> None:
        self._predicted_videos.close_many(session_ids)

    def _resolve_prompt(self, obs: dict[str, Any]) -> str:
        if self._fixed_language_embeddings is not None:
            return self._default_prompt
        prompt = obs.get("prompt", self._default_prompt)
        if prompt is None:
            raise ValueError("prompt is required when no fixed --default_prompt is configured")
        return prompt

    def _resolve_language_embeddings(self, obs: dict[str, Any], prompt: str) -> list[torch.Tensor]:
        if "t5_embeds" in obs:
            return self._normalize_language_embeddings(obs["t5_embeds"])

        if self._fixed_language_embeddings is not None:
            return self._clone_language_embeddings(self._fixed_language_embeddings)

        cached = self._prompt_cache.get(prompt)
        if cached is not None:
            return self._clone_language_embeddings(cached)

        cache_path = self._prompt_t5_cache_dir / prompt_cache_filename(prompt)
        if cache_path.exists():
            normalized = self._normalize_language_embeddings(torch.load(cache_path, map_location=self._device))
            self._prompt_cache[prompt] = self._clone_language_embeddings(normalized)
            return normalized

        if self._wan_path is None:
            raise ValueError(
                f"No cached T5 embedding found for prompt {prompt!r}, and --wan_path was not provided for generation."
            )

        encoded = self._ensure_t5_encoder()([prompt], device=str(self._device))
        normalized = self._normalize_language_embeddings(encoded)
        torch.save(normalized[0].detach().cpu(), cache_path)
        self._prompt_cache[prompt] = self._clone_language_embeddings(normalized)
        return normalized

    def _ensure_t5_encoder(self) -> T5EncoderModel:
        if self._t5_encoder is not None:
            return self._t5_encoder
        if self._wan_path is None:
            raise ValueError("WAN path is required to generate missing prompt T5 embeddings.")
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=str(self._device),
            checkpoint_path=os.path.join(self._wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self._wan_path, "Wan2.2-TI2V-5B", "google", "umt5-xxl"),
        )
        return self._t5_encoder

    def _extract_state(self, obs: dict[str, Any]) -> torch.Tensor:
        state = None
        if "observation/state" in obs:
            state = obs["observation/state"]
        elif "state" in obs:
            state = obs["state"]
        elif isinstance(obs.get("observation"), dict) and "state" in obs["observation"]:
            state = obs["observation"]["state"]

        if state is None:
            state_array = np.zeros((self._state_dim,), dtype=np.float32)
        else:
            state_array = np.asarray(state)
            if state_array.ndim > 1:
                state_array = state_array.reshape(-1)

        return torch.as_tensor(state_array, dtype=torch.bfloat16, device=self._device).view(1, -1)

    def _extract_image(self, obs: dict[str, Any]) -> tuple[torch.Tensor, Image.Image]:
        image = None
        if "observation/image" in obs:
            image = obs["observation/image"]
        elif "image" in obs:
            image = obs["image"]
        elif isinstance(obs.get("observation"), dict) and "image" in obs["observation"]:
            image = obs["observation"]["image"]

        if image is None:
            raise ValueError("image is required")

        image_array = np.asarray(image)
        if image_array.ndim == 4:
            image_array = image_array[0]
        if image_array.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, got {image_array.shape}")

        if image_array.shape[0] == 3 and image_array.shape[-1] != 3:
            image_array = np.transpose(image_array, (1, 2, 0))

        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.float32)
            if image_array.max() <= 1.0:
                image_array = image_array * 255.0
            image_array = np.clip(image_array, 0.0, 255.0).astype(np.uint8)

        image_pil = Image.fromarray(image_array, mode="RGB").resize(
            (self._video_width, self._video_height),
            Image.BICUBIC,
        )
        image_float = np.asarray(image_pil).astype(np.float32) / 255.0
        first_frame = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0).to(self._device)
        return first_frame, image_pil

    def _normalize_language_embeddings(self, language_embeddings: Any) -> list[torch.Tensor]:
        if isinstance(language_embeddings, torch.Tensor):
            if language_embeddings.dim() == 3:
                return [language_embeddings[0].to(self._device)]
            if language_embeddings.dim() == 2:
                return [language_embeddings.to(self._device)]
            raise ValueError(f"Unsupported T5 tensor shape: {tuple(language_embeddings.shape)}")

        if isinstance(language_embeddings, np.ndarray):
            return self._normalize_language_embeddings(torch.from_numpy(language_embeddings))

        if isinstance(language_embeddings, list):
            normalized = []
            for tensor in language_embeddings:
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                normalized.append(tensor.to(self._device))
            return normalized

        raise ValueError(f"Unsupported t5_embeds type: {type(language_embeddings)}")

    def _clone_language_embeddings(self, language_embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor.clone() for tensor in language_embeddings]


class WebsocketPolicyServer:
    """Serve a policy using the same msgpack websocket protocol as openpi."""

    def __init__(self, policy: MotusRemotePolicy, *, host: str, port: int) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._policy.metadata))

        prev_total_time = None
        session_ids: set[str] = set()
        try:
            while True:
                try:
                    start_time = time.monotonic()
                    obs = unpackb(await websocket.recv())
                    if obs.get("_request") == "get_predicted_video":
                        payload = self._policy.get_predicted_video(str(obs.get("session_id") or "default"))
                        await websocket.send(packer.pack(payload))
                        continue
                    session_ids.add(str(obs.get("session_id") or "default"))

                    infer_start = time.monotonic()
                    action = self._policy.infer(obs)
                    infer_ms = (time.monotonic() - infer_start) * 1000.0

                    action["server_timing"] = {"infer_ms": infer_ms}
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time
                except websockets.ConnectionClosed:
                    logger.info("Connection from %s closed", websocket.remote_address)
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            self._policy.close_predicted_videos(session_ids)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Motus with an openpi-style websocket API")
    parser.add_argument("--model_config", required=True, help="Path to a Motus YAML config")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory containing mp_rank_00_model_states.pt")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--wan_path", default=None, help="Base pretrained model directory for WAN T5 loading")
    parser.add_argument("--default_prompt", default=None, help="Default instruction text when the client omits prompt")
    parser.add_argument("--t5_embeds", default=None, help="Path to a pre-encoded T5 embedding .pt file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = MotusRemotePolicy(
        model_config=args.model_config,
        ckpt_dir=args.ckpt_dir,
        wan_path=args.wan_path,
        default_prompt=args.default_prompt,
        t5_embeds=args.t5_embeds,
        device=args.device,
    )
    logger.info("Starting Motus websocket server on %s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
