"""Compatibility wrapper for the canonical Motus websocket client helper."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.challenge_deploy.websocket_client_policy import Packer, WebsocketClientPolicy, pack_array, unpack_array, unpackb
