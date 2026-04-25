#!/usr/bin/env bash
set -euo pipefail

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /workspace/Motus/.venv/bin/python -m pytest \
  /workspace/Motus/tests/test_lerobot_adapter.py \
  /workspace/Motus/tests/test_piper_click_bell_contract.py \
  -q
