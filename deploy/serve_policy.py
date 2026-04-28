#!/usr/bin/env python3
"""Compatibility wrapper for the canonical Motus challenge deploy server."""

import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.challenge_deploy.serve_policy import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
