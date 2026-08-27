"""Load non-secret configuration (endpoints, model names) from config.toml.

Secrets (API keys) stay in .env and are read from os.environ, not here.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

with open(_CONFIG_PATH, "rb") as _fh:
    CONFIG: dict[str, Any] = tomllib.load(_fh)
