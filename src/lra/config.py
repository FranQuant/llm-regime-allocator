"""Load project configuration (TOML, stdlib only)."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.toml"


def load_config(path: Path | str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "rb") as fh:
        return tomllib.load(fh)
