"""The master configuration file, configs/default.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

from spacegen.paths import CONFIG_DIR

DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
