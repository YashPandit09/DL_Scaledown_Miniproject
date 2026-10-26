"""Where a result came from: every dataset build and training run records its seed, configs,
git commit and the hash of the dataset it used."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from spacegen.catalog import CATALOG_PATH
from spacegen.config import DEFAULT_CONFIG
from spacegen.paths import REPO_ROOT
from spacegen.rules import RULES_PATH


def git_state() -> dict:
    """The current commit and whether the working tree has uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}


def config_files() -> dict[str, str]:
    """The text of the three config files, as used."""
    return {path.name: path.read_text(encoding="utf-8") for path in (DEFAULT_CONFIG, RULES_PATH, CATALOG_PATH)}


def software() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda or "none"}


def dataset_hash(data_dir: Path) -> str:
    """The hash recorded by the dataset build (spacegen.build_dataset)."""
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))["hash"]


def write_run_record(out_dir: Path, seed: int, data_dir: Path, **details) -> dict:
    """Write out_dir/run.json: seed, dataset and its hash, git state, software, configs, details."""
    record = {"seed": seed, "dataset": str(data_dir), "dataset_hash": dataset_hash(data_dir), "git": git_state(),
              "software": software(), **details, "configs": config_files()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    return record
