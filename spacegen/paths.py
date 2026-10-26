"""Repository paths, so no module depends on the current working directory.

Everything a run writes (data/, runs/, reports/) goes below OUTPUT_ROOT. That is the repository
itself, unless the environment variable SPACEGEN_OUTPUT names another folder: a regeneration
(`python run.py all`) writes there, starts from nothing and leaves the saved results alone.
The configuration files are always read from the repository.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_ROOT = Path(os.environ["SPACEGEN_OUTPUT"]).resolve() if os.environ.get("SPACEGEN_OUTPUT") else REPO_ROOT
DATA_DIR = OUTPUT_ROOT / "data"
REPORTS_DIR = OUTPUT_ROOT / "reports"
RUNS_DIR = OUTPUT_ROOT / "runs"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"  # committed copies of the final models (the frozen seeds, the evaluator)


def saved_run(*parts: str) -> Path:
    """A trained run by its path below runs/, for example saved_run("cvae", "frozen", "seed-0"):
    the run folder if its model is there, otherwise the committed copy in checkpoints/ if that
    exists (a fresh clone has no runs/), otherwise the run folder (so the caller can say it is missing)."""
    run, copy = RUNS_DIR.joinpath(*parts), CHECKPOINTS_DIR.joinpath(*parts)
    return copy if not (run / "model.pt").exists() and (copy / "model.pt").exists() else run
