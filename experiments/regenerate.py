"""Regenerate every result from the seed, and check it against what is saved.

python run.py all [--from STEP] [--to STEP]     run the steps below in order (--list prints them)
python run.py check-regeneration                compare with the saved results
python run.py check-regeneration --record       save the models' and the dataset's hashes
python run.py export-checkpoints                copy the final models to checkpoints/ (committed)

A real regeneration must reuse nothing, and must not destroy the saved results before the new
ones are checked. So it writes to another folder:

    set SPACEGEN_OUTPUT=C:\\some\\empty\\folder        (PowerShell: $env:SPACEGEN_OUTPUT = "...")
    python run.py all
    python run.py check-regeneration

With SPACEGEN_OUTPUT set, data/, runs/ and reports/ are written below that folder
(spacegen/paths.py), where nothing exists yet, and the check compares them with the repository's
saved results. Without it, `all` runs in place and reuses finished runs and cached rows, which is
only a way to fill in what is missing.

`all` covers every step that produces a saved table or figure: the dataset, the generator's
report, the baselines, the evaluator, the screening runs, the first passes, the Gate 2 comparison,
the frozen seeds, the headline experiments, the pinned-furniture experiment, the failure analysis
and the figures. It takes about eight hours on the project's laptop: keep it on mains power, and
run it in parts with --from and --to if a part may be interrupted. `--from frozen-seeds` gives the
headline numbers alone (about four hours).

The check compares, on the same machine:
  tables    every reports/tables/*.csv with the version in the repository's last commit, leaving
            out the timing columns (seconds, milliseconds), which no two runs share. Everything
            else must be exactly equal: every random stream is seeded and PyTorch runs in
            deterministic mode.
  models    the SHA-256 of every trained model.pt with reports/model_hashes.json.
  dataset   the content hash of data/<version>/ with the one in reports/model_hashes.json.
  frozen    the regenerated configs/frozen.yaml with the repository's (comments aside).
  copies    the models committed in checkpoints/ with the same hashes.
  figures   after a regeneration into another folder, every reports/figures/*.png with the
            repository's file, byte by byte. The four figures that plot a measured time
            (TIMED_FIGURES) are expected to change; any other difference counts.

checkpoints/ holds the three frozen CVAE seeds and the evaluator with their run records, so a
fresh clone can run the app and the pipeline without training (spacegen.paths.saved_run).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pandas as pd
import yaml

from spacegen.config import load_config
from spacegen.paths import CHECKPOINTS_DIR, CONFIG_DIR, DATA_DIR, OUTPUT_ROOT, REPO_ROOT, REPORTS_DIR, RUNS_DIR

PY = sys.executable
FROZEN = [f"runs/cvae/frozen/seed-{k}" for k in range(3)]  # relative to OUTPUT_ROOT; resolved in final()
HELD_OUT = ("interpolation", "unseen_combination", "out_of_range")
TRAINED = ("cvae/default", "cvae/screen", "cvae/gate2", "cvae/frozen", "evaluator/e9a")  # below runs/
HASHES = REPO_ROOT / "reports" / "model_hashes.json"  # the saved reference, in the repository
EXPORTED = ("cvae/frozen/seed-0", "cvae/frozen/seed-1", "cvae/frozen/seed-2", "evaluator/e9a")  # copied to checkpoints/
TIMED_FIGURES = ("e1_results.png", "e1_final_results.png", "e8_steps.png", "e8_final_steps.png")  # they plot a time


def final(experiment: str, *options: str) -> list[str]:
    """A headline command on the three frozen seeds."""
    return [experiment, "--tag", "final", *options, "--cvae", *(str(OUTPUT_ROOT / run) for run in FROZEN)]


def steps() -> list[tuple[str, list[list[str]]]]:
    """(name, the `python run.py` commands of the step), in dependency order."""
    return [
        ("data", [["data"]]),
        ("generator-report", [["generator-report"]]),  # a Week 1 diagnostic; its figure is among the saved ones
        ("baselines", [["baselines"]]),
        ("evaluator", [["train-evaluator", "--epochs", "30", "--name", "e9a"], ["evaluator-report"]]),
        ("cvae", [["train-cvae"]]),
        ("screening", [["screen", "all"]]),
        ("first-pass-e1", [["e1"]]),
        ("first-pass-e10", [["e10"]]),
        ("gate2", [["gate2"]]),
        ("frozen-seeds", [["train-cvae", "--config", "configs/frozen.yaml", "--name", f"frozen/seed-{k}",
                           "--seed", str(k)] for k in range(3)]),
        ("e1", [final("e1")]),
        ("e8", [final("e8")]),
        *[(f"e10-{name.replace('_', '-')}", [final("e10", "--sets", name)]) for name in HELD_OUT],
        ("e10", [final("e10")]),
        ("e12", [["e12"]]),
        ("failures", [["failures"]]),
        ("figures", [["figures", "--tables"]]),
    ]


def is_timing(column: str) -> bool:
    """Columns that hold a measured time: no two runs give the same value."""
    return "seconds" in column or "ms_per_valid" in column


def run_all(first: str | None = None, last: str | None = None, run=subprocess.run, log=print) -> int:
    """Run the steps from `first` to `last` (all of them by default); stops at the first failure."""
    plan = steps()
    names = [name for name, _ in plan]
    start, stop = names.index(first or names[0]), names.index(last or names[-1])
    if OUTPUT_ROOT == REPO_ROOT:
        log("running in place: finished runs and cached rows are reused. For a regeneration from nothing, "
            "set SPACEGEN_OUTPUT to an empty folder first.")
    else:
        log(f"writing data/, runs/ and reports/ below {OUTPUT_ROOT}")
    for name, commands in plan[start:stop + 1]:
        for command in commands:
            log(f"=== {name}: python run.py {' '.join(command)}")
            code = run([PY, str(REPO_ROOT / "run.py"), *command], cwd=REPO_ROOT).returncode
            if code != 0:
                log(f"=== {name} failed (exit {code}); continue with: python run.py all --from {name}")
                return code
    return 0


def model_hashes(runs_dir: Path = RUNS_DIR) -> dict[str, str]:
    """SHA-256 of every model.pt in the trained-run folders, by its path below runs/."""
    found = {}
    for folder in TRAINED:
        for path in sorted((runs_dir / folder).rglob("model.pt")):
            found[path.parent.relative_to(runs_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def dataset_hash(data_dir: Path) -> str | None:
    metadata = data_dir / "metadata.json"
    return json.loads(metadata.read_text(encoding="utf-8"))["hash"] if metadata.exists() else None


def record(path: Path = HASHES, runs_dir: Path = RUNS_DIR, data_dir: Path | None = None) -> dict:
    """Save the hashes of the trained models and of the dataset as the reference for later checks."""
    data_dir = data_dir or DATA_DIR / load_config()["dataset"]["version"]
    saved = {"dataset": {data_dir.name: dataset_hash(data_dir)}, "models": model_hashes(runs_dir)}
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8", newline="\n")
    return saved


def export_checkpoints(runs_dir: Path = RUNS_DIR, target: Path = CHECKPOINTS_DIR) -> list[Path]:
    """Copy the final models and their run records to checkpoints/. In the copied record the
    dataset's path is written relative to the repository, as data/<version>."""
    copies = []
    for run in EXPORTED:
        source, copy = runs_dir / run, target / run
        copy.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "summary.json", "log.csv"):
            shutil.copyfile(source / name, copy / name)
        record = json.loads((source / "run.json").read_text(encoding="utf-8"))
        record["dataset"] = "data/" + PureWindowsPath(record["dataset"]).name
        (copy / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
        copies.append(copy)
    return copies


def committed(name: str, repo: Path = REPO_ROOT) -> str | None:
    """reports/tables/<name> in the repository's last commit, or None if it is not committed."""
    shown = subprocess.run(["git", "show", f"HEAD:reports/tables/{name}"], cwd=repo, capture_output=True, text=True,
                           encoding="utf-8")
    return shown.stdout if shown.returncode == 0 else None


def saved_tables(repo: Path = REPO_ROOT) -> list[str]:
    """The names of the tables in the repository's last commit."""
    listed = subprocess.run(["git", "ls-tree", "--name-only", "HEAD", "reports/tables/"], cwd=repo,
                            capture_output=True, text=True, encoding="utf-8")
    return sorted(Path(line).name for line in listed.stdout.splitlines() if line.endswith(".csv"))


def compare_table(saved: pd.DataFrame, new: pd.DataFrame) -> str | None:
    """None if the two tables agree outside their timing columns, else what differs."""
    keep = [c for c in saved.columns if not is_timing(c)]
    if keep != [c for c in new.columns if not is_timing(c)]:
        return "different columns"
    if len(saved) != len(new):
        return f"{len(saved)} rows saved, {len(new)} now"
    different = [c for c in keep if not saved[c].equals(new[c])]
    return f"columns differ: {', '.join(different)}" if different else None


def compare_figures(figures_dir: Path, saved_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """The saved figures as (identical, different, not regenerated), compared byte by byte."""
    same, different, absent = [], [], []
    for saved in sorted(saved_dir.glob("*.png")):
        new = figures_dir / saved.name
        (absent if not new.exists() else same if new.read_bytes() == saved.read_bytes() else different).append(saved.name)
    return same, different, absent


def check(tables_dir: Path = REPORTS_DIR / "tables", hashes: Path = HASHES, runs_dir: Path = RUNS_DIR,
          data_dir: Path | None = None, frozen: Path | None = None, show=committed, log=print,
          copies: Path = CHECKPOINTS_DIR, listed=saved_tables, figures_dir: Path = REPORTS_DIR / "figures",
          saved_figures: Path = REPO_ROOT / "reports" / "figures") -> int:
    """Compare tables, models, the dataset, the frozen configuration and the figures with what is
    saved; returns the number of differences. Tables, runs, figures or a dataset that are not there
    are reported, not counted."""
    problems = 0
    for path in sorted(tables_dir.glob("*.csv")):
        before = show(path.name)
        if before is None:
            log(f"new    {path.name} (not in the last commit)")
            continue
        difference = compare_table(pd.read_csv(io.StringIO(before)), pd.read_csv(path))
        problems += difference is not None
        log(f"{'DIFF ' if difference else 'equal'}  {path.name}" + (f": {difference}" if difference else ""))
    absent = [name for name in listed() if not (tables_dir / name).exists()]
    if absent:
        log(f"tables: {len(absent)} saved but not regenerated: {', '.join(absent)}")
    if hashes.exists():
        saved = json.loads(hashes.read_text(encoding="utf-8"))
        now = model_hashes(runs_dir)
        same = [run for run, digest in saved["models"].items() if now.get(run) == digest]
        changed = [run for run, digest in saved["models"].items() if run in now and now[run] != digest]
        missing = [run for run in saved["models"] if run not in now]
        problems += len(changed)
        log(f"models: {len(same)} identical, {len(changed)} different, {len(missing)} not regenerated"
            + (f"; different: {', '.join(changed)}" if changed else ""))
        data_dir = data_dir or DATA_DIR / load_config()["dataset"]["version"]
        wanted, found = saved["dataset"].get(data_dir.name), dataset_hash(data_dir)
        if found is None:
            log(f"dataset {data_dir.name}: not built")
        else:
            problems += found != wanted
            log(f"dataset {data_dir.name}: {'identical' if found == wanted else 'DIFFERENT'} ({found[:16]}...)")
        exported = {run: hashlib.sha256((copies / run / "model.pt").read_bytes()).hexdigest()
                    for run in EXPORTED if (copies / run / "model.pt").exists()}
        wrong = [run for run, digest in exported.items() if saved["models"].get(run) != digest]
        problems += len(wrong)
        if exported:
            log(f"checkpoints: {len(exported) - len(wrong)} of {len(exported)} committed copies match the saved hashes"
                + (f"; different: {', '.join(wrong)}" if wrong else ""))
    else:
        log(f"no {hashes.name}: record the hashes first (python run.py check-regeneration --record)")
    frozen = frozen or OUTPUT_ROOT / "configs" / "frozen.yaml"
    if frozen.exists() and frozen.resolve() != (CONFIG_DIR / "frozen.yaml").resolve():
        same = yaml.safe_load(frozen.read_text(encoding="utf-8")) == load_config(CONFIG_DIR / "frozen.yaml")
        problems += not same
        log(f"frozen configuration: {'identical' if same else 'DIFFERENT'}")
    if figures_dir.exists() and figures_dir.resolve() != saved_figures.resolve():
        same, different, absent = compare_figures(figures_dir, saved_figures)
        unexpected = [name for name in different if name not in TIMED_FIGURES]
        problems += len(unexpected)
        log(f"figures: {len(same)} identical, {len(different) - len(unexpected)} redrawn with new timings, "
            f"{len(unexpected)} different, {len(absent)} not regenerated"
            + (f"; different: {', '.join(unexpected)}" if unexpected else ""))
    log("everything compared is reproduced exactly" if problems == 0 else f"{problems} difference(s)")
    return problems


def main(argv: list[str] | None = None) -> int:
    names = [name for name, _ in steps()]
    parser = argparse.ArgumentParser(description="Regenerate every result, or check a regeneration.")
    parser.add_argument("action", choices=["all", "check", "export"])
    parser.add_argument("--from", dest="first", choices=names, default=None)
    parser.add_argument("--to", dest="last", choices=names, default=None)
    parser.add_argument("--list", action="store_true", help="all: print the steps and their commands, run nothing")
    parser.add_argument("--record", action="store_true", help="check: save the current model and dataset hashes")
    args = parser.parse_args(argv)
    if args.list:
        for name, commands in steps():
            for command in commands:
                print(f"{name:22s} python run.py {' '.join(command)}")
        return 0
    if args.action == "all":
        return run_all(args.first, args.last)
    if args.action == "export":
        print(f"copied {len(export_checkpoints())} runs to {CHECKPOINTS_DIR}")
        return 0
    if args.record:
        saved = record()
        print(f"recorded {len(saved['models'])} model hashes and the dataset hash in {HASHES}")
        return 0
    return 1 if check() else 0


if __name__ == "__main__":
    sys.exit(main())
