"""Redraw every figure from the saved tables and logs, without training or sampling.

python run.py figures            redraw the figures
python run.py figures --tables   first rebuild the screening tables from the saved runs

Figures come from reports/tables/*.csv. The training curves (E2, E3b, E4) also need the per-epoch
logs of the screening runs: they are read from runs/cvae/screen/ when the runs are on this
machine, and from the copies in reports/logs/screen/ otherwise (a fresh clone). Nothing is
trained and no layout is sampled, so this takes seconds.

  dataset_<version>_*      from data/<version>/ (skipped when the dataset is not built)
  evaluator_confusion      from reports/tables/evaluator.csv
  e2_* ... e8_*            the screening
  e1_*, e10_*              the first pass on the default configuration
  gate2_candidates         from reports/tables/gate2_runs.csv
  e1_final_*, e8_final_*, e10_final_*     the headline runs on the frozen configuration

--tables rebuilds reports/tables/e2.csv to e7.csv from the runs in runs/cvae/screen/ (a missing
run is an error) and refreshes the copied logs. The other tables need sampling and are written
by their own tasks (`screen e8`, `e1`, `e8`, `e10`, `gate2`, `baselines`, `evaluator-report`).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from experiments.figures import FIGURES, LOGS_DIR
from experiments.screening import EXPERIMENTS, run_experiment
from spacegen.config import load_config
from spacegen.paths import DATA_DIR, REPORTS_DIR, RUNS_DIR
from spacegen.viz import SURFACE, plot_confusion

SCREEN_RUNS = RUNS_DIR / "cvae" / "screen"
TAGGED = ("e1", "e8", "e10")  # experiments that also have a headline table, <experiment>_final.csv


def redraw(tables_dir: Path = REPORTS_DIR / "tables", figures_dir: Path = REPORTS_DIR / "figures",
           runs_dir: Path | None = SCREEN_RUNS, data_dir: Path | None = None, log=print) -> list[Path]:
    """Draw every figure whose table exists; returns the files written."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(figures: dict, prefix: str) -> None:
        for name, figure in figures.items():
            written.append(figures_dir / f"{prefix}_{name}.png")
            figure.savefig(written[-1], facecolor=SURFACE)

    for name in FIGURES:
        sources = [(name, tables_dir / ("gate2_runs.csv" if name == "gate2" else f"{name}.csv"))]
        if name in TAGGED:
            sources.append((f"{name}_final", tables_dir / f"{name}_final.csv"))
        for prefix, path in sources:
            if not path.exists():
                log(f"{prefix}: no table {path.name}, skipped")
                continue
            try:
                save(FIGURES[name](pd.read_csv(path), runs_dir), prefix)
            except FileNotFoundError as error:  # a curve figure without its per-epoch logs
                log(f"{prefix}: {error}, skipped")
    evaluator = tables_dir / "evaluator.csv"
    if evaluator.exists():
        row = pd.read_csv(evaluator).iloc[0]
        written.append(figures_dir / "evaluator_confusion.png")
        plot_confusion(row.to_dict(), row["run"]).savefig(written[-1], facecolor=SURFACE)
    if data_dir is not None and (data_dir / "metadata.json").exists():
        from spacegen.build_dataset import make_figures
        from spacegen.splits import load_split_config

        written += make_figures(data_dir, figures_dir, load_split_config())
    else:
        log("dataset figures: the dataset is not built, skipped")
    return written


def rebuild_tables(data_dir: Path, seed: int, tables_dir: Path = REPORTS_DIR / "tables",
                   runs_dir: Path = SCREEN_RUNS, experiments=tuple(EXPERIMENTS), log=print) -> list[Path]:
    """The screening tables from the saved runs; nothing is trained."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in experiments:
        table = run_experiment(name, data_dir, seed, "cpu", runs_dir, train=False, log=log)
        written.append(tables_dir / f"{name}.csv")
        table.round(5).to_csv(written[-1], index=False, lineterminator="\n")
    return written


def copy_logs(tables_dir: Path = REPORTS_DIR / "tables", runs_dir: Path = SCREEN_RUNS,
              copies: Path = LOGS_DIR) -> list[Path]:
    """Copy the per-epoch log of every run named in a screening table next to the reports, so
    the curves can be redrawn without the runs."""
    copies.mkdir(parents=True, exist_ok=True)
    written = []
    for name in EXPERIMENTS:
        table = tables_dir / f"{name}.csv"
        if not table.exists():
            continue
        for run in pd.read_csv(table)["run"].unique():
            source = runs_dir / run / "log.csv"
            if source.exists():
                written.append(copies / f"{run}.csv")
                shutil.copyfile(source, written[-1])
    return sorted(set(written))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redraw every figure from the saved tables and logs.")
    parser.add_argument("--tables", action="store_true",
                        help="first rebuild the screening tables E2 to E7 from runs/cvae/screen/ and copy their logs")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    args = parser.parse_args(argv)
    raw = load_config()
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    if args.tables:
        tables = rebuild_tables(data_dir, raw["seed"])
        print(f"rebuilt {len(tables)} tables and copied {len(copy_logs())} logs to {LOGS_DIR}")
    written = redraw(runs_dir=SCREEN_RUNS if SCREEN_RUNS.exists() else None, data_dir=data_dir)
    print(f"wrote {len(written)} figures to {REPORTS_DIR / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
