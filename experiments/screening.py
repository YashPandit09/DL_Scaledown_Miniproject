"""Screening runs for the CVAE ablations (E2 to E8): one factor at a time, one seed each.

python run.py screen e2          (or e3a, e3b, e4, e5, e6, e7, e8, all)

  E2   position loss: MSE, MAE, Huber (delta 0.01, 0.05, 0.1) on the clean data and on the
       outlier variant (4% of training layouts with one item moved to a random spot), plus MSE
       with the overlap penalty
  E3a  hidden activation with BatchNorm on (the deployed setting): decides the activation
  E3b  the same activations with BatchNorm off at depth 2 and 6, plus BatchNorm-on references
       for Sigmoid and ReLU at depth 6. These run a fixed 60 epochs, so that gradient norms
       exist at epochs 1, 10 and 50 for every run
  E4   optimizer and learning rate: Adam, SGD, SGD with momentum 0.9, RMSProp
  E5   KL weight beta (0.01, 0.1, 0.5, 1) and latent size (4, 16, 32)
  E6   regularization: BatchNorm off, dropout 0 and 0.3, 200 fixed epochs instead of early stopping
  E7   position head: Sigmoid against Linear clamped to [0, 1], error near walls against the rest
  E8   no training: latent-optimization steps 0 to 200 on the default model (M2)

Every run is the default configuration with a few overrides. Runs with the same overrides are
trained once (runs/cvae/screen/<overrides>/) and shared between experiments, and a finished run
is not trained again. Each experiment writes reports/tables/<experiment>.csv and its figures.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from spacegen.catalog import load_room_catalog
from spacegen.config import load_config
from spacegen.dataset import load_layouts
from spacegen.evaluate import evaluate, evaluation_rooms, summarize
from spacegen.latent_opt import load_latent_opt_config
from spacegen.models.cvae import ACTIVATIONS
from spacegen.paths import DATA_DIR, REPORTS_DIR, RUNS_DIR
from spacegen.pipeline import LatentOptSampler, load_cvae_run
from spacegen.rules import load_rules
from spacegen.train_cvae import load_configs, m1_check, train_cvae, train_eval_loss

HUBER = (0.01, 0.05, 0.1)
FIXED_60 = ["cvae_training.early_stopping=false", "cvae_training.max_epochs=60"]
OUTLIERS = "cvae_training.outliers=0.04"
OVERLAP = "cvae.lambda_overlap=1.0"  # the overlap term, on the scale of the reconstruction loss


def _losses(extra: list[str], suffix: str) -> list[tuple[str, list[str]]]:
    return ([(f"mse{suffix}", extra), (f"mae{suffix}", ["cvae.position_loss=mae", *extra])]
            + [(f"huber {d:g}{suffix}", ["cvae.position_loss=huber", f"cvae.huber_delta={d}", *extra]) for d in HUBER])


EXPERIMENTS: dict[str, list[tuple[str, list[str]]]] = {
    "e2": _losses([], "") + _losses([OUTLIERS], ", outliers") + [("mse, overlap term", [OVERLAP])],
    "e3a": [(name, [f"cvae.activation={name}"]) for name in ACTIVATIONS],
    "e3b": [(f"{name}, depth {depth}", [f"cvae.activation={name}", f"cvae.depth={depth}", "cvae.batch_norm=false",
                                         *FIXED_60])
            for depth in (2, 6) for name in ACTIVATIONS]
           + [(f"{name}, depth 6, BatchNorm on", [f"cvae.activation={name}", "cvae.depth=6", *FIXED_60])
              for name in ("sigmoid", "relu")],
    "e4": [("adam 1e-3", []), ("adam 1e-4", ["cvae_training.lr=0.0001"]), ("adam 1e-2", ["cvae_training.lr=0.01"]),
           *[(f"{label} {lr:g}", [f"cvae_training.optimizer={name}", f"cvae_training.lr={lr}"])
             for label, name, rates in (("sgd", "sgd", (0.01, 0.1)), ("sgd momentum", "sgd_momentum", (0.01, 0.1)),
                                        ("rmsprop", "rmsprop", (0.0001, 0.001)))
             for lr in rates]],
    "e5": [("beta 0.1, latent 16", []), *[(f"beta {b:g}", [f"cvae_training.beta_target={b}"]) for b in (0.01, 0.5, 1.0)],
           *[(f"latent {z}", [f"cvae.latent={z}"]) for z in (4, 32)]],
    "e6": [("default", []), ("BatchNorm off", ["cvae.batch_norm=false"]), ("dropout 0", ["cvae.dropout=0.0"]),
           ("dropout 0.3", ["cvae.dropout=0.3"]), ("200 fixed epochs", ["cvae_training.early_stopping=false"])],
    "e7": [("sigmoid head", []), ("linear head, clamped", ["cvae.position_head=linear"])],
}
TARGET_LOSS = 0.65  # E4: epochs until the validation loss first reaches this
E8_STEPS = (0, 25, 50, 100, 200)


def effective(overrides: list[str]) -> list[str]:
    """The overrides that actually change the default configuration."""
    defaults = dict(zip(("cvae", "cvae_training"), (dataclasses.asdict(c) for c in load_configs())))
    changed = []
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        if yaml.safe_load(value) != defaults[section][field]:
            changed.append(override)
    return changed


def run_name(overrides: list[str]) -> str:
    """A folder name for a set of overrides: the same settings always map to the same run."""
    parts = sorted(o.split(".", 1)[1].replace("=", "-") for o in effective(overrides))
    return "__".join(parts) or "default"


def run_experiment(name: str, data_dir: Path, seed: int, device: str, runs_dir: Path = RUNS_DIR / "cvae" / "screen",
                   settings: list[tuple[str, list[str]]] | None = None, base: list[str] = (),
                   log=print, train: bool = True) -> pd.DataFrame:
    """Train what is missing, then collect one row per setting of the experiment. With
    train=False a missing run is an error (the tables are rebuilt from the saved runs only)."""
    rows = []
    for label, overrides in settings or EXPERIMENTS[name]:
        run_dir = runs_dir / run_name([*base, *overrides])
        if not (run_dir / "summary.json").exists():
            if not train:
                raise FileNotFoundError(f"{name}: the run {run_dir.name} is not in {runs_dir}")
            log(f"{name}: training {label} -> {run_dir.name}")
            model, training = load_configs(overrides=[*base, *overrides])
            train_cvae(data_dir, run_dir, seed, device, model, training, log=lambda message: None)
        summary = _backfilled(run_dir, data_dir, seed, device)
        epochs = pd.read_csv(run_dir / "log.csv")
        m1 = summary.get("m1_check") or {}
        final = epochs.iloc[-1]
        selected = epochs.loc[epochs["epoch"] == summary["selected_epoch"]].iloc[0]
        reached = epochs.loc[epochs["val_loss"] <= TARGET_LOSS, "epoch"]
        rows.append({"experiment": name, "setting": label, "run": run_dir.name,
                     "epochs": summary["epochs_run"], "selected_epoch": summary["selected_epoch"],
                     "val_loss": summary["selected_validation_loss"], "val_position": summary["selected_position_loss"],
                     "val_rotation": summary["selected_rotation_loss"], "val_kl": summary["selected_kl"],
                     "active_units": summary["active_units"],
                     "position_error_median": summary["position_error_median"],
                     "position_error_mean": summary["position_error_mean"],
                     "position_error_near_walls": summary.get("position_error_near_walls"),
                     "position_error_rest": summary.get("position_error_rest"),
                     "train_loss": float(selected["train_loss"]), "train_eval_loss": summary["train_eval_loss"],
                     "epochs_to_target": int(reached.min()) if len(reached) else None,
                     "m1_rvr": m1.get("rvr"), "m1_quality": m1.get("quality"), "m1_diversity": m1.get("diversity"),
                     "dead_share": float(final.filter(like="val_dead_").mean()),
                     "seconds": float(epochs["seconds"].sum())})
    return pd.DataFrame(rows)


def _backfilled(run_dir: Path, data_dir: Path, seed: int, device: str) -> dict:
    """The run's summary, with the fields added after the run was trained: the diversity of the M1
    check (rerun with the same seed, which must reproduce the stored raw valid rate exactly) and
    the eval-mode training loss (E6). Both are written back to summary.json."""
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    check = summary.get("m1_check")
    missing_diversity = check is not None and "diversity" not in check
    if not missing_diversity and "train_eval_loss" in summary:
        return summary
    catalog = load_room_catalog("living_room")
    set_a = load_layouts(data_dir / "set_a.npz")
    with np.load(data_dir / "splits.npz") as splits:
        rows = {part: splits[f"set_a_{part}"] for part in ("train", "test")}
    # the M1 check draws z on the run's device, and the CPU and the GPU give different draws for one seed
    device = json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get("device", device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{run_dir.name} was trained on the GPU; its M1 check can only be repeated on one")
    model = load_cvae_run(run_dir, device)
    if missing_diversity:
        redone = m1_check(model, set_a, rows["test"], check["rooms"], seed, catalog, device)
        if abs(redone["rvr"] - check["rvr"]) > 1e-12:
            raise RuntimeError(f"{run_dir.name}: the rerun M1 check gives {redone['rvr']}, not {check['rvr']}")
        summary["m1_check"] = redone
    if "train_eval_loss" not in summary:
        beta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["cvae_training"]["beta_target"]
        summary["train_eval_loss"] = train_eval_loss(model, set_a.subset(rows["train"]), seed, beta, catalog, device)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return summary


def run_latent_steps(data_dir: Path, seed: int, device: str, steps=E8_STEPS, rooms: int = 100,
                     model_dir: Path = RUNS_DIR / "cvae" / "screen" / "default", log=print) -> pd.DataFrame:
    """E8: M2 with 0 to 200 optimization steps on the same rooms (the diversity-reference rooms of
    the Set A test split), 64 samples each: raw validity, diversity ratio to G0, time per room."""
    catalog, rules = load_room_catalog("living_room"), load_rules()
    model = load_cvae_run(model_dir, device)
    conditions, reference = evaluation_rooms(data_dir, rooms, np.random.default_rng(seed), catalog)
    rows = []
    for count in steps:
        sampler = LatentOptSampler(model, catalog, rules, load_latent_opt_config(steps=count), device)
        table = evaluate(sampler, conditions, 64, np.random.default_rng(seed), catalog, rules)
        rows.append({"steps": count, **summarize(f"M2, {count} steps", table, reference)})
        log(f"e8: {count} steps, raw valid {rows[-1]['rvr']:.1%}, {rows[-1]['seconds_per_room']:.2f} s per room")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CVAE screening experiments (E2, E3a, E3b).")
    parser.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "e8", "all"])
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    raw = load_config()
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    names = [*EXPERIMENTS, "e8"] if "all" in args.experiments else args.experiments
    tables_dir, figures_dir = REPORTS_DIR / "tables", REPORTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name == "e8":  # latent optimization of 64 candidates is faster on the CPU
            table = run_latent_steps(data_dir, raw["seed"], "cpu")
        else:
            table = run_experiment(name, data_dir, raw["seed"], args.device)
        table.round(5).to_csv(tables_dir / f"{name}.csv", index=False, lineterminator="\n")
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(table.drop(columns=["experiment", "run"], errors="ignore").round(4).to_string(index=False))
        from experiments.figures import FIGURES
        saved = pd.read_csv(tables_dir / f"{name}.csv")  # the table as saved, as `python run.py figures` draws it
        for figure_name, figure in FIGURES[name](saved, RUNS_DIR / "cvae" / "screen").items():
            figures_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(figures_dir / f"{name}_{figure_name}.png", facecolor="#fcfcfb")
        print(f"wrote reports/tables/{name}.csv and its figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
