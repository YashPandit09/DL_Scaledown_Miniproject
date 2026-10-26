"""Gate 2, the configuration freeze: three seeds of the default and of
the shortlisted changes, judged on M2 over Set A validation rooms.

python run.py gate2    (trains the missing runs on the GPU, then evaluates on the CPU; about an hour)

Shortlist: in the screening (one seed each) three changes raised M1's raw valid rate the most:
Tanh (E3a, 22% against 12%), the overlap term (E2, 20%) and MAE (E2, 17%). The screening's M1
check sampled test rooms, so it only shortlists; the choice is made here on 200 rooms of the Set A
validation split, which the headline experiments never use.

Each candidate is trained with the configured seed and the next two. The first is the screening
run in runs/cvae/screen/, the others go to runs/cvae/gate2/. Every model samples the same
validation rooms with the same draws (experiments.headline; per-room rows cached in
runs/gate2/<run>/), as M1 and as M2, the deployed method: raw valid rate, quality, quality of the
evaluator's top 3, and diversity (mean over the rooms with at least two valid layouts).

Rule (Gate 2): a candidate replaces the default if its mean over the seeds beats the default's by
more than the spread (the larger of the two standard deviations) on M2's raw valid rate or on its
top-3 quality, and falls short by no more than the spread on the other. Among several, the larger
gain in top-3 quality wins. If two or more qualify, the best two are also tried together (three
more seeds) and the combination competes under the same rule.

Writes reports/tables/gate2_runs.csv (one row per run), reports/tables/gate2.csv (mean and
standard deviation per candidate, with the verdict), reports/figures/gate2_candidates.png and
configs/frozen.yaml (configs/default.yaml with the winning overrides).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.headline import Headline
from experiments.screening import OVERLAP, run_name
from spacegen.catalog import load_room_catalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.evaluate import EvaluationConfig
from spacegen.latent_opt import load_latent_opt_config
from spacegen.paths import CONFIG_DIR, DATA_DIR, OUTPUT_ROOT, REPO_ROOT, REPORTS_DIR, RUNS_DIR
from spacegen.pipeline import load_cvae_run, load_evaluator_run, load_pipeline_config
from spacegen.provenance import dataset_hash
from spacegen.rules import load_rules
from spacegen.seed import set_seed
from spacegen.train_cvae import load_configs, train_cvae

CANDIDATES: dict[str, list[str]] = {"default": [], "tanh": ["cvae.activation=tanh"],
                                    "overlap term": [OVERLAP], "mae": ["cvae.position_loss=mae"]}
SEEDS = 3
ROOMS = 200
DECIDING = ("m2_rvr", "m2_top3")  # the rule's two metrics; the gain is measured on the second


def run_dir(overrides: list[str], seed: int, first_seed: int, screen_dir: Path, gate2_dir: Path) -> Path:
    """The training run of a candidate and seed: the screening run for the first seed."""
    name = run_name(overrides)
    return screen_dir / name if seed == first_seed else gate2_dir / f"{name}__seed-{seed}"


def run_gate2(data_dir: Path, seed: int, device: str, candidates: dict[str, list[str]] = CANDIDATES,
              seeds: int = SEEDS, rooms: int = ROOMS, samples: int = 64, base: list[str] = (),
              screen_dir: Path = RUNS_DIR / "cvae" / "screen", gate2_dir: Path = RUNS_DIR / "cvae" / "gate2",
              cache_dir: Path = RUNS_DIR / "gate2", evaluator=None, raster=None, rank_device: str = "cpu",
              train: bool = True, log=print) -> pd.DataFrame:
    """Train what is missing, then one row per candidate and seed with its validation metrics.
    `base` overrides apply to every candidate (the tests use it for tiny models)."""
    catalog, rules = load_room_catalog("living_room"), load_rules()
    latent, pipeline = load_latent_opt_config(), load_pipeline_config()
    config = EvaluationConfig(rooms=rooms, samples=samples)
    rows = []
    for name, overrides in candidates.items():
        for run_seed in range(seed, seed + seeds):
            run = run_dir([*base, *overrides], run_seed, seed, screen_dir, gate2_dir)
            if not (run / "summary.json").exists():
                if not train:
                    log(f"gate2: {run.name} is not trained, skipped")
                    continue
                log(f"gate2: training {name}, seed {run_seed} -> {run}")
                model, training = load_configs(overrides=[*base, *overrides])
                train_cvae(data_dir, run, run_seed, device, model, training, catalog, log=lambda message: None)
            inputs = {"dataset_hash": dataset_hash(data_dir), "cvae": _file_hash(run / "model.pt"),
                      "evaluator": None if evaluator is None else _model_hash(evaluator)}
            headline = Headline(data_dir, cache_dir / f"{run_name([*base, *overrides])}__seed-{run_seed}", seed,
                                config, load_cvae_run(run), catalog, rules, latent, pipeline, evaluator, raster,
                                rank_device, inputs, log=log)
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            row = {"candidate": name, "seed": run_seed, "run": str(run.relative_to(run.parents[1])),
                   "val_loss": summary["selected_validation_loss"],
                   "position_error_mean": summary["position_error_mean"],
                   "active_units": summary["active_units"], "epochs": summary["epochs_run"]}
            for method in ("M1", "M2"):
                row.update(_metrics(headline, method))
            rows.append(row)
    return pd.DataFrame(rows)


def _metrics(headline: Headline, method: str) -> dict:
    rooms = headline.per_room("validation", method)
    valid = rooms["valid"].sum()
    prefix = method.lower()
    return {f"{prefix}_rvr": valid / rooms["attempts"].sum(),
            f"{prefix}_quality": rooms["quality"].sum() / valid if valid else np.nan,
            f"{prefix}_top3": rooms["top_evaluator"].mean(), f"{prefix}_top3_rule": rooms["top_rule"].mean(),
            f"{prefix}_diversity": rooms["diversity"].mean(),
            f"{prefix}_overlap": rooms["overlap"].sum() / rooms["returned"].sum(),
            f"{prefix}_seconds_per_room": rooms["seconds"].mean()}


def summarize_candidates(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over the seeds of every metric, one row per candidate."""
    numeric = runs.drop(columns=["seed", "run"]).groupby("candidate", sort=False)
    table = numeric.mean().add_suffix("_mean").join(numeric.std(ddof=1).add_suffix("_std"))
    table.insert(0, "seeds", runs.groupby("candidate", sort=False).size())
    return table


def verdicts(summary: pd.DataFrame, baseline: str = "default") -> dict[str, tuple[bool, float, str]]:
    """For each candidate: (qualifies, gain in top-3 quality, reason), under the Gate 2 rule."""
    base = summary.loc[baseline]
    result = {}
    for name, row in summary.iterrows():
        if name == baseline:
            continue
        better, worse, notes = [], [], []
        for metric in DECIDING:
            spread = max(row[f"{metric}_std"], base[f"{metric}_std"])
            difference = row[f"{metric}_mean"] - base[f"{metric}_mean"]
            better.append(difference > spread)
            worse.append(-difference > spread)
            notes.append(f"{metric} {difference:+.4f} (spread {spread:.4f})")
        qualifies = any(better) and not any(worse)
        gain = row[f"{DECIDING[1]}_mean"] - base[f"{DECIDING[1]}_mean"]
        result[name] = (qualifies, gain, ("qualifies: " if qualifies else "does not qualify: ") + ", ".join(notes))
    return result


def choose(summary: pd.DataFrame, baseline: str = "default") -> str:
    """The winner under the Gate 2 rule: the qualifying candidate with the largest top-3 gain, or the default."""
    qualified = {name: gain for name, (ok, gain, _) in verdicts(summary, baseline).items() if ok}
    return max(qualified, key=qualified.get) if qualified else baseline


def write_frozen(overrides: list[str], note: str, source: Path = DEFAULT_CONFIG,
                 target: Path = CONFIG_DIR / "frozen.yaml") -> Path:
    """configs/default.yaml with the overrides applied in place (comments kept) and a header."""
    lines = source.read_text(encoding="utf-8").splitlines()
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        start = lines.index(next(line for line in lines if re.match(rf"^{section}:", line)))
        for i in range(start + 1, len(lines)):
            if re.match(r"^\S", lines[i]):
                raise ValueError(f"{key} not found in {source}")
            match = re.match(rf"^(  {field}:\s*)(\S+)(\s*)(#\s*)?(.*)$", lines[i])
            if match:
                prefix, old, gap, _, comment = match.groups()
                lines[i] = f"{prefix}{value}{gap or ' '}# (Gate 2: was {old}) {comment}".rstrip()
                break
    header = ["# Frozen configuration (Gate 2, " + date.today().isoformat() + "): configs/default.yaml with the",
              "# choice of the Gate 2 review (reports/gate2.md): " + note,
              "# The final runs on three seeds and the headline experiments use this file."]
    target.write_text("\n".join(header + lines) + "\n", encoding="utf-8", newline="\n")
    if load_config(target) != _applied(load_config(source), overrides):
        raise RuntimeError(f"{target} does not match {source} with {overrides}")
    return target


def _applied(raw: dict, overrides: list[str]) -> dict:
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        raw[section][field] = yaml.safe_load(value)
    return raw


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate 2: three seeds of the shortlisted CVAE settings.")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="training and the evaluator's ranking; the samplers always run on the CPU")
    parser.add_argument("--evaluator", type=Path, default=RUNS_DIR / "evaluator" / "e9a")
    parser.add_argument("--no-train", action="store_true", help="evaluate the runs that exist, train nothing")
    args = parser.parse_args(argv)
    raw = load_config()
    seed = raw["seed"]
    set_seed(seed)
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    evaluator, raster = load_evaluator_run(args.evaluator, args.device)
    candidates = dict(CANDIDATES)

    def run() -> tuple[pd.DataFrame, pd.DataFrame]:
        table = run_gate2(data_dir, seed, args.device, candidates, evaluator=evaluator, raster=raster,
                          rank_device=args.device, train=not args.no_train)
        return table, summarize_candidates(table)

    runs, summary = run()
    qualified = sorted(((gain, name) for name, (ok, gain, _) in verdicts(summary).items() if ok), reverse=True)
    if len(qualified) >= 2:  # the best two together, under the same rule
        first, second = qualified[0][1], qualified[1][1]
        candidates[f"{first} + {second}"] = candidates[first] + candidates[second]
        runs, summary = run()
    chosen = choose(summary)
    found = verdicts(summary)
    summary["verdict"] = [("chosen; " if name == chosen else "") + (found[name][2] if name in found else "the reference")
                          for name in summary.index]
    tables, figures = REPORTS_DIR / "tables", REPORTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    runs.round(5).to_csv(tables / "gate2_runs.csv", index=False, lineterminator="\n")
    summary.round(5).to_csv(tables / "gate2.csv", lineterminator="\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(runs.drop(columns=["run"]).round(4).to_string(index=False))
        columns = [f"{m}_{s}" for m in ("val_loss", "m1_rvr", "m2_rvr", "m2_quality", "m2_top3", "m2_diversity")
                   for s in ("mean", "std")]
        print(summary[columns].round(4).to_string())
    for name, (ok, gain, reason) in found.items():
        print(f"{name}: {reason}")
    from experiments.figures import FIGURES

    figures.mkdir(parents=True, exist_ok=True)
    saved = pd.read_csv(tables / "gate2_runs.csv")  # the table as saved, as `python run.py figures` draws it
    for figure_name, figure in FIGURES["gate2"](saved, None).items():
        figure.savefig(figures / f"gate2_{figure_name}.png", facecolor="#fcfcfb")
    if not args.no_train:
        note = (f"{chosen} ({', '.join(candidates[chosen])})." if candidates[chosen]
                else "the default, unchanged (no candidate passed the rule).")
        # a regeneration into another folder (SPACEGEN_OUTPUT) must not touch the repository's frozen config
        target = (CONFIG_DIR if OUTPUT_ROOT == REPO_ROOT else OUTPUT_ROOT / "configs") / "frozen.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"wrote {write_frozen(candidates[chosen], note, target=target)}")
    print(f"chosen: {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
