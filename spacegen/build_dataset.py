"""The dataset build: every set the experiments use, from one seed.

python run.py data  writes data/<version>/ (data/ is not committed; rebuild it instead):

  set_a.npz, set_a.csv, set_a_attempts.csv  good layouts for the CVAE, per-layout info, attempt log
  set_b.npz, set_b.csv                      labelled layouts for the evaluator
  splits.npz                                train/validation/test rows of Set A and Set B
  held_out_<region>.npz, .csv               held-out test sets (E10)
  diversity_reference.npz, .csv             generator layouts for the diversity ratio (E1);
                                            info["source"] is the room's row in Set A
  calibration.csv                           the rooms that calibrate f_max
  metadata.json                             seed, configs, git commit, counts, labels, f_max, hashes

and figures to reports/figures/ (redraw them with --figures-only). Each part draws from its own
random stream spawned from the seed, so changing the size of one part leaves the others as
they were.

f_max, the crowding limit above which a request is turned away, is calibrated on rooms spread
evenly over the footprint ratio, each optional item present with calibration_items whatever the
room's size: a logistic fit of "the generator furnished the room within its attempts" against
the ratio gives the ratio at which half of the rooms can still be furnished (0.38).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import load_layouts, save_layouts
from spacegen.generator import (GeneratorConfig, generate_layout, generate_set_a, load_generator_config,
                                sample_condition)
from spacegen.paths import DATA_DIR, REPORTS_DIR
from spacegen.perturb import SetBConfig, generate_set_b, label_summary, load_set_b_config
from spacegen.provenance import config_files, git_state
from spacegen.rules import Rules, footprint_ratio, load_rules
from spacegen.splits import (SplitConfig, diversity_reference, generate_held_out, leaks, load_split_config,
                             split_indices, training_rooms)

RATIO_BINS = np.round(np.arange(0.05, 0.476, 0.025), 3)  # footprint-ratio bins for calibrating f_max


@dataclass(frozen=True)
class DatasetConfig:
    version: str
    set_a: int  # layouts
    set_b: int  # layouts
    calibration_rooms: int
    calibration_items: float  # chance of each optional item in a calibration room


def load_dataset_config(path: Path = DEFAULT_CONFIG) -> DatasetConfig:
    return DatasetConfig(**load_config(path)["dataset"])


def build_dataset(out_dir: Path, seed: int, sizes: DatasetConfig, catalog: RoomCatalog, rules: Rules,
                  generator: GeneratorConfig, set_b: SetBConfig, split: SplitConfig,
                  figures_dir: Path | None = None, log=print) -> dict:
    """Generate, check and write every set; returns the metadata (also written to metadata.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["set_a", "set_b", "splits", "diversity", "calibration", *split.held_out]
    streams = {name: np.random.default_rng(s) for name, s in zip(names, np.random.SeedSequence(seed).spawn(len(names)))}
    rooms = training_rooms(rules.rooms[catalog.room_type], split)
    seconds = {}

    def timed(name, make):
        start = time.perf_counter()
        result = make()
        seconds[name] = round(time.perf_counter() - start, 1)
        log(f"{name}: {seconds[name]} s")
        return result

    set_a = timed("set_a", lambda: generate_set_a(sizes.set_a, streams["set_a"], catalog, rules, generator, rooms))
    labelled = timed("set_b", lambda: generate_set_b(sizes.set_b, streams["set_b"], catalog, rules, generator,
                                                     set_b, rooms))
    for name, batch in (("Set A", set_a.layouts), ("Set B", labelled.layouts)):
        found = leaks(batch, split)
        if any(found.values()):
            raise RuntimeError(f"{name} has rooms in held-out regions: {found}")
    splits = {f"{name}_{part}": rows
              for name, batch in (("set_a", set_a.layouts), ("set_b", labelled.layouts))
              for part, rows in split_indices(len(batch), split.fractions, streams["splits"]).items()}
    held_out = {name: timed(name, lambda name=name: generate_held_out(
                    name, split.held_out_rooms, streams[name], catalog, rules, generator, split))
                for name in split.held_out}
    test_rows = splits["set_a_test"]
    diversity = timed("diversity", lambda: diversity_reference(
        set_a.layouts.subset(test_rows), split.diversity_rooms, split.diversity_layouts, streams["diversity"],
        catalog, rules, generator))
    diversity.info["source"] = test_rows[diversity.info["source"]]  # rows of Set A, not of its test split
    calibration = timed("calibration", lambda: calibration_rooms(
        sizes.calibration_rooms, streams["calibration"], catalog, rules, generator, sizes.calibration_items))
    f_max, coefficients = fit_f_max(calibration)

    def table(frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")

    save_layouts(out_dir / "set_a.npz", set_a.layouts)
    table(set_a.info, "set_a")
    table(set_a.attempts, "set_a_attempts")
    save_layouts(out_dir / "set_b.npz", labelled.layouts)
    table(labelled.info, "set_b")
    np.savez_compressed(out_dir / "splits.npz", **splits)
    for name, result in held_out.items():
        save_layouts(out_dir / f"held_out_{name}.npz", result.layouts)
        table(result.info, f"held_out_{name}")
    save_layouts(out_dir / "diversity_reference.npz", diversity.layouts)
    table(diversity.info, "diversity_reference")
    table(calibration, "calibration")

    attempts = set_a.attempts
    metadata = {
        "version": sizes.version,
        "seed": seed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "configs": config_files(),
        "counts": {"set_a": len(set_a.layouts), "set_b": len(labelled.layouts),
                   **{f"held_out_{name}": len(result.layouts) for name, result in held_out.items()},
                   "diversity_reference": len(diversity.layouts), "calibration_rooms": len(calibration),
                   **{name: len(rows) for name, rows in splits.items()}},
        "set_a": {"rooms_tried": int(attempts["room"].nunique()), "attempts": len(attempts),
                  "kept_per_attempt": round(float((attempts["outcome"] == "valid").mean()), 4),
                  "rooms_dropped": int(attempts["room"].nunique() - len(set_a.layouts)),
                  "styles": _shares(set_a.info["style"]), "quality_mean": round(float(set_a.info["quality"].mean()), 4)},
        "set_b": {"valid": round(float(labelled.info["valid"].mean()), 4),
                  "per_type": {kind: {"samples": int(row.samples), "valid": round(float(row.valid), 4)}
                               for kind, row in label_summary(labelled.info).iterrows()}},
        "held_out": {name: {"styles": _shares(result.info["style"])} for name, result in held_out.items()},
        "f_max": {"calibrated": None if f_max is None else round(f_max, 4), "in_rules_yaml": rules.f_max,
                  "logistic": [round(float(c), 4) for c in coefficients],
                  "furnished": round(float(calibration["furnished"].mean()), 4)},
        "seconds": seconds,
    }
    metadata["files"] = file_hashes(out_dir)
    metadata["hash"] = dataset_hash(metadata["files"])
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")

    if figures_dir is not None:
        make_figures(out_dir, figures_dir, split)
    return metadata


# --------------------------------------------------------------------------- f_max

def calibration_rooms(n: int, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                      config: GeneratorConfig, chance: float, pool: int = 200_000) -> pd.DataFrame:
    """About n rooms spread evenly over the footprint-ratio bins, and whether the generator furnishes each.

    Candidates come from the full room ranges with each optional item present with `chance`;
    each bin keeps at most n / bins of them (the most crowded bins may hold fewer).
    """
    crowded = dataclasses.replace(config, optional_probability=(chance, chance))
    quota = int(np.ceil(n / (len(RATIO_BINS) - 1)))
    chosen: list[list] = [[] for _ in range(len(RATIO_BINS) - 1)]
    for _ in range(pool):
        cond = sample_condition(rng, catalog, rules, crowded)
        ratio = footprint_ratio(cond.furniture_area(catalog), cond.width, cond.depth, rules.door)
        k = int(np.searchsorted(RATIO_BINS, ratio, side="right")) - 1
        if 0 <= k < len(chosen) and len(chosen[k]) < quota:
            chosen[k].append((cond, ratio))
            if sum(map(len, chosen)) >= n:
                break
    rows = []
    for cond, ratio in (pair for group in chosen for pair in group):
        result = generate_layout(cond, rng, catalog, rules, config)
        rows.append({"width": cond.width, "depth": cond.depth, "area": cond.area, "items": len(cond.items),
                     "furniture": cond.furniture_area(catalog), "ratio": ratio,
                     "furnished": result.layout is not None, "attempts": len(result.attempts)})
    return pd.DataFrame(rows)


def fit_f_max(calibration: pd.DataFrame) -> tuple[float | None, np.ndarray]:
    """The footprint ratio where a logistic fit of `furnished` against `ratio` crosses one half.

    Returns None (and the coefficients) when the fit does not fall with the ratio, for example
    when every calibration room was furnished.
    """
    x, y = calibration["ratio"].to_numpy(), calibration["furnished"].to_numpy(dtype=float)
    scale = x.std() or 1.0  # fit on a standardized ratio, so both coefficients are of order 1
    z = (x - x.mean()) / scale

    def loss(b):
        logit = b[0] + b[1] * z
        p = 1 / (1 + np.exp(-logit))
        return np.sum(np.logaddexp(0, logit) - y * logit), np.array([np.sum(p - y), np.sum((p - y) * z)])

    b = minimize(loss, x0=np.zeros(2), jac=True, method="BFGS").x
    coefficients = np.array([b[0] - b[1] * x.mean() / scale, b[1] / scale])  # back to the raw ratio
    if not 0 < y.mean() < 1 or coefficients[1] >= 0:
        return None, coefficients
    return float(-coefficients[0] / coefficients[1]), coefficients


# --------------------------------------------------------------------------- hashes

def file_hashes(directory: Path) -> dict[str, str]:
    """SHA-256 of each data file's contents: the arrays of an .npz (its zip timestamps change
    between builds), the bytes of anything else. metadata.json is left out."""
    hashes = {}
    for path in sorted(directory.iterdir()):
        if path.name == "metadata.json" or not path.is_file():
            continue
        digest = hashlib.sha256()
        if path.suffix == ".npz":
            with np.load(path) as data:
                for key in sorted(data.files):
                    array = data[key]
                    digest.update(f"{key}|{array.dtype}|{array.shape}|".encode())
                    digest.update(np.ascontiguousarray(array).tobytes())
        else:
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        hashes[path.name] = digest.hexdigest()
    return hashes


def dataset_hash(files: dict[str, str]) -> str:
    """One hash for the whole dataset, stored with every checkpoint trained on it."""
    return hashlib.sha256("".join(f"{name}:{digest}\n" for name, digest in sorted(files.items())).encode()).hexdigest()


def _shares(values: pd.Series) -> dict[str, float]:
    return {str(k): round(float(v), 4) for k, v in values.value_counts(normalize=True).sort_index().items()}


def make_figures(data_dir: Path, figures_dir: Path, split: SplitConfig) -> list[Path]:
    """Draw the dataset figures from the files of a built dataset (no regeneration needed)."""
    from spacegen.viz import SURFACE, plot_dataset_rooms, plot_f_max, plot_rejection, plot_set_b_labels

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    attempts_cap = yaml.safe_load(metadata["configs"]["default.yaml"])["generator"]["attempts"]
    f_max = metadata["f_max"]
    figures = {
        "rooms": plot_dataset_rooms(load_layouts(data_dir / "set_a.npz"),
                                    {name: load_layouts(data_dir / f"held_out_{name}.npz") for name in split.held_out},
                                    split.held_out),
        "set_b": plot_set_b_labels(pd.read_csv(data_dir / "set_b.csv")),
        "rejection": plot_rejection(pd.read_csv(data_dir / "set_a_attempts.csv"), attempts_cap),
        "f_max": plot_f_max(pd.read_csv(data_dir / "calibration.csv"), f_max["calibrated"], f_max["logistic"],
                            f_max["in_rules_yaml"]),
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        paths.append(figures_dir / f"dataset_{metadata['version']}_{name}.png")
        fig.savefig(paths[-1], facecolor=SURFACE)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the dataset (Set A, Set B, splits, held-out sets, f_max).")
    parser.add_argument("--out", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--seed", type=int, default=None, help="default: the seed in configs/default.yaml")
    parser.add_argument("--figures-only", action="store_true", help="redraw the figures of an existing build")
    args = parser.parse_args(argv)
    sizes = load_dataset_config()
    seed = load_config()["seed"] if args.seed is None else args.seed
    out = args.out or DATA_DIR / sizes.version
    if args.figures_only:
        for path in make_figures(out, REPORTS_DIR / "figures", load_split_config()):
            print(f"wrote {path}")
        return 0
    metadata = build_dataset(out, seed, sizes, load_room_catalog("living_room"), load_rules(), load_generator_config(),
                             load_set_b_config(), load_split_config(), figures_dir=REPORTS_DIR / "figures")
    print(json.dumps({key: metadata[key] for key in ("counts", "set_a", "set_b", "held_out", "f_max", "seconds", "hash")},
                     indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
