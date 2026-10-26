"""E12: completion around a pinned item (the user fixes one piece of furniture).

python run.py e12 [--cvae RUN ...] [--rooms N]

A request is a test room of the diversity-reference set with one of its items pinned. The pin is
the position and facing that item has in one of the room's generator reference layouts, so a
valid completion is known to exist. Every method gets 64 samples per request:

  M2      CVAE with latent optimization: the pin term pulls the item to its spot, then it is snapped
  M1      the CVAE's samples with the item snapped onto its spot
  G0-pin  generator layouts with the item moved onto its spot
  B1, B2  the item fixed first, the others sampled (B2 draws them again while they overlap it)

For each request the methods (and the CVAE seeds) sample in turn, so their timings compare even
when the laptop changes speed (experiments.headline explains why). M1 and M2 share their draws.

Results are split by the pinned item: the raw valid rate after the snap, the share of requests
with at least one valid layout, the quality of the valid layouts, how far the item
was from its pin before the snap (M1, M2, G0-pin), and the cost per valid layout. M1 and M2 are
the mean and standard deviation over the CVAE seeds.

Writes reports/tables/e12.csv and reports/figures/e12_pinned.png; the per-request rows are kept
in runs/e12/rows.csv and reused while the inputs stay the same (--fresh recomputes them).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.headline import _file_hash, _keep_or_clear
from spacegen.baselines import StatisticalBaseline, UniformBaseline, load_baseline_config
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import load_config
from spacegen.dataset import load_layouts
from spacegen.evaluate import evaluate, generator_sampler
from spacegen.generator import Condition
from spacegen.latent_opt import LatentOptConfig, load_latent_opt_config
from spacegen.models.cvae import CVAE
from spacegen.paths import CHECKPOINTS_DIR, DATA_DIR, REPORTS_DIR, RUNS_DIR
from spacegen.pins import Pin, PinnedSampler
from spacegen.pipeline import CVAESampler, LatentOptSampler, load_cvae_run
from spacegen.provenance import dataset_hash, write_run_record
from spacegen.rules import Rules, load_rules

METHODS = ("B1", "B2", "G0-pin", "M1", "M2")
MEASURES = ("rvr", "requests_with_valid", "quality", "displacement_m", "ms_per_valid", "seconds_per_request")


def pinned_requests(data_dir: Path, catalog: RoomCatalog, per_item: int, seed: int) -> list[tuple[str, Condition, dict[str, Pin]]]:
    """(pinned item, room, pins): for every slot, the first `per_item` diversity-reference rooms that
    have the item, pinned where it stands in one of that room's reference layouts (drawn at random)."""
    info = pd.read_csv(data_dir / "diversity_reference.csv")
    reference = load_layouts(data_dir / "diversity_reference.npz")
    source = info["source"].to_numpy()
    rng = np.random.default_rng([seed, 12])
    chosen = {room: int(rng.choice(np.flatnonzero(source == room))) for room in dict.fromkeys(source.tolist())}
    requests = []
    for slot in catalog.slots:
        found = 0
        for row in chosen.values():
            layout = reference.layout(row, catalog)
            if not layout.mask[slot.index]:
                continue
            pin = Pin(float(layout.center[slot.index, 0]), float(layout.center[slot.index, 1]), int(layout.rot[slot.index]))
            requests.append((slot.name, Condition.of(layout, catalog), {slot.name: pin}))
            found += 1
            if found == per_item:
                break
    return requests


def run_requests(requests: list, models: dict[str, CVAE], b2: StatisticalBaseline, n: int, seed: int,
                 catalog: RoomCatalog, rules: Rules, latent: LatentOptConfig, log=print) -> pd.DataFrame:
    """One row per request and sampler. `models` maps a run name to its CVAE; the samplers are
    B1, B2, G0-pin and, per run, "M1|run" and "M2|run"."""
    def samplers_for(pins: dict[str, Pin]) -> dict:
        samplers = {"B1": PinnedSampler(UniformBaseline(catalog), pins, catalog, takes_pins=True),
                    "B2": PinnedSampler(b2, pins, catalog, takes_pins=True),
                    "G0-pin": PinnedSampler(generator_sampler(catalog, rules), pins, catalog)}
        for method in ("M1", "M2"):
            for run, cvae in models.items():
                samplers[f"{method}|{run}"] = (CVAESampler(cvae, catalog, pins=pins) if method == "M1"
                                               else LatentOptSampler(cvae, catalog, rules, latent, pins=pins))
        return samplers

    names = list(samplers_for({}))
    rngs = {name: np.random.default_rng([seed, 12, min(index, 3)]) for index, name in enumerate(names)}  # M1, M2 paired
    rows = []
    for number, (item, cond, pins) in enumerate(requests):
        for name, sampler in samplers_for(pins).items():
            row = evaluate(sampler, [cond], n, rngs[name], catalog, rules).iloc[0]
            moved = sampler.last_displacement if not getattr(sampler, "takes_pins", False) else np.zeros(0)
            rows.append({"item": item, "request": number, "sampler": name, "attempts": int(row["attempts"]),
                         "valid": int(row["valid"]), "quality": float(row["quality"]), "seconds": float(row["seconds"]),
                         "displacement": float(np.mean(moved)) if len(moved) else np.nan})
        if (number + 1) % 20 == 0 or number + 1 == len(requests):
            log(f"e12: {number + 1} of {len(requests)} requests")
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per pinned item (and "all") and method; M1 and M2 as mean and standard deviation over the runs."""
    def measures(group: pd.DataFrame) -> pd.Series:
        valid = group["valid"].sum()
        return pd.Series({"requests": len(group), "rvr": valid / group["attempts"].sum(),
                          "requests_with_valid": (group["valid"] > 0).mean(),
                          "quality": group["quality"].sum() / valid if valid else np.nan,
                          "displacement_m": group["displacement"].mean(),
                          "ms_per_valid": 1000 * group["seconds"].sum() / valid if valid else np.nan,
                          "seconds_per_request": group["seconds"].mean()})

    both = pd.concat([rows, rows.assign(item="all")], ignore_index=True)
    per_sampler = both.groupby(["item", "sampler"], sort=False).apply(measures, include_groups=False).reset_index()
    per_sampler["method"] = per_sampler["sampler"].str.split("|").str[0]
    groups = per_sampler.groupby(["item", "method"], sort=False)
    table = groups[list(MEASURES)].mean().join(groups[list(MEASURES)].std(ddof=1).add_suffix("_std"))
    table.insert(0, "requests", groups["requests"].first().astype(int))
    table.insert(0, "seeds", groups.size())
    return table.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E12: completion around a pinned item.")
    frozen = (sorted((RUNS_DIR / "cvae" / "frozen").glob("seed-*"))
              or sorted((CHECKPOINTS_DIR / "cvae" / "frozen").glob("seed-*")))  # a fresh clone has only the copies
    parser.add_argument("--cvae", type=Path, nargs="+", default=frozen or [RUNS_DIR / "cvae" / "default"])
    parser.add_argument("--rooms", type=int, default=60, help="requests per pinned item")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--fresh", action="store_true", help="recompute the cached per-request rows")
    args = parser.parse_args(argv)
    raw = load_config()
    seed = raw["seed"]
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    catalog, rules, latent = load_room_catalog("living_room"), load_rules(), load_latent_opt_config()
    cache = RUNS_DIR / "e12"
    inputs = {"dataset_hash": dataset_hash(data_dir), "cvae": [_file_hash(run / "model.pt") for run in args.cvae],
              "seed": seed, "rooms": args.rooms, "samples": args.samples}
    _keep_or_clear(cache, inputs, args.fresh)
    if not (cache / "rows.csv").exists():
        set_a = load_layouts(data_dir / "set_a.npz")
        with np.load(data_dir / "splits.npz") as splits:
            b2 = StatisticalBaseline(catalog, load_baseline_config()).fit(set_a.subset(splits["set_a_train"]))
        requests = pinned_requests(data_dir, catalog, args.rooms, seed)
        print(f"e12: {len(requests)} requests, {len(args.cvae)} CVAE run(s)")
        write_run_record(cache, seed, data_dir, task="E12, pinned furniture", cvae_runs=[str(r) for r in args.cvae],
                         rooms_per_item=args.rooms, samples=args.samples)
        rows = run_requests(requests, {run.name: load_cvae_run(run) for run in args.cvae}, b2, args.samples, seed,
                            catalog, rules, latent)
        rows.to_csv(cache / "rows.csv", index=False, lineterminator="\n")
    table = summarize(pd.read_csv(cache / "rows.csv"))
    tables, figures = REPORTS_DIR / "tables", REPORTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(tables / "e12.csv", index=False, lineterminator="\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[["item", "method", "seeds", "requests", *MEASURES]].round(3).to_string(index=False))
    from experiments.figures import FIGURES

    saved = pd.read_csv(tables / "e12.csv")  # the table as saved, as `python run.py figures` draws it
    for name, figure in FIGURES["e12"](saved, None).items():
        figure.savefig(figures / f"e12_{name}.png", facecolor="#fcfcfb")
    print("wrote reports/tables/e12.csv and reports/figures/e12_pinned.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
