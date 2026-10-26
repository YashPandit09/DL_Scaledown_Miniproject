"""The evaluation harness: any method on the same test rooms, with the same metrics.

python run.py baselines  evaluates B1, B2 and G0 on dataset v1 and writes
reports/tables/baselines.csv.

A sampler has sample(cond, n, rng) -> Samples (spacegen.baselines), so the CVAE methods M1 and
M2 are evaluated exactly like the baselines. Every method gets the same rooms and n raw samples
per room:
  - the rooms are the diversity-reference rooms of the Set A test split, then further test
    rooms drawn at random;
  - G0 runs single attempts, so its raw samples are attempts and its raw valid rate is its
    acceptance per attempt. Its outputs are valid by construction, so overlap and
    reachability are left empty for it and its cost excludes the re-check done here.

Summary columns (one row per method):
  rvr                 raw valid rate: valid / raw samples, with no filtering
  mean_overlap        summed pairwise overlap per raw sample (m^2)
  reachability        reachable / items needing access, per raw sample
  quality             mean rule score of the valid samples
  valid_per_room      valid samples per room, out of n; rooms_3_valid: share with at least 3
  diversity           mean diversity of the valid samples over the diversity rooms (m)
  diversity_ratio     that, over the G0 reference's diversity on the same rooms
  attempts_per_valid, ms_per_valid   cost per valid layout; seconds_per_room: latency for n samples
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spacegen.baselines import GeneratorBaseline, StatisticalBaseline, UniformBaseline, load_baseline_config
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch, load_layouts
from spacegen.generator import Condition, load_generator_config
from spacegen.metrics import diversity, score_layouts
from spacegen.paths import DATA_DIR, REPORTS_DIR
from spacegen.rules import Rules, load_rules


@dataclass(frozen=True)
class EvaluationConfig:
    rooms: int  # test rooms per method
    samples: int  # raw samples per room


def load_evaluation_config(path: Path = DEFAULT_CONFIG) -> EvaluationConfig:
    return EvaluationConfig(**load_config(path)["evaluation"])


def evaluate(sampler, conditions: list[Condition], n: int, rng: np.random.Generator, catalog: RoomCatalog,
             rules: Rules, prefiltered: bool = False, extra=None) -> pd.DataFrame:
    """One row per room: attempts, layouts returned, valid ones, summed metrics, diversity, seconds.
    `extra(valid layouts, their quality)` returns further columns for the room, not timed (used
    for the quality of the top 3 the pipeline would show)."""
    rows = []
    for room, cond in enumerate(conditions):
        start = time.perf_counter()
        samples = sampler.sample(cond, n, rng)
        sampling = time.perf_counter() - start
        start = time.perf_counter()
        scores = score_layouts(samples.layouts, catalog, rules)
        checking = time.perf_counter() - start
        valid = scores["valid"].to_numpy(dtype=bool)
        kept = [lay for lay, ok in zip(samples.layouts, valid) if ok]
        rows.append({"room": room, "attempts": samples.attempts, "returned": len(samples.layouts),
                     "valid": int(valid.sum()), "overlap": scores["overlap"].sum(),
                     "reachability": scores["reachability"].sum(), "quality": scores.loc[valid, "quality"].sum(),
                     "diversity": diversity(kept), "seconds": sampling + (0.0 if prefiltered else checking),
                     **({} if extra is None else extra(kept, scores.loc[valid, "quality"].to_numpy(dtype=float)))})
    return pd.DataFrame(rows)


def summarize(name: str, rooms: pd.DataFrame, reference: pd.Series, prefiltered: bool = False) -> dict:
    """The summary row of one method. `reference` is the G0 reference diversity per room index
    (the diversity rooms); rooms missing from it do not count towards diversity."""
    valid, attempts, returned = rooms["valid"].sum(), rooms["attempts"].sum(), rooms["returned"].sum()
    both = rooms.set_index("room")["diversity"].dropna().index.intersection(reference.dropna().index)
    method_diversity = rooms.set_index("room").loc[both, "diversity"].astype(float).mean()
    return {
        "method": name, "rooms": len(rooms), "samples": int(attempts),
        "rvr": valid / attempts,
        "mean_overlap": np.nan if prefiltered else rooms["overlap"].sum() / returned,
        "reachability": np.nan if prefiltered else rooms["reachability"].sum() / returned,
        "quality": rooms["quality"].sum() / valid if valid else np.nan,
        "valid_per_room": rooms["valid"].mean(),
        "rooms_3_valid": (rooms["valid"] >= 3).mean(),
        "diversity": method_diversity,
        "diversity_ratio": method_diversity / reference.loc[both].mean() if len(both) else np.nan,
        "diversity_rooms": len(both),
        "attempts_per_valid": attempts / valid if valid else np.nan,
        "ms_per_valid": 1000 * rooms["seconds"].sum() / valid if valid else np.nan,
        "seconds_per_room": rooms["seconds"].mean(),
    }


def evaluation_rooms(data_dir: Path, n_rooms: int, rng: np.random.Generator, catalog: RoomCatalog) -> tuple[list, pd.Series]:
    """The evaluation rooms (diversity rooms first, then random Set A test rooms) and the G0
    reference diversity of each diversity room, indexed by its position in the list."""
    set_a = load_layouts(data_dir / "set_a.npz")
    with np.load(data_dir / "splits.npz") as splits:
        test_rows = splits["set_a_test"]
    reference_info = pd.read_csv(data_dir / "diversity_reference.csv")
    reference_layouts = load_layouts(data_dir / "diversity_reference.npz")
    sources = list(dict.fromkeys(reference_info["source"].tolist()))[:n_rooms]
    others = rng.permutation(np.setdiff1d(test_rows, sources))[: max(n_rooms - len(sources), 0)]
    rows = sources + others.tolist()
    reference = pd.Series({position: diversity([reference_layouts.layout(i, catalog)
                                                for i in np.flatnonzero(reference_info["source"].to_numpy() == source)])
                           for position, source in enumerate(sources)}, dtype=float)
    return [Condition.of(set_a.layout(int(row), catalog), catalog) for row in rows], reference


def baseline_samplers(data_dir: Path, catalog: RoomCatalog, rules: Rules) -> list[tuple[str, object, bool]]:
    """(name, sampler, prefiltered) for B1, B2 (fitted on the Set A training split) and G0 (single
    attempts, so its raw valid rate is its acceptance per attempt)."""
    set_a = load_layouts(data_dir / "set_a.npz")
    with np.load(data_dir / "splits.npz") as splits:
        train: LayoutBatch = set_a.subset(splits["set_a_train"])
    return [("B1", UniformBaseline(catalog), False),
            ("B2", StatisticalBaseline(catalog, load_baseline_config()).fit(train), False),
            ("G0", generator_sampler(catalog, rules), True)]


def generator_sampler(catalog: RoomCatalog, rules: Rules) -> GeneratorBaseline:
    """G0 with single attempts: every attempt is one raw sample."""
    return GeneratorBaseline(catalog, rules, dataclasses.replace(load_generator_config(), attempts=1))


def evaluate_baselines(data_dir: Path, config: EvaluationConfig, seed: int, catalog: RoomCatalog,
                       rules: Rules, log=print) -> pd.DataFrame:
    """B1, B2 and G0 on the same rooms, drawing from one generator in that order; one summary row each."""
    rng = np.random.default_rng(seed)
    conditions, reference = evaluation_rooms(data_dir, config.rooms, rng, catalog)
    rows = []
    for name, sampler, prefiltered in baseline_samplers(data_dir, catalog, rules):
        start = time.perf_counter()
        rooms = evaluate(sampler, conditions, config.samples, rng, catalog, rules, prefiltered)
        rows.append(summarize(name, rooms, reference, prefiltered))
        log(f"{name}: {time.perf_counter() - start:.0f} s, RVR {rows[-1]['rvr']:.1%}")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the baselines B1, B2 and G0 on dataset v1.")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "tables" / "baselines.csv")
    args = parser.parse_args(argv)
    raw = load_config()
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    table = evaluate_baselines(data_dir, load_evaluation_config(), raw["seed"], load_room_catalog("living_room"),
                               load_rules())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out, index=False, lineterminator="\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
