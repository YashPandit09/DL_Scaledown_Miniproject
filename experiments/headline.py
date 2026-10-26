"""The headline experiments E1 (every method on the same test rooms), E8 and E10 (held-out rooms).

python run.py e1     B1, B2, G0, M1 and M2 on the E1 rooms: reports/tables/e1.csv and figures
python run.py e10    M1, M2 and G0 on the four test sets: reports/tables/e10.csv and its figure

--cvae RUN chooses the CVAE (default runs/cvae/default) and --evaluator RUN the ranker (default
runs/evaluator/e9a). Before Gate 2 both experiments ran on the default configuration to debug the
pipeline (the first pass). The headline numbers come from the three seeds of the frozen
configuration:

python run.py e1 --tag final --cvae runs/cvae/frozen/seed-0 runs/cvae/frozen/seed-1 runs/cvae/frozen/seed-2
python run.py e8 --tag final --cvae ...      (E8: latent-optimization steps, on the 200 diversity rooms)
python run.py e10 --tag final --cvae ... [--sets interpolation ...]

With several runs M1 and M2 are reported as the mean and standard deviation over the seeds (every
seed samples the same rooms with the same draws, so the spread is the training's); B1, B2 and G0
do not depend on the CVAE and run once. The tables are reports/tables/<experiment>_<tag>.csv.

E1   The evaluation protocol of spacegen.evaluate: the same 500 rooms and 64 raw samples per room
     for every method. B1, B2 and G0 draw from one generator in that order, as `python run.py
     baselines` does, so their rows reproduce reports/tables/baselines.csv (timings aside). M1 and
     M2 start from the same generator seed, so M2 repairs exactly M1's draws. Every sampler runs
     on the CPU, so the costs compare. Also measured: the mean rule quality of the diverse top 3
     of each room, with the valid layouts ranked by the CNN evaluator (what the pipeline shows),
     by the exact rule score (the reference) and in random order (no ranking). The ranking is
     not timed.
E10  M1, M2 and G0 on the E1 rooms (in distribution) and on the first 500 rooms of each held-out
     set: interpolation (22 to 26 m^2), unseen combination (above 32 m^2) and out of range (W 7 to
     8 m, D 6 to 7 m). G0, next to M1 and M2, shows how hard each set is by itself:
     room size alone moves the quality score, through circulation for example. The E10 table has no
     timing columns: its sets run one after another for a long time, and such timings do not compare
     (see Timings below).

Timings. The laptop's speed changes with its power and thermal state: the same M2 run was
measured 1.6 times faster on one evening than on the afternoon before, and within one E8 run the
machine slowed down by half. Methods timed one after another therefore cannot be compared. E1 and
E8 take their timing columns (seconds_per_room, ms_per_valid) from an interleaved timing pass
instead: on the first 100 E1 rooms every method (and every seed, and every step count of E8)
samples each room in turn, so a change of speed falls on all of them alike. The timings of the
long runs themselves stay in the columns ending in _run. --retime repeats the pass. Each command
also times a fixed workload before and after (speed_check, saved in speed.json).

Per-room rows are kept in runs/headline/<cvae run>/<set>/<method>.csv (with --tag:
runs/headline/<tag>/<cvae run>/..., and runs/headline/<tag>/baselines/... for B1, B2 and G0) and
reused while the inputs (inputs.json: dataset, models, seed, rooms, samples, latent-optimization
and top-3 settings) stay the same; --fresh recomputes them, which a change to the code needs. So
E10 takes the in-distribution rows from E1, and an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import load_config
from spacegen.dataset import load_layouts
from spacegen.evaluate import (EvaluationConfig, baseline_samplers, evaluate, evaluation_rooms, generator_sampler,
                               load_evaluation_config, summarize)
from spacegen.generator import Condition
from spacegen.latent_opt import LatentOptConfig, load_latent_opt_config
from spacegen.layout import Layout
from spacegen.models.cvae import CVAE
from spacegen.models.evaluator import Evaluator
from spacegen.paths import DATA_DIR, REPORTS_DIR, RUNS_DIR
from spacegen.pipeline import (CVAESampler, LatentOptSampler, PipelineConfig, load_cvae_run, load_evaluator_run,
                               load_pipeline_config, select_diverse)
from spacegen.provenance import dataset_hash, write_run_record
from spacegen.raster import RasterConfig, rasterize_layouts
from spacegen.rules import Rules, load_rules
from spacegen.seed import set_seed
from experiments.screening import E8_STEPS, run_latent_steps

SETS = ("in_distribution", "interpolation", "unseen_combination", "out_of_range")  # the E10 test sets
ROOM_SETS = (*SETS, "validation")  # Set A validation rooms, for choosing the frozen configuration
METHODS = ("B1", "B2", "G0", "M1", "M2")
MODEL_FREE = ("B1", "B2", "G0")  # do not depend on the CVAE: computed once for several runs
SAME_IN_EVERY_RUN = ("rooms", "samples", "area_m2", "items")  # set by the protocol: no spread over the seeds
E10_METHODS = ("M1", "M2", "G0")
ORDERS = {"evaluator": "quality_top3", "rule": "quality_top3_rule", "random": "quality_top3_random"}


class TopThree:
    """The diverse top k of a room's valid layouts in three orders: the CNN
    evaluator's score (what the pipeline shows), the exact rule score (the reference) and a random
    order (no ranking). Gives the mean rule quality of each top k; the evaluator's is NaN without
    an evaluator. The random order draws from its own generator, not the sampler's."""

    def __init__(self, config: PipelineConfig, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                 evaluator: Evaluator | None = None, raster: RasterConfig | None = None,
                 device: str | torch.device = "cpu"):
        self.config, self.rng, self.catalog, self.rules = config, rng, catalog, rules
        self.evaluator, self.raster, self.device = evaluator, raster, device

    def __call__(self, layouts: list[Layout], quality: np.ndarray) -> dict[str, float]:
        row = {f"top_{order}": np.nan for order in ORDERS}
        if not layouts:
            return row
        orders = {"rule": np.argsort(-quality, kind="stable"), "random": self.rng.permutation(len(layouts))}
        if self.evaluator is not None:
            with torch.no_grad():
                _, score = self.evaluator(rasterize_layouts(layouts, self.catalog, self.rules, self.raster,
                                                            self.device))
            orders["evaluator"] = np.argsort(-score.cpu().numpy(), kind="stable")
        for order, ranked in orders.items():
            picks, _ = select_diverse([layouts[i] for i in ranked], self.config.top_k, self.config.tau_div,
                                      self.config.relax, self.config.relax_times)
            row[f"top_{order}"] = float(quality[ranked[picks]].mean())
        return row


class Headline:
    """E1 and E10 for one CVAE, with the per-room rows of every method and test set cached in `cache`."""

    def __init__(self, data_dir: Path, cache: Path, seed: int, config: EvaluationConfig, cvae: CVAE,
                 catalog: RoomCatalog, rules: Rules, latent: LatentOptConfig, pipeline: PipelineConfig,
                 evaluator: Evaluator | None = None, raster: RasterConfig | None = None,
                 rank_device: str | torch.device = "cpu", inputs: dict | None = None, fresh: bool = False,
                 log=print, baseline_cache: Path | None = None):
        self.data_dir, self.cache, self.seed, self.config = data_dir, cache, seed, config
        self.baseline_cache = baseline_cache or cache  # B1, B2 and G0 (shared by several CVAE runs)
        self.cvae = cvae.cpu().eval()  # every sampler runs on the CPU, so the costs compare
        self.catalog, self.rules, self.latent, self.pipeline = catalog, rules, latent, pipeline
        self.evaluator, self.raster, self.rank_device, self.log = evaluator, raster, rank_device, log
        inputs = json.loads(json.dumps({**(inputs or {}), "seed": seed, "rooms": config.rooms,
                                        "samples": config.samples, "latent_opt": dataclasses.asdict(latent),
                                        "pipeline": dataclasses.asdict(pipeline)}))
        _keep_or_clear(cache, inputs, fresh)
        if self.baseline_cache != cache:
            _keep_or_clear(self.baseline_cache, {k: v for k, v in inputs.items() if k not in ("cvae", "latent_opt")},
                           fresh)

    def rooms(self, set_name: str) -> tuple[list[Condition], pd.Series]:
        """The rooms of a test set (or of the validation split), and the G0 reference diversity per
        room (the E1 rooms only)."""
        if set_name == "in_distribution":
            return evaluation_rooms(self.data_dir, self.config.rooms, np.random.default_rng(self.seed), self.catalog)
        if set_name == "validation":
            set_a = load_layouts(self.data_dir / "set_a.npz")
            with np.load(self.data_dir / "splits.npz") as splits:
                rows = splits["set_a_validation"]
            chosen = self._rng(set_name).choice(rows, size=min(self.config.rooms, len(rows)), replace=False)
            return [Condition.of(set_a.layout(int(row), self.catalog), self.catalog) for row in chosen], \
                pd.Series(dtype=float)
        batch = load_layouts(self.data_dir / f"held_out_{set_name}.npz")
        count = min(self.config.rooms, len(batch))
        return [Condition.of(batch.layout(i, self.catalog), self.catalog) for i in range(count)], pd.Series(dtype=float)

    def per_room(self, set_name: str, method: str) -> pd.DataFrame:
        """The per-room rows of a method on a test set (spacegen.evaluate.evaluate), computed once."""
        path = self._path(set_name, method)
        if not path.exists():
            if set_name == "in_distribution" and method in MODEL_FREE:
                self._baselines()
            else:
                self._run(set_name, method)
        return pd.read_csv(path)

    def summary(self, set_name: str, method: str, reference: pd.Series) -> dict:
        """The summary row of spacegen.evaluate, plus the quality of the top 3 in each order (mean over
        the rooms with a valid layout)."""
        rooms = self.per_room(set_name, method)
        row = summarize(method, rooms, reference, prefiltered=method == "G0")
        return {**row, **{column: rooms[f"top_{order}"].mean() for order, column in ORDERS.items()}}

    def e1(self) -> pd.DataFrame:
        """One row per method on the E1 rooms."""
        _, reference = self.rooms("in_distribution")
        table = pd.DataFrame([self.summary("in_distribution", method, reference) for method in METHODS])
        first = ["method", "rooms", "samples", "rvr", "mean_overlap", "reachability", "quality", *ORDERS.values()]
        return table[first + [c for c in table.columns if c not in first]]

    def e10(self, sets=SETS) -> pd.DataFrame:
        """One row per test set and method (M1, M2, G0), with the set's mean room area and item count."""
        rows = []
        for set_name in sets:
            conditions, reference = self.rooms(set_name)
            context = {"set": set_name, "area_m2": float(np.mean([c.area for c in conditions])),
                       "items": float(np.mean([len(c.items) for c in conditions]))}
            for method in E10_METHODS:
                row = self.summary(set_name, method, reference)
                rows.append({**context, **{k: v for k, v in row.items()
                                           if not k.startswith("diversity") and k not in TIMING}})
        return pd.DataFrame(rows)

    def _baselines(self) -> None:
        """B1, B2 and G0 on the E1 rooms, drawing from one generator in the order of `python run.py baselines`."""
        rng = np.random.default_rng(self.seed)
        conditions, _ = evaluation_rooms(self.data_dir, self.config.rooms, rng, self.catalog)
        for name, sampler, prefiltered in baseline_samplers(self.data_dir, self.catalog, self.rules):
            self._evaluate("in_distribution", name, sampler, conditions, rng, prefiltered)

    def _run(self, set_name: str, method: str) -> None:
        conditions, _ = self.rooms(set_name)
        if method == "G0":
            sampler, prefiltered, salt = generator_sampler(self.catalog, self.rules), True, 1
        elif method == "M1":
            sampler, prefiltered, salt = CVAESampler(self.cvae, self.catalog), False, 0
        elif method == "M2":  # the same seed as M1: M2 repairs M1's draws
            sampler, prefiltered, salt = LatentOptSampler(self.cvae, self.catalog, self.rules, self.latent), False, 0
        else:
            raise ValueError(f"{method} is evaluated only on the in-distribution rooms")
        self._evaluate(set_name, method, sampler, conditions, self._rng(set_name, salt), prefiltered)

    def _evaluate(self, set_name: str, method: str, sampler, conditions: list[Condition], rng: np.random.Generator,
                  prefiltered: bool) -> None:
        start = time.perf_counter()
        top = TopThree(self.pipeline, self._rng(set_name, 2, METHODS.index(method)), self.catalog, self.rules,
                       self.evaluator, self.raster, self.rank_device)
        rows = evaluate(sampler, conditions, self.config.samples, rng, self.catalog, self.rules, prefiltered, top)
        path = self._path(set_name, method)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, index=False, lineterminator="\n")
        self.log(f"{set_name}, {method}: {len(rows)} rooms in {time.perf_counter() - start:.0f} s, "
                 f"raw valid {rows['valid'].sum() / rows['attempts'].sum():.1%}")

    def _rng(self, set_name: str, *salt: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, ROOM_SETS.index(set_name), *salt])

    def _path(self, set_name: str, method: str) -> Path:
        return (self.baseline_cache if method in MODEL_FREE else self.cache) / set_name / f"{method}.csv"


def _keep_or_clear(cache: Path, inputs: dict, fresh: bool) -> None:
    """Keep the cached rows only if they were made with these inputs (and no fresh start was asked for)."""
    stored = cache / "inputs.json"
    if fresh or not stored.exists() or json.loads(stored.read_text(encoding="utf-8")) != inputs:
        for old in [*cache.glob("*.csv"), *cache.glob("*/*.csv")]:  # per-room rows and the E8 table
            old.unlink()
        cache.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps(inputs, indent=2), encoding="utf-8", newline="\n")


def combine(tables: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Mean and standard deviation (columns <metric>_std) over the runs' tables, in their row order.
    B1, B2 and G0 come from the same cached rows in every run, so they count as one run without a spread."""
    stacked = pd.concat(tables, ignore_index=True)
    numeric = [c for c in stacked.columns if c not in keys and pd.api.types.is_numeric_dtype(stacked[c])]
    groups = stacked.groupby(keys, sort=False)
    measured = [c for c in numeric if c not in SAME_IN_EVERY_RUN]
    table = groups[numeric].mean().join(groups[measured].std(ddof=1).add_suffix("_std")).reset_index()
    table.insert(len(keys), "seeds", len(tables))
    model_free = table["method"].isin(MODEL_FREE)
    table.loc[model_free, "seeds"] = 1
    table.loc[model_free, [f"{c}_std" for c in measured]] = np.nan
    return table


def speed_check(repeats: int = 3) -> float:
    """Seconds for a fixed CPU workload, the best of `repeats`: NumPy and PyTorch matrix products
    and a plain Python loop, the three kinds of work the samplers and the checker do."""
    a = np.random.default_rng(0).standard_normal((400, 400))
    x = torch.randn(256, 256, generator=torch.Generator().manual_seed(0))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(20):
            a @ a
        with torch.no_grad():
            for _ in range(200):
                torch.tanh(x @ x)
        sum(i * i for i in range(200_000))
        best = min(best, time.perf_counter() - start)
    return best


TIMING = ("seconds_per_room", "ms_per_valid")


def timing_pass(samplers: dict[str, tuple[object, bool]], conditions: list[Condition], n: int, seed: int,
                catalog: RoomCatalog, rules: Rules) -> pd.DataFrame:
    """One row per sampler and room (seconds, valid samples), with the samplers taking turns room
    by room. `samplers` maps a name to (sampler, prefiltered); the time is measured as in
    spacegen.evaluate.evaluate: sampling, plus the checks unless the sampler filters itself."""
    rngs = {name: np.random.default_rng([seed, 7, index]) for index, name in enumerate(samplers)}
    rows = []
    for room, cond in enumerate(conditions):
        for name, (sampler, prefiltered) in samplers.items():
            row = evaluate(sampler, [cond], n, rngs[name], catalog, rules, prefiltered).iloc[0]
            rows.append({"sampler": name, "room": room, "seconds": float(row["seconds"]), "valid": int(row["valid"]),
                         "attempts": int(row["attempts"])})
    return pd.DataFrame(rows)


def with_timing(table: pd.DataFrame, rows: pd.DataFrame, key: str) -> pd.DataFrame:
    """A table with its timing columns replaced by those of a timing pass. Sampler names are
    "<value of the key column>|<run>": several runs of one value give a mean and a standard
    deviation (with a `seeds` column in the table). The long run's own timings move to *_run."""
    groups = rows.groupby("sampler", sort=False)
    per_sampler = pd.DataFrame({"timing_rooms": groups.size(), "seconds_per_room": groups["seconds"].mean(),
                                "ms_per_valid": 1000 * groups["seconds"].sum() / groups["valid"].sum()})
    per_sampler[key] = [str(name).split("|")[0] for name in per_sampler.index]
    by_key = per_sampler.groupby(key, sort=False)
    mean, spread, rooms = by_key[list(TIMING)].mean(), by_key[list(TIMING)].std(ddof=1), by_key["timing_rooms"].first()
    table = table.drop(columns=[f"{c}_std" for c in TIMING if f"{c}_std" in table.columns])
    table = table.rename(columns={c: f"{c}_run" for c in TIMING})
    lookup = table[key].astype(str)
    for column in TIMING:
        table[column] = lookup.map(mean[column])
        if "seeds" in table.columns:
            table[f"{column}_std"] = lookup.map(spread[column])
    table["timing_rooms"] = lookup.map(rooms)
    return table


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1, E8 and E10.")
    parser.add_argument("experiments", nargs="+", choices=["e1", "e8", "e10"])
    parser.add_argument("--cvae", type=Path, nargs="+", default=[RUNS_DIR / "cvae" / "default"],
                        help="one CVAE run, or several (the seeds of the frozen configuration)")
    parser.add_argument("--tag", default=None,
                        help="name of the results (reports/tables/<experiment>_<tag>.csv, runs/headline/<tag>/); "
                             "needed with several runs")
    parser.add_argument("--evaluator", type=Path, default=RUNS_DIR / "evaluator" / "e9a",
                        help="ranks the top 3; without it only the rule-score and random orders are reported")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="for the evaluator's ranking only; the samplers always run on the CPU")
    parser.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS),
                        help="E10: compute only these sets (the table is written once all four are done)")
    parser.add_argument("--e8-rooms", type=int, default=200, help="E8: the first rooms of the E1 list (diversity rooms)")
    parser.add_argument("--fresh", action="store_true", help="recompute the cached per-room rows")
    parser.add_argument("--timing-rooms", type=int, default=100,
                        help="E1 and E8: rooms of the interleaved timing pass (0 keeps the long runs' own timings)")
    parser.add_argument("--retime", action="store_true", help="repeat the interleaved timing pass")
    args = parser.parse_args(argv)
    if len(args.cvae) > 1 and args.tag is None:
        parser.error("several --cvae runs need a --tag")
    raw = load_config()
    seed = raw["seed"]
    set_seed(seed)
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    catalog, rules = load_room_catalog("living_room"), load_rules()
    has_evaluator = (args.evaluator / "model.pt").exists()
    evaluator, raster = load_evaluator_run(args.evaluator, args.device) if has_evaluator else (None, None)
    latent, pipeline, config = load_latent_opt_config(), load_pipeline_config(), load_evaluation_config()
    root = RUNS_DIR / "headline" / args.tag if args.tag else None
    headlines = []
    for run in args.cvae:
        cache = root / run.name if root else RUNS_DIR / "headline" / run.name
        inputs = {"dataset_hash": dataset_hash(data_dir), "cvae": _file_hash(run / "model.pt"),
                  "evaluator": _file_hash(args.evaluator / "model.pt") if has_evaluator else None}
        headlines.append(Headline(data_dir, cache, seed, config, load_cvae_run(run), catalog, rules, latent, pipeline,
                                  evaluator, raster, args.device, inputs, args.fresh,
                                  baseline_cache=root / "baselines" if root else None))
        write_run_record(cache, seed, data_dir, task="E1, E8 and E10", cvae_run=str(run),
                         evaluator_run=str(args.evaluator) if has_evaluator else None,
                         ranking_device=str(args.device), evaluation=dataclasses.asdict(config),
                         latent_opt=dataclasses.asdict(latent), pipeline=dataclasses.asdict(pipeline))
    from experiments.figures import FIGURES

    tables_dir, figures_dir = REPORTS_DIR / "tables", REPORTS_DIR / "figures"
    suffix = f"_{args.tag}" if args.tag else ""
    speed_file = (root or headlines[0].cache) / "speed.json"
    for name in args.experiments:
        before = speed_check()
        if name == "e1":
            tables = [h.e1() for h in headlines]
            keys = ["method"]
        elif name == "e10":
            tables = [h.e10(args.sets) for h in headlines]
            keys = ["set", "method"]
            if set(args.sets) != set(SETS):
                print(f"computed {', '.join(args.sets)}; the E10 table is written once all four sets are done")
                continue
        else:
            tables = [_latent_steps(h, run, args.e8_rooms) for h, run in zip(headlines, args.cvae)]
            keys = ["steps", "method"]
        table = tables[0] if len(tables) == 1 else combine(tables, keys)
        if name in ("e1", "e8") and args.timing_rooms > 0:
            timing_file = (root or headlines[0].cache) / f"timing_{name}.csv"
            if args.retime or args.fresh or not timing_file.exists():
                first = headlines[0]
                if name == "e1":
                    samplers = {method: (sampler, prefiltered)
                                for method, sampler, prefiltered in baseline_samplers(data_dir, catalog, rules)}
                    for h, run in zip(headlines, args.cvae):
                        samplers[f"M1|{run.name}"] = (CVAESampler(h.cvae, catalog), False)
                        samplers[f"M2|{run.name}"] = (LatentOptSampler(h.cvae, catalog, rules, latent), False)
                else:
                    samplers = {f"{steps}|{run.name}": (LatentOptSampler(h.cvae, catalog, rules,
                                                                         load_latent_opt_config(steps=steps)), False)
                                for steps in E8_STEPS for h, run in zip(headlines, args.cvae)}
                conditions = first.rooms("in_distribution")[0][:args.timing_rooms]
                print(f"{name}: timing {len(samplers)} samplers in turn on {len(conditions)} rooms")
                timing_pass(samplers, conditions, config.samples, seed, catalog, rules).to_csv(
                    timing_file, index=False, lineterminator="\n")
            table = with_timing(table, pd.read_csv(timing_file), keys[0])
        tables_dir.mkdir(parents=True, exist_ok=True)
        table.round(5).to_csv(tables_dir / f"{name}{suffix}.csv", index=False, lineterminator="\n")
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(table.round(4).to_string(index=False))
        figures_dir.mkdir(parents=True, exist_ok=True)
        saved = pd.read_csv(tables_dir / f"{name}{suffix}.csv")  # as `python run.py figures` draws it
        for figure_name, figure in FIGURES[name](saved, None).items():
            figure.savefig(figures_dir / f"{name}{suffix}_{figure_name}.png", facecolor="#fcfcfb")
        after = speed_check()
        # read again here: experiments run side by side share this file
        speeds = json.loads(speed_file.read_text(encoding="utf-8")) if speed_file.exists() else {}
        speeds[name] = {"before": round(before, 4), "after": round(after, 4),
                        "finished": time.strftime("%Y-%m-%d %H:%M")}
        speed_file.write_text(json.dumps(speeds, indent=2), encoding="utf-8", newline="\n")
        steady = max(before, after) / min(before, after) <= 1.2
        print(f"wrote reports/tables/{name}{suffix}.csv and its figures; speed check {before:.3f} s before, "
              f"{after:.3f} s after" + ("" if steady else ": the machine changed speed, do not trust the timings"))
    return 0


def _latent_steps(headline: Headline, run: Path, rooms: int) -> pd.DataFrame:
    """E8 for one CVAE run (experiments.screening.run_latent_steps), cached next to its per-room rows."""
    path = headline.cache / f"e8_{rooms}_rooms.csv"
    if not path.exists():
        table = run_latent_steps(headline.data_dir, headline.seed, "cpu", E8_STEPS, rooms, run, log=headline.log)
        table.to_csv(path, index=False, lineterminator="\n")
    return pd.read_csv(path)


if __name__ == "__main__":
    sys.exit(main())
