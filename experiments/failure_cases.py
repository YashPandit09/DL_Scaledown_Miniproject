"""Failure-case analysis: what goes wrong, how often, and what it looks like.

python run.py failures [--cvae RUN] [--rooms N]

B1, B2, M1 and M2 sample the same test rooms (the first rooms of the E1 list, 64 samples each).
M1 and M2 start from the same draws, so sample i of M2 is sample i of M1 after latent
optimization. Every sample is classified:

  out of room (H1), overlap (H2), door blocked (H3), unreachable (H4)
           the hard checks it breaks. A sample can break several; its `cause` is the first in
           this order, and the table also gives the share that breaks each check at all
  poor     valid, but the rule score is below POOR
  good     valid with a higher score

and every room: `no valid layout` when none of its samples passes, `collapsed` when its valid
layouts are nearly the same (diversity below COLLAPSED meters).

Writes reports/tables/failure_causes.csv (one row per method), reports/tables/failure_cases.csv
(the cases of the gallery) and reports/figures/failure_cases.png. The discussion is in
reports/failure_cases.md.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from spacegen.baselines import StatisticalBaseline, UniformBaseline, load_baseline_config
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import load_config
from spacegen.dataset import load_layouts
from spacegen.evaluate import evaluation_rooms
from spacegen.latent_opt import load_latent_opt_config
from spacegen.layout import Layout
from spacegen.metrics import diversity
from spacegen.paths import DATA_DIR, REPORTS_DIR, saved_run
from spacegen.pipeline import CVAESampler, LatentOptSampler, load_cvae_run
from spacegen.quality import quality_score
from spacegen.rules import Rules, check_layout, load_rules, reachability
from spacegen.viz import INK, SURFACE, draw_layout

CAUSES = ("out of room", "overlap", "door blocked", "unreachable")  # H1 to H4, in the checker's order
POOR = 0.5  # a valid layout below this rule score counts as poor
COLLAPSED = 0.5  # a room whose valid layouts are closer than this on average (m) counts as collapsed
METHODS = ("B1", "B2", "M1", "M2")


@dataclass
class Sample:
    method: str
    room: int
    index: int  # position among the room's samples; M1 and M2 share it
    layout: Layout
    broken: tuple[str, ...]  # the hard checks it breaks, in the order of CAUSES
    quality: float | None  # rule score of a valid sample
    weakest: str | None  # the lowest of the four quality terms of a valid sample

    @property
    def outcome(self) -> str:
        if self.broken:
            return self.broken[0]
        return "poor" if self.quality < POOR else "good"


def classify(method: str, room: int, index: int, layout: Layout, catalog: RoomCatalog, rules: Rules) -> Sample:
    reach = reachability(layout, catalog, rules)
    result = check_layout(layout, catalog, rules, reach=reach)
    found = (result.items_out_of_room, result.overlapping_pairs, result.items_in_door_zone, result.unreachable_items)
    broken = tuple(cause for cause, slots in zip(CAUSES, found) if slots)
    if broken:
        return Sample(method, room, index, layout, broken, None, None)
    quality = quality_score(layout, catalog, rules, reach=reach)
    terms = {"alignment": quality.align, "relations": quality.relations, "circulation": quality.circulation,
             "space": quality.space}
    return Sample(method, room, index, layout, (), float(quality.total), min(terms, key=terms.get))


def collect(samplers: dict, conditions: list, n: int, seed: int, catalog: RoomCatalog, rules: Rules,
            log=print) -> list[Sample]:
    """Every method's samples on every room, classified. M1 and M2 draw from generators with the same seed."""
    samples = []
    for position, (method, sampler) in enumerate(samplers.items()):
        rng = np.random.default_rng([seed, 2 if method in ("M1", "M2") else position])
        for room, cond in enumerate(conditions):
            layouts = sampler.sample(cond, n, rng).layouts
            samples += [classify(method, room, i, layout, catalog, rules) for i, layout in enumerate(layouts)]
        mine = [s for s in samples if s.method == method]
        log(f"{method}: {len(mine)} samples, {np.mean([not s.broken for s in mine]):.1%} valid")
    return samples


def cause_table(samples: list[Sample]) -> pd.DataFrame:
    """One row per method: the share of its samples with each outcome, the share breaking each
    check at all, and the shares of rooms without a valid layout or with collapsed ones."""
    rows = []
    for method in dict.fromkeys(s.method for s in samples):
        mine = [s for s in samples if s.method == method]
        row = {"method": method, "samples": len(mine), "valid": np.mean([not s.broken for s in mine])}
        row.update({f"cause: {cause}": np.mean([s.outcome == cause for s in mine]) for cause in CAUSES})
        row.update({f"breaks: {cause}": np.mean([cause in s.broken for s in mine]) for cause in CAUSES})
        row["several checks"] = np.mean([len(s.broken) > 1 for s in mine])
        row["poor"] = np.mean([s.outcome == "poor" for s in mine])
        row["good"] = np.mean([s.outcome == "good" for s in mine])
        rooms = {}
        for s in mine:
            rooms.setdefault(s.room, []).append(s)
        spread = {room: diversity([s.layout for s in group if not s.broken]) for room, group in rooms.items()}
        row["rooms"] = len(rooms)
        row["rooms without a valid layout"] = np.mean([all(s.broken for s in group) for group in rooms.values()])
        row["rooms collapsed"] = np.mean([d is not None and d < COLLAPSED for d in spread.values()])
        measured = [d for d in spread.values() if d is not None]
        row["lowest room diversity (m)"] = min(measured) if measured else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pick_cases(samples: list[Sample]) -> list[tuple[str, Sample]]:
    """The gallery: (caption, sample) for at least ten failed or poor outputs across the methods.
    Each case is the first sample, in room order, that fits its description."""
    by_method = {method: [s for s in samples if s.method == method] for method in dict.fromkeys(s.method for s in samples)}
    m1 = {(s.room, s.index): s for s in by_method.get("M1", [])}
    m2 = by_method.get("M2", [])
    cases: list[tuple[str, Sample]] = []

    def add(caption: str, found) -> None:
        sample = next(iter(found), None)
        if sample is not None:
            cases.append((caption, sample))

    for cause in CAUSES:
        add(f"M2: {cause} only", (s for s in m2 if s.broken == (cause,)))
    add("M2: several checks at once", (s for s in m2 if len(s.broken) > 1))
    valid = sorted((s for s in m2 if not s.broken), key=lambda s: s.quality)
    add("M2: valid but poor (its lowest score)", valid[:1])
    repaired = [s for s in m2 if not s.broken and (s.room, s.index) in m1 and m1[(s.room, s.index)].broken]
    if repaired:  # the same candidate before and after latent optimization
        cases.append(("M1: before repair", m1[(repaired[0].room, repaired[0].index)]))
        cases.append(("M2: the same candidate, repaired", repaired[0]))
    stuck = [s for s in m2 if s.broken and (s.room, s.index) in m1 and m1[(s.room, s.index)].broken
             and (not repaired or s.room != repaired[0].room)]
    if stuck:
        cases.append(("M1: before repair", m1[(stuck[0].room, stuck[0].index)]))
        cases.append(("M2: the same candidate, not repaired", stuck[0]))
    b2 = sorted((s for s in by_method.get("B2", []) if not s.broken), key=lambda s: s.quality)
    add("B2: valid but poor (its lowest score)", b2[:1])
    add("B1: a typical random placement", (s for s in by_method.get("B1", []) if len(s.broken) > 1))
    return cases


def case_table(cases: list[tuple[str, Sample]], catalog: RoomCatalog) -> pd.DataFrame:
    def items(sample: Sample) -> str:
        return ", ".join(catalog.slots[k].name for k in np.flatnonzero(sample.layout.mask))

    return pd.DataFrame([{"case": number, "caption": caption, "method": s.method, "room": s.room, "sample": s.index,
                          "width": s.layout.width, "depth": s.layout.depth, "door_wall": s.layout.door_wall,
                          "items": items(s), "breaks": "; ".join(s.broken), "quality": s.quality, "weakest_term": s.weakest}
                         for number, (caption, s) in enumerate(cases, start=1)])


def plot_cases(cases: list[tuple[str, Sample]], catalog: RoomCatalog, rules: Rules, columns: int = 4) -> Figure:
    """The gallery: one floor plan per case; items that break a check have a red edge and its number."""
    rows = int(np.ceil(len(cases) / columns))
    fig = Figure(figsize=(3.3 * columns, 3.4 * rows + 0.5), dpi=150, facecolor=SURFACE, layout="constrained")
    fig.suptitle("Failure cases: red edges mark the items that break a hard check (H1 out of room, H2 overlap, "
                 "H3 door zone, H4 unreachable);\nthe shaded floor is what can be walked to from the door",
                 x=0.01, ha="left", fontsize=9, color=INK, linespacing=1.4)
    axes = np.atleast_1d(fig.subplots(rows, columns)).ravel()
    for number, (ax, (caption, sample)) in enumerate(zip(axes, cases), start=1):
        draw_layout(ax, sample.layout, catalog, rules)
        detail = "; ".join(sample.broken) if sample.broken else f"quality {sample.quality:.2f}, weakest: {sample.weakest}"
        ax.set_title(f"{number}. {caption}\n{detail}", fontsize=7, color=INK, loc="left")
    for ax in axes[len(cases):]:
        ax.set_visible(False)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Failure-case analysis.")
    frozen = saved_run("cvae", "frozen", "seed-0")
    parser.add_argument("--cvae", type=Path, default=frozen if (frozen / "model.pt").exists()
                        else saved_run("cvae", "default"))
    parser.add_argument("--rooms", type=int, default=100)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    args = parser.parse_args(argv)
    raw = load_config()
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    catalog, rules = load_room_catalog("living_room"), load_rules()
    conditions, _ = evaluation_rooms(data_dir, args.rooms, np.random.default_rng(raw["seed"]), catalog)
    set_a = load_layouts(data_dir / "set_a.npz")
    with np.load(data_dir / "splits.npz") as splits:
        train = set_a.subset(splits["set_a_train"])
    cvae = load_cvae_run(args.cvae)
    samplers = {"B1": UniformBaseline(catalog), "B2": StatisticalBaseline(catalog, load_baseline_config()).fit(train),
                "M1": CVAESampler(cvae, catalog), "M2": LatentOptSampler(cvae, catalog, rules, load_latent_opt_config())}
    samples = collect(samplers, conditions, args.samples, raw["seed"], catalog, rules)
    causes, cases = cause_table(samples), pick_cases(samples)
    tables, figures = REPORTS_DIR / "tables", REPORTS_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    causes.round(4).to_csv(tables / "failure_causes.csv", index=False, lineterminator="\n")
    case_table(cases, catalog).round(4).to_csv(tables / "failure_cases.csv", index=False, lineterminator="\n")
    plot_cases(cases, catalog, rules).savefig(figures / "failure_cases.png", facecolor=SURFACE)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(causes.round(3).T.to_string())
        print(case_table(cases, catalog).drop(columns=["items"]).round(3).to_string(index=False))
    print(f"wrote reports/tables/failure_causes.csv, failure_cases.csv and reports/figures/failure_cases.png "
          f"({len(cases)} cases, CVAE {args.cvae.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
