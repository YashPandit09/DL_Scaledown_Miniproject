"""The demo's backup pictures and the report's demo figures (report Section 6).

python run.py demo-assets [--cvae RUN]

Runs the app's own logic (spacegen/app_logic.py) for the room of the demo script and saves what
the app would show, in reports/demo/:

  top1.png to top3.png, top1.json to top3.json   the top 3 for the 5 x 4 m room
  layouts.png      the same three side by side, with quality, cost and floor use
  pinned.png       the top 3 with the sofa pinned at the north wall, facing south
  compare.csv, compare.png    the five methods on that room: the table and each method's best layout
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from spacegen import app_logic as logic
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.paths import REPORTS_DIR
from spacegen.pins import Pin
from spacegen.pipeline import PipelineResult, Request
from spacegen.rules import Rules, load_rules
from spacegen.viz import INK, SURFACE, draw_layout

ROOM = dict(width=5.0, depth=4.0, door_wall="W", door_offset=0.5)
ITEMS = {"sofa": None, "tv_unit": None, "coffee_table": "coffee_table_standard", "armchair": "armchair_standard"}
PIN = {"sofa": Pin(2.5, 3.4, facing=2)}  # at the north wall, facing south


def plans(layouts: list, titles: list[str], heading: str, catalog: RoomCatalog, rules: Rules) -> Figure:
    """Floor plans side by side, each with its title."""
    fig = Figure(figsize=(3.6 * max(len(layouts), 1), 4.1), dpi=150, facecolor=SURFACE, layout="constrained")
    fig.suptitle(heading, x=0.01, ha="left", fontsize=9, color=INK, linespacing=1.4)
    axes = np.atleast_1d(fig.subplots(1, max(len(layouts), 1)))
    for ax, layout, title in zip(axes, layouts, titles):
        if layout is None:
            ax.text(0.5, 0.5, "no valid layout", ha="center", va="center", fontsize=8, color=INK, transform=ax.transAxes)
            ax.set_axis_off()
        else:
            draw_layout(ax, layout, catalog, rules)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    for ax in axes[len(layouts):]:
        ax.set_axis_off()
    return fig


def top_titles(result: PipelineResult, catalog: RoomCatalog, rules: Rules) -> list[str]:
    titles = []
    for rank, candidate in enumerate(result.top, start=1):
        facts = logic.describe(candidate.layout, catalog, rules)
        titles.append(f"Layout {rank}: quality {facts['quality']:.2f}, {facts['cost']:,} INR, "
                      f"{facts['floor_use']:.0%} of the floor")
    return titles


def make_assets(models: logic.Models, out: Path, catalog: RoomCatalog, rules: Rules, seed: int = 0,
                candidates: int = 64, log=print) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(figure: Figure, name: str) -> None:
        written.append(out / name)
        figure.savefig(written[-1], facecolor=SURFACE)

    request = Request(**ROOM, items=ITEMS)
    result = logic.run_request(models, request, seed, candidates, True, catalog, rules)
    log(f"top 3: {result.valid} of {result.candidates} candidates valid, {sum(result.seconds.values()):.1f} s")
    for rank, candidate in enumerate(result.top, start=1):
        facts = logic.describe(candidate.layout, catalog, rules)
        for suffix, content in (("png", logic.layout_png(candidate.layout, catalog, rules)),
                                ("json", logic.layout_json(candidate.layout, catalog, rules,
                                                           {"quality": round(facts["quality"], 4), "cost": facts["cost"]},
                                                           {"rank": rank, "seed": seed}).encode("utf-8"))):
            written.append(out / f"top{rank}.{suffix}")
            written[-1].write_bytes(content)
    save(plans([c.layout for c in result.top], top_titles(result, catalog, rules),
               f"The top 3 for a {ROOM['width']:g} x {ROOM['depth']:g} m room: {result.valid} of {result.candidates} "
               "candidates passed every hard check", catalog, rules), "layouts.png")

    pinned = logic.run_request(models, Request(**ROOM, items=ITEMS, pins=PIN), seed, candidates, True, catalog, rules)
    log(f"pinned: {pinned.valid} of {pinned.candidates} candidates valid")
    save(plans([c.layout for c in pinned.top], top_titles(pinned, catalog, rules),
               f"The sofa pinned at ({PIN['sofa'].x:g}, {PIN['sofa'].y:g}) m facing south: {pinned.valid} of "
               f"{pinned.candidates} candidates valid with it exactly there", catalog, rules), "pinned.png")

    condition, _, _ = logic.condition_for(request, catalog, rules)
    table, best = logic.compare_methods(models, condition, candidates, seed, catalog, rules)
    written.append(out / "compare.csv")
    table.drop(columns=["seconds", "note"]).round(4).to_csv(written[-1], index=False, lineterminator="\n")
    titles = [f"{name}\n{int(row['valid'])} of {int(row['raw samples'])} valid" if row.get("note", "") == ""
              else f"{name}\nnot available" for name, (_, row) in zip(best, table.iterrows())]
    save(plans(list(best.values()), titles, "Five methods on the same room: each method's best valid layout by the "
               "rule score", catalog, rules), "compare.png")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The demo's backup pictures and the report's demo figures.")
    parser.add_argument("--cvae", type=Path, default=None, help="default: the app's model (the frozen seed 0)")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "demo")
    args = parser.parse_args(argv)
    catalog, rules = load_room_catalog("living_room"), load_rules()
    models = logic.load_models(catalog, cvae_run=args.cvae)
    for note in models.notes:
        print("note:", note)
    if models.cvae is None:
        return 1
    written = make_assets(models, args.out, catalog, rules)
    print(f"wrote {len(written)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
