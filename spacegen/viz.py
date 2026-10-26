"""2D floor plans of layouts (walls, door, clearance zone, walkable area and furniture), and the data figures.

Render a saved layout JSON:  python -m spacegen.viz layout.json layout.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from spacegen import geometry
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.layout import Layout, layout_from_dict
from spacegen.rules import CheckResult, Rules, check_layout, door_geometry, load_rules, reachability

# Colours from the data-viz reference palette (light mode). Every item is named by a direct
# label, so all furniture shares one hue; the status red is kept for items that break a
# hard check, always together with the check's name in the label.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
WALKABLE = "#f0efec"
FURNITURE_FILL = "#cde2fb"
FURNITURE_EDGE = "#2a78d6"
FRONT_EDGE = "#1c5cab"
CRITICAL = "#d03b3b"
CONTEXT = "#a9a8a1"  # neutral grey for background data (the training rooms behind the held-out sets)
HELD_OUT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, the most a scatter may use

# Sequential blue ramp (steps 100 to 700 of the reference palette), starting at the surface
# so that zero coverage disappears into the background.
COVERAGE = LinearSegmentedColormap.from_list("coverage", [
    SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

_WALL_ENDS = {"S": ((0, 0), (1, 0)), "E": ((1, 0), (1, 1)), "N": ((1, 1), (0, 1)), "W": ((0, 1), (0, 0))}


def plot_layout(layout: Layout, catalog: RoomCatalog, rules: Rules, title: str | None = None,
                show_walkable: bool = True, width: float | None = None) -> Figure:
    """A figure with one floor plan. Matplotlib's object API is used, so no global state.

    `width` (inches) gives a small plan for a narrow column, as in the app: the drawing shrinks
    and the fonts keep their size, so the labels stay readable. A small plan has only `title`
    above it (no summary of the checks) and no legend: the caller explains the marks once."""
    if width is not None:
        scale = (width - 0.3) / (layout.width + 0.6)  # inches per meter, the tick labels aside
        height = scale * (layout.depth + 0.6) + (0.5 if title else 0.25)
        fig = Figure(figsize=(width, height), dpi=220, facecolor=SURFACE, layout="constrained")
        draw_layout(fig.add_subplot(), layout, catalog, rules, show_walkable, scale=scale)
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=9, color=INK)
        return fig
    fig = Figure(figsize=(1.6 + 1.3 * layout.width, 2.4 + 1.3 * layout.depth), dpi=150, facecolor=SURFACE,
                 layout="constrained")
    ax = fig.add_subplot()
    result = draw_layout(ax, layout, catalog, rules, show_walkable)
    fig.suptitle(title or _summary(layout, catalog, result), x=0.02, ha="left", fontsize=9, color=INK,
                 linespacing=1.4)
    legend = [Patch(facecolor=WALKABLE, edgecolor=INK_MUTED, linewidth=0.5, label="walkable from the door"),
              Patch(facecolor="none", edgecolor=INK_MUTED, hatch="///", label="door clearance zone"),
              Line2D([], [], color=FRONT_EDGE, linewidth=3, label="front of an item")]
    if not result.valid:
        legend.append(Patch(facecolor=FURNITURE_FILL, edgecolor=CRITICAL, linewidth=2, label="breaks a hard check"))
    fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False, fontsize=7,
               labelcolor=INK_SECONDARY)
    return fig


def draw_layout(ax: Axes, layout: Layout, catalog: RoomCatalog, rules: Rules,
                show_walkable: bool = True, scale: float | None = None) -> CheckResult:
    """Draw one floor plan onto existing axes; returns the hard-check result it shows.

    `scale` (inches per meter) is given for a small plan: the door's label then goes inside its
    clearance zone, clear of the axis, and a two-word name too long for its item takes two lines."""
    result = check_layout(layout, catalog, rules)
    w, d = layout.width, layout.depth
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.3, w + 0.3)
    ax.set_ylim(-0.3, d + 0.3)
    ax.set_aspect("equal")

    if show_walkable:
        reach = reachability(layout, catalog, rules)
        ax.imshow(reach.reached.T, origin="lower", extent=(0, w, 0, d), interpolation="nearest",
                  cmap=ListedColormap([SURFACE, WALKABLE]), vmin=0, vmax=1, zorder=0)

    door = door_geometry(w, d, layout.door_wall, layout.door_offset, rules.door)
    lo, _ = geometry.box_bounds(door.zone_center, door.zone_size)
    ax.add_patch(Rectangle(lo, *door.zone_size, facecolor="none", edgecolor=INK_MUTED, hatch="///",
                           linewidth=0.6, zorder=1))
    _draw_walls(ax, layout, door, rules.door.width, label=scale is None)
    if scale is not None:
        ax.text(*door.zone_center, "door", ha="center", va="center", fontsize=6.5, color=INK_SECONDARY, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))

    broken = _broken_checks(result)
    eff = layout.eff_size
    for k in np.flatnonzero(layout.mask):
        slot = catalog.slots[k]
        lo, _ = geometry.box_bounds(layout.center[k], eff[k])
        failing = broken.get(int(k), [])
        ax.add_patch(Rectangle(lo, *eff[k], facecolor=FURNITURE_FILL, linewidth=2.0 if failing else 1.0,
                               edgecolor=CRITICAL if failing else FURNITURE_EDGE, zorder=2))
        if slot.rot_symmetry == 1:  # symmetric items have no meaningful front
            _draw_front(ax, layout.center[k], eff[k], int(layout.rot[k]))
        name = slot.name.replace("_", " ")
        if scale is not None and len(name) * 0.58 * 6.5 / 72 > scale * eff[k].max():  # wider than the item
            name = name.replace(" ", "\n", 1)
        label = name + (f"\n({', '.join(failing)})" if failing else "")
        ax.text(*layout.center[k], label, ha="center", va="center", fontsize=6.5, color=INK, zorder=4,
                rotation=90 if eff[k, 0] < eff[k, 1] else 0)  # run the label along a tall, thin item

    ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return result


def plot_raster(raster, canvas: float, title: str | None = None) -> Figure:
    """The four channels of one evaluator input image side by side, in meters on the canvas."""
    from spacegen.raster import CHANNELS

    image = np.asarray(raster.detach().cpu() if hasattr(raster, "detach") else raster)
    fig = Figure(figsize=(12, 3.6), dpi=150, facecolor=SURFACE, layout="constrained")
    for c, (ax, name) in enumerate(zip(fig.subplots(1, len(CHANNELS)), CHANNELS)):
        channel = image[c].T  # rows are y, so the picture matches the floor plan
        shown = ax.imshow(channel, origin="lower", extent=(0, canvas, 0, canvas), cmap=COVERAGE,
                          vmin=0, vmax=max(1.0, float(channel.max())), interpolation="nearest")
        ax.set_title(f"{c}: {name}", fontsize=8, color=INK, loc="left")
        ax.tick_params(labelsize=6, colors=INK_MUTED, length=2)
        for spine in ax.spines.values():
            spine.set_color(INK_MUTED)
            spine.set_linewidth(0.5)
        bar = fig.colorbar(shown, ax=ax, shrink=0.8)
        bar.ax.tick_params(labelsize=6, colors=INK_MUTED)
        bar.outline.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK)
    return fig


def plot_rejection(attempts, cap: int) -> Figure:
    """How often the generator rejects a placement attempt, by room area and by item count.

    `attempts` is the attempt log of generator.generate_set_a(); `cap` the attempts per room.
    """
    from spacegen.generator import rejection_summary

    fig = Figure(figsize=(10, 4.1), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharey=True, width_ratios=[3, 2])
    for ax, by, xlabel in zip(axes, ("area", "items"), ("room floor area (m²)", "items in the room")):
        table = rejection_summary(attempts, by)
        groups = [f"{g.left:.0f}–{g.right:.0f}" if by == "area" else str(g) for g in table.index]
        labels = [f"{g}\n{n}" for g, n in zip(groups, table["attempts"])]  # attempts under each group
        bars = ax.bar(labels, 100 * table["rejected"], width=0.62, color=FURNITURE_EDGE)
        ax.bar_label(bars, labels=[f"{r:.0%}" for r in table["rejected"]], padding=2, fontsize=6.5,
                     color=INK_SECONDARY)
        ax.set_xlabel(f"{xlabel}, with the number of attempts below", fontsize=7.5, color=INK_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)
        for name, spine in ax.spines.items():
            spine.set_visible(name == "bottom")
            spine.set_color(INK_MUTED)
    axes[1].tick_params(left=False)
    axes[0].set_ylabel("attempts rejected (%)", fontsize=7.5, color=INK_SECONDARY)
    rooms = attempts["room"].nunique()
    dropped = rooms - attempts.loc[attempts["outcome"] == "valid", "room"].nunique()
    fig.suptitle("Generator attempts rejected: no position left for the furniture, or a hard check failed\n"
                 f"{rooms} rooms, {len(attempts)} attempts; {dropped} of {rooms} rooms dropped after "
                 f"{cap} failed attempts", x=0.01, ha="left", fontsize=9, color=INK, linespacing=1.4)
    return fig


def plot_dataset_rooms(set_a, held_out: dict, regions: dict) -> Figure:
    """Where the rooms of each set lie: training rooms by width and depth with the held-out
    sets around them, the training floor areas with the held-out area bands, and items per room.

    `set_a` and the values of `held_out` are LayoutBatch objects; `regions` the held-out regions.
    """
    fig = Figure(figsize=(12.5, 4.3), dpi=150, facecolor=SURFACE, layout="constrained")
    rooms_ax, area_ax, items_ax = fig.subplots(1, 3, width_ratios=[1.3, 1.2, 0.75])
    shown = set_a.room[:: max(1, len(set_a) // 4000)]  # thin the training cloud for the scatter
    rooms_ax.scatter(*shown.T, s=3, color=CONTEXT, alpha=0.6, linewidths=0, label="training rooms")
    colors = dict(zip(held_out, HELD_OUT_COLORS))
    for name, batch in held_out.items():
        label = name.replace("_", " ")
        rooms_ax.scatter(*batch.room.T, s=4, color=colors[name], alpha=0.7, linewidths=0, label=label)
        width, depth = np.median(batch.room, axis=0)
        rooms_ax.text(width, depth, label, fontsize=7, color=INK, ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    rooms_ax.set(xlabel="width W (m)", ylabel="depth D (m)", aspect="equal")
    rooms_ax.legend(loc="upper left", fontsize=6.5, frameon=False, markerscale=3, labelcolor=INK_SECONDARY)

    area = set_a.room.prod(axis=1)
    area_ax.hist(area, bins=np.arange(10.0, 43.0, 1.0), color=CONTEXT, edgecolor=SURFACE, linewidth=0.6)
    for name, region in regions.items():
        low, high = region.area
        if np.isfinite(low):
            area_ax.axvspan(low, min(high, 43.0), color=colors[name], alpha=0.15, linewidth=0)
            area_ax.text((low + min(high, 42.0)) / 2, 0.97, f"held out:\n{name.replace('_', ' ')}", fontsize=6.5,
                         color=INK_SECONDARY, ha="center", va="top", transform=area_ax.get_xaxis_transform())
    area_ax.set(xlabel="floor area (m²)", ylabel="training rooms")

    counts = pd.Series(set_a.mask.sum(axis=1)).value_counts(normalize=True).sort_index()
    bars = items_ax.bar([str(k) for k in counts.index], 100 * counts.to_numpy(), width=0.6, color=CONTEXT)
    items_ax.bar_label(bars, labels=[f"{v:.0%}" for v in counts], padding=2, fontsize=6.5, color=INK_SECONDARY)
    items_ax.set(xlabel="items in the room", ylabel="training rooms (%)")
    for ax in (rooms_ax, area_ax, items_ax):
        _quiet_axes(ax)
    fig.suptitle(f"Dataset rooms: {len(set_a)} training layouts (grey) and the held-out test sets",
                 x=0.01, ha="left", fontsize=9, color=INK)
    return fig


def plot_set_b_labels(info) -> Figure:
    """Share of valid Set B layouts per perturbation type, from the checker's labels."""
    from spacegen.perturb import label_summary

    summary = label_summary(info).iloc[::-1]  # first type at the top
    fig = Figure(figsize=(7.5, 3.9), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot()
    bars = ax.barh([k.replace("_", " ") for k in summary.index], 100 * summary["valid"], height=0.6,
                   color=FURNITURE_EDGE)
    ax.bar_label(bars, labels=[f"{v:.0%} of {n}" for v, n in zip(summary["valid"], summary["samples"])], padding=3,
                 fontsize=6.5, color=INK_SECONDARY)
    ax.set(xlim=(0, 115), xlabel="layouts valid by the checker (%)")
    _quiet_axes(ax, keep="left")
    fig.suptitle(f"Set B labels per perturbation type: {len(info)} layouts, {info['valid'].mean():.0%} valid",
                 x=0.01, ha="left", fontsize=9, color=INK)
    return fig


def plot_f_max(calibration, f_max: float | None, coefficients, configured: float) -> Figure:
    """Calibration of the feasibility pre-check (f_max): share of rooms the generator furnishes
    against the footprint ratio, the logistic fit, and where it crosses one half."""
    bins = np.round(np.arange(0.05, 0.476, 0.025), 3)
    groups = calibration.groupby(pd.cut(calibration["ratio"], bins), observed=True)["furnished"]
    middle = np.array([interval.mid for interval in groups.mean().index])
    fig = Figure(figsize=(7.5, 4.0), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot()
    ratio = np.linspace(bins[0], bins[-1], 200)
    ax.plot(ratio, 100 / (1 + np.exp(-(coefficients[0] + coefficients[1] * ratio))), color=HELD_OUT_COLORS[1],
            linewidth=2, label="logistic fit")
    ax.scatter(middle, 100 * groups.mean().to_numpy(), s=24, color=FURNITURE_EDGE, zorder=3,
               label="rooms in a 0.025-wide bin")
    for x, (share, n) in zip(middle, zip(groups.mean(), groups.size())):
        ax.annotate(str(n), (x, 100 * share), textcoords="offset points", xytext=(0, -11), ha="center",
                    fontsize=5.5, color=INK_MUTED)
    ax.axhline(50, color=INK_MUTED, linewidth=0.6, linestyle=":")
    ax.axvline(configured, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(configured, 4, f" rules.yaml when built: {configured:g}", fontsize=6.5, color=INK_SECONDARY)
    if f_max is not None:
        ax.axvline(f_max, color=INK, linewidth=1.2)
        ax.text(f_max, 12, f"calibrated f_max = {f_max:.3f} ", fontsize=7, color=INK, ha="right")
    ax.set(ylim=(0, 105), xlabel="furniture footprint / floor outside the door zone",
           ylabel="rooms furnished within the attempts (%)")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, labelcolor=INK_SECONDARY)
    _quiet_axes(ax)
    fig.suptitle(f"f_max calibration: {len(calibration)} rooms, each optional item in half of them "
                 "(number of rooms under each point)", x=0.01, ha="left", fontsize=9, color=INK)
    return fig


def plot_confusion(metrics: dict, run: str) -> Figure:
    """The evaluator's confusion matrix on the Set B test split (E9a): counts, and shares of each
    true class, shaded by that share. `metrics` comes from evaluator_report.overall_metrics."""
    counts = np.array([[metrics["true_valid"], metrics["false_invalid"]],
                       [metrics["false_valid"], metrics["true_invalid"]]])  # rows: checker; columns: evaluator
    shares = counts / counts.sum(axis=1, keepdims=True)
    fig = Figure(figsize=(5.2, 4.3), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot()
    ax.imshow(shares, cmap=COVERAGE, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{counts[i, j]:,}\n{shares[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color=SURFACE if shares[i, j] > 0.6 else INK)
    labels = ["valid", "invalid"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="the evaluator says", ylabel="the checker says")
    ax.tick_params(labelsize=8, colors=INK_SECONDARY, length=0)
    ax.xaxis.label.set(fontsize=8, color=INK_SECONDARY)
    ax.yaxis.label.set(fontsize=8, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(f"Evaluator ({run}) on the Set B test split, {int(counts.sum()):,} layouts\n"
                 f"accuracy {metrics['accuracy']:.1%}, F1 (valid) {metrics['f1']:.3f}, "
                 f"F1 (invalid) {metrics['f1_invalid']:.3f}, ROC-AUC {metrics['roc_auc']:.3f}",
                 x=0.02, ha="left", fontsize=8.5, color=INK, linespacing=1.4)
    return fig


def _quiet_axes(ax: Axes, keep: str = "bottom") -> None:
    """Muted ticks and labels; only the baseline spine (and the left one for scatter plots) shows."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)
    ax.xaxis.label.set(fontsize=7.5, color=INK_SECONDARY)
    ax.yaxis.label.set(fontsize=7.5, color=INK_SECONDARY)
    for name, spine in ax.spines.items():
        spine.set_visible(name in (keep, "left") if keep == "bottom" else name == keep)
        spine.set_color(INK_MUTED)


def _draw_walls(ax: Axes, layout: Layout, door, door_width: float, label: bool = True) -> None:
    room = layout.room
    for wall, (start, end) in _WALL_ENDS.items():
        a, b = np.array(start) * room, np.array(end) * room
        if wall == layout.door_wall:  # leave the door opening out of the wall
            along = (b - a) / np.linalg.norm(b - a)
            segments = [(a, door.center - along * door_width / 2), (door.center + along * door_width / 2, b)]
            if label:
                ax.text(*(door.center - door.inward * 0.15), "door", ha="center", va="center",
                        fontsize=6.5, color=INK_SECONDARY)
        else:
            segments = [(a, b)]
        for p, q in segments:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, linewidth=2.5, solid_capstyle="projecting", zorder=3)


def _draw_front(ax: Axes, center: np.ndarray, eff: np.ndarray, rot: int) -> None:
    """A thick line along the item's front face."""
    middle = geometry.front_point(center, eff, rot)
    along = np.array([1.0, 0.0]) if rot % 2 == 0 else np.array([0.0, 1.0])
    half = eff[0 if rot % 2 == 0 else 1] / 2
    p, q = middle - along * half, middle + along * half
    ax.plot([p[0], q[0]], [p[1], q[1]], color=FRONT_EDGE, linewidth=2.5, solid_capstyle="butt", zorder=3)


def _broken_checks(result: CheckResult) -> dict[int, list[str]]:
    """Slot -> the hard checks it breaks."""
    broken: dict[int, list[str]] = {}
    for check, slots in [("H1", result.items_out_of_room),
                         ("H2", sorted({k for pair in result.overlapping_pairs for k in pair})),
                         ("H3", result.items_in_door_zone),
                         ("H4", result.unreachable_items)]:
        for k in slots:
            broken.setdefault(int(k), []).append(check)
    return broken


def _summary(layout: Layout, catalog: RoomCatalog, result: CheckResult) -> str:
    def status(ok: bool, slots) -> str:
        return "pass" if ok else "fail (" + ", ".join(catalog.slots[k].name.replace("_", " ") for k in slots) + ")"

    room = layout.room_type.replace("_", " ")
    checks = [("H1 in room", result.in_room, result.items_out_of_room),
              ("H2 no overlap", result.no_overlap, sorted({k for p in result.overlapping_pairs for k in p})),
              ("H3 door clear", result.door_clear, result.items_in_door_zone),
              ("H4 reachable", result.reachable, result.unreachable_items)]
    lines = [f"{room}, {layout.width:.2f} x {layout.depth:.2f} m (axes in meters), "
             f"door on the {layout.door_wall} wall"]
    lines += [f"{name}: {status(ok, slots)}" for name, ok, slots in checks]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a layout JSON as a floor-plan PNG.")
    parser.add_argument("layout_json", type=Path)
    parser.add_argument("png", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.layout_json.read_text(encoding="utf-8"))
    catalog = load_room_catalog(data["room"]["type"])
    fig = plot_layout(layout_from_dict(data, catalog), catalog, load_rules())
    fig.savefig(args.png, facecolor=SURFACE)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
