"""The rule-based layout generator: it makes the training data (Set A), and it is the reference
method G0 that every result is compared with.

A Condition is what a layout is made for: the room, its door and the furniture with their
catalog variants. generate_layout() tries up to `attempts` placements for one condition. Each
attempt picks a layout style that fits the room, places the items at rule-driven anchors with
random spacing and jitter, and keeps the layout only if the rule checker passes it (rejection
sampling).

Living-room styles:
  (a) wall sofa      sofa back against a wall, TV unit against the opposite wall
  (b) floating sofa  TV unit against a wall, the sofa facing it 1.5 to 3.5 m away with a
                     walkway behind it
  (c) L-shape        as (a), with the armchair's back against a side wall, facing the coffee table

Items are placed in the frame of the wall behind the sofa: t runs along that wall and n is
the distance from it. Every item placed relative to the sofa (TV unit, side table, coffee
table, armchair) limits where the sofa can stand along its wall without an item leaving the
room or entering the door's clearance zone, and the sofa's position is drawn from what is
left. The bookshelf goes last, on any free stretch of wall. Spacings are drawn uniformly
inside the ranges in configs/rules.yaml, so every kept layout meets every relation and
alignment rule; attempts fail on reachability or when no position is left.

Report the rejection rates:  python -m spacegen.generator --layouts 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spacegen import geometry
from spacegen.catalog import RoomCatalog, Variant, load_room_catalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch, stack_layouts
from spacegen.layout import Layout, canonicalize, make_layout
from spacegen.paths import REPORTS_DIR
from spacegen.quality import Quality, quality_score
from spacegen.rules import WALLS, CheckResult, Rules, check_layout, door_geometry, load_rules, reachability

STYLES = ("a", "b", "c")
AREA_BINS = np.arange(8.0, 60.0, 4.0)  # m^2, for the rejection report

# Rotation class of an item standing with its back to each wall, facing into the room.
_INTO_ROOM = {"N": 2, "E": 3, "S": 0, "W": 1}


@dataclass(frozen=True)
class GeneratorConfig:
    attempts: int  # placements tried per room before it is dropped
    wall_gap: float  # items against a wall stand 0 to wall_gap from it (m)
    jitter: float  # sigma of the position jitter (m)
    walkway: float  # style (b): clear floor behind the floating sofa (m)
    armchair_gap: tuple[float, float]  # armchair to the end of the coffee table (m)
    shelf_clearance: float  # clear floor kept in front of the bookshelf (m)
    optional_area: tuple[float, float]  # floor areas (m^2) between which ...
    optional_probability: tuple[float, float]  # ... the chance of each optional item rises linearly


def load_generator_config(path: Path = DEFAULT_CONFIG) -> GeneratorConfig:
    raw = dict(load_config(path)["generator"])
    optional = raw.pop("optional_items")
    return GeneratorConfig(armchair_gap=tuple(raw.pop("armchair_gap")), optional_area=tuple(optional["area"]),
                           optional_probability=tuple(optional["probability"]), **raw)


@dataclass(frozen=True)
class Condition:
    """A room to furnish: its size, its door, and the items in it with their catalog variants."""
    width: float
    depth: float
    door_wall: str
    door_offset: float
    items: dict[str, str]  # slot name -> variant id

    @property
    def area(self) -> float:
        return self.width * self.depth

    def furniture_area(self, catalog: RoomCatalog) -> float:
        """Total footprint of the items (m^2)."""
        return sum(catalog.slot(name).variant(variant).area for name, variant in self.items.items())

    @classmethod
    def of(cls, layout: Layout, catalog: RoomCatalog) -> Condition:
        """The condition a layout was made for."""
        items = {catalog.slots[k].name: layout.variant_ids[k] for k in np.flatnonzero(layout.mask)}
        return cls(layout.width, layout.depth, layout.door_wall, layout.door_offset, items)


# --------------------------------------------------------------------------- sampling rooms

INF = float("inf")


@dataclass(frozen=True)
class Region:
    """The rooms whose width, depth and floor area all lie in these closed ranges."""
    width: tuple[float, float] = (-INF, INF)
    depth: tuple[float, float] = (-INF, INF)
    area: tuple[float, float] = (-INF, INF)

    def contains(self, width: float, depth: float) -> bool:
        return all(low <= value <= high for value, (low, high)
                   in ((width, self.width), (depth, self.depth), (width * depth, self.area)))


@dataclass(frozen=True)
class RoomRange:
    """W and D drawn uniformly from their ranges, and drawn again until the room lies inside
    `within` and outside every region in `exclude` (the held-out test regions)."""
    width: tuple[float, float]
    depth: tuple[float, float]
    within: Region = Region()
    exclude: tuple[Region, ...] = ()

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        for _ in range(100_000):
            width, depth = float(rng.uniform(*self.width)), float(rng.uniform(*self.depth))
            if self.within.contains(width, depth) and not any(r.contains(width, depth) for r in self.exclude):
                return width, depth
        raise ValueError(f"no room of {self} can be drawn")


def sample_condition(rng: np.random.Generator, catalog: RoomCatalog, rules: Rules, config: GeneratorConfig,
                     rooms: RoomRange | None = None) -> Condition:
    """A random room: its size, its door (wall and position along it) and its furniture.

    `rooms` defaults to the room type's ranges in rules.yaml with nothing excluded.
    """
    if rooms is None:
        room = rules.rooms[catalog.room_type]
        rooms = RoomRange(room.width, room.depth)
    width, depth = rooms.sample(rng)
    door_wall = WALLS[rng.integers(len(WALLS))]
    return Condition(width, depth, door_wall, float(rng.uniform()),
                     sample_furniture(rng, width * depth, catalog, config))


def optional_probability(area: float, config: GeneratorConfig) -> float:
    """The chance that each optional item is in a room of this floor area."""
    return float(np.interp(area, config.optional_area, config.optional_probability))


def sample_furniture(rng: np.random.Generator, area: float, catalog: RoomCatalog,
                     config: GeneratorConfig) -> dict[str, str]:
    """Mandatory items always, each optional item with optional_probability(); variants equally likely."""
    chance = optional_probability(area, config)
    return {slot.name: slot.variants[rng.integers(len(slot.variants))].id
            for slot in catalog.slots if slot.mandatory or rng.uniform() < chance}


# --------------------------------------------------------------------------- styles and placement

@dataclass(frozen=True)
class WallFrame:
    """Coordinates relative to one wall: t along it (x for the S and N walls, y for W and E)
    and n, the distance from it into the room."""
    wall: str
    width: float
    depth: float

    @property
    def length(self) -> float:
        """Extent of the wall along t."""
        return self.width if self.wall in "SN" else self.depth

    @property
    def span(self) -> float:
        """Distance to the opposite wall."""
        return self.depth if self.wall in "SN" else self.width

    def to_room(self, t: float, n: float) -> tuple[float, float]:
        return {"S": (t, n), "N": (t, self.depth - n), "W": (n, t), "E": (self.width - n, t)}[self.wall]

    def to_wall(self, lo, hi) -> tuple[float, float, float, float]:
        """A box given by its lower-left and upper-right room corners, as (t0, t1, n0, n1)."""
        (x0, y0), (x1, y1) = lo, hi
        return {"S": (x0, x1, y0, y1), "N": (x0, x1, self.depth - y1, self.depth - y0),
                "W": (y0, y1, x0, x1), "E": (y0, y1, self.width - x1, self.width - x0)}[self.wall]

    def rotation(self, facing: str) -> int:
        """Rotation class of an item facing "+n" (into the room), "-n", "+t" or "-t"."""
        into = _INTO_ROOM[self.wall]
        plus_t = 1 if self.wall in "SN" else 0  # +x is East, +y is North
        return {"+n": into, "-n": (into + 2) % 4, "+t": plus_t, "-t": (plus_t + 2) % 4}[facing]


def fitting_styles(cond: Condition, catalog: RoomCatalog, rules: Rules,
                   config: GeneratorConfig) -> list[tuple[str, str]]:
    """The (style, wall behind the sofa) pairs whose anchors fit this room.

    (a) and (c) need the sofa-TV distance to land in its range with both against opposite
    walls; (b) needs room for the shortest distance plus the walkway; (c) also needs the
    armchair and the coffee table.
    """
    _check_living_room(catalog)
    low, high = rules.rooms[catalog.room_type].relations["sofa_tv_distance"]
    depths = catalog.slot("sofa").variant(cond.items["sofa"]).d + catalog.slot("tv_unit").variant(cond.items["tv_unit"]).d
    group = {"armchair", "coffee_table"} <= set(cond.items)
    fits = {style: [] for style in STYLES}
    for wall in WALLS:
        between = WallFrame(wall, cond.width, cond.depth).span - depths  # sofa front to TV front, both on walls
        on_walls = between - 2 * config.wall_gap >= low and between <= high
        if on_walls:
            fits["a"].append(wall)
            if group:
                fits["c"].append(wall)
        if between - config.wall_gap - config.walkway >= low:
            fits["b"].append(wall)
    return [(style, wall) for style in STYLES for wall in fits[style]]


@dataclass(frozen=True)
class _Anchor:
    """Where an item goes in the frame of the wall behind the sofa."""
    name: str
    t: float  # along the wall: from the sofa's centre line, or from the wall's start if fixed
    n: float  # distance of the item's centre from the wall
    half: tuple[float, float]  # half extents along t and n
    facing: str
    fixed: bool = False


def _place(cond: Condition, style: str, wall: str, rng: np.random.Generator, catalog: RoomCatalog,
           rules: Rules, config: GeneratorConfig) -> dict[str, tuple[float, float, int]] | None:
    """One attempt: {slot name: (x, y, rotation)}, or None when no position is left for the sofa.

    1. Draw the distances: the sofa's gap to its wall (or, in style (b), the sofa-TV distance),
       the coffee table's gap in front of the sofa, and small random shifts.
    2. Place every item relative to the sofa ("anchors" in the frame of the sofa's wall).
    3. Find where along the wall the sofa can stand so that no item leaves the room or enters
       the door zone, and draw the sofa's position from that.
    4. Put the bookshelf on a free stretch of any wall."""
    relations = rules.rooms[catalog.room_type].relations
    frame = WallFrame(wall, cond.width, cond.depth)
    door = door_geometry(cond.width, cond.depth, cond.door_wall, cond.door_offset, rules.door)
    zone = [geometry.box_bounds(door.zone_center, door.zone_size)]

    def variant(name: str) -> Variant:  # the room's variant; the default one for an absent item
        return catalog.slot(name).variant(cond.items.get(name))

    sofa, tv, table, side_table, chair = map(variant, ("sofa", "tv_unit", "coffee_table", "side_table", "armchair"))
    gap, jitter = config.wall_gap, config.jitter

    # 1. The distances from the sofa's wall (n) and along it (t).
    tv_gap = rng.uniform(0, gap)
    if style == "b":  # draw the sofa-TV distance; the walkway behind the sofa takes the rest
        low, high = relations["sofa_tv_distance"]
        distance = rng.uniform(low, min(high, frame.span - sofa.d - tv.d - tv_gap - config.walkway))
        back = frame.span - tv_gap - tv.d - distance - sofa.d
    else:
        back = rng.uniform(0, gap)
    # where the coffee table stands; an armchair is placed as if it were there when it is not
    table_n = back + sofa.d + rng.uniform(*relations["coffee_table_gap"]) + table.d / 2
    table_t = rng.normal(0, jitter)
    side = int(rng.choice((-1, 1)))  # the side table's end of the sofa; the armchair takes the other

    # 2. Every item relative to the sofa's centre line (t) and to the wall (n).
    anchors = [_Anchor("sofa", 0.0, back + sofa.d / 2, (sofa.w / 2, sofa.d / 2), "+n"),
               _Anchor("tv_unit", rng.normal(0, jitter), frame.span - tv_gap - tv.d / 2, (tv.w / 2, tv.d / 2), "-n")]
    if "side_table" in cond.items:
        t = side * (sofa.w / 2 + rng.uniform(0, relations["side_table_max_gap"]) + side_table.w / 2)
        anchors.append(_Anchor("side_table", t, back + rng.uniform(0, gap) + side_table.d / 2,
                               (side_table.w / 2, side_table.d / 2), "+n"))
    if "coffee_table" in cond.items:
        anchors.append(_Anchor("coffee_table", table_t, table_n, (table.w / 2, table.d / 2), "+n"))
    axis = [(0.0, frame.length)]  # positions of the sofa's centre line along the wall
    if "armchair" in cond.items:
        chair_n, half = table_n + rng.normal(0, jitter), (chair.d / 2, chair.w / 2)  # it faces along the wall
        if style == "c":  # back to the side wall at the armchair's end; its gap to the table limits the axis
            front = rng.uniform(0, gap) + chair.d  # the armchair's front face, from that wall
            near, far = (front + g + table.w / 2 for g in config.armchair_gap)  # coffee-table centre, from that wall
            if side > 0:  # side table at the +t end: the armchair stands at the t = 0 wall
                anchors.append(_Anchor("armchair", front - chair.d / 2, chair_n, half, "+t", fixed=True))
                axis = [(near - table_t, far - table_t)]
            else:
                anchors.append(_Anchor("armchair", frame.length - front + chair.d / 2, chair_n, half, "-t", fixed=True))
                axis = [(frame.length - far - table_t, frame.length - near - table_t)]
        else:
            t = table_t - side * (table.w / 2 + rng.uniform(*config.armchair_gap) + chair.d / 2)
            anchors.append(_Anchor("armchair", t, chair_n, half, "+t" if t < 0 else "-t"))

    # 3. Where the sofa can stand: each item must stay in the room and out of the door zone.
    for anchor in anchors:
        allowed = _centres(_free_stretches(frame, anchor.n - anchor.half[1], anchor.n + anchor.half[1], zone),
                           anchor.half[0])
        if anchor.fixed:
            if not any(a <= anchor.t <= b for a, b in allowed):
                return None
        else:
            axis = _intersect(axis, [(a - anchor.t, b - anchor.t) for a, b in allowed])
    centre = _uniform(axis, rng)
    if centre is None:
        return None
    items = {a.name: (*frame.to_room(a.t if a.fixed else centre + a.t, a.n), frame.rotation(a.facing))
             for a in anchors}
    # 4. The bookshelf, against any free stretch of wall.
    if "bookshelf" in cond.items:
        shelf = _shelf(cond, variant("bookshelf"), items, zone, rng, catalog, config)
        if shelf is None:
            return None
        items["bookshelf"] = shelf
    return items


def _shelf(cond: Condition, shelf: Variant, items: dict, zone: list, rng: np.random.Generator,
           catalog: RoomCatalog, config: GeneratorConfig) -> tuple[float, float, int] | None:
    """The bookshelf against any stretch of wall with room for it and clear floor in front."""
    boxes = zone + [_box(catalog.slot(name).variant(cond.items[name]), x, y, r) for name, (x, y, r) in items.items()]
    gap = rng.uniform(0, config.wall_gap)
    options = []  # (wall, lowest and highest position of the shelf's centre along it)
    for wall in WALLS:
        frame = WallFrame(wall, cond.width, cond.depth)
        clear = _free_stretches(frame, 0.0, gap + shelf.d + config.shelf_clearance, boxes)  # room for it and in front
        for a, b in _centres(clear, shelf.w / 2):
            options.append((frame, a, b))
    if not options:
        return None
    frame, a, b = options[_pick_by_length([(a, b) for _, a, b in options], rng)]
    return (*frame.to_room(rng.uniform(a, b), gap + shelf.d / 2), frame.rotation("+n"))


def _box(variant: Variant, x: float, y: float, rot: int):
    eff = geometry.effective_size(np.array([variant.w, variant.d]), rot)
    return geometry.box_bounds(np.array([x, y]), eff)


# --------------------------------------------------------------------------- stretches of wall

def _free_stretches(frame: WallFrame, n0: float, n1: float, boxes) -> list[tuple[float, float]]:
    """The t ranges along the wall where the band from n0 to n1 is clear of every box."""
    stretches = [(0.0, frame.length)]
    for lo, hi in boxes:
        t0, t1, m0, m1 = frame.to_wall(lo, hi)
        if m0 < n1 and m1 > n0:  # the box reaches into the band: cut [t0, t1] out of every stretch
            remaining = []
            for a, b in stretches:
                for part in ((a, min(b, t0)), (max(a, t1), b)):  # what is left before and after the box
                    if part[1] > part[0]:
                        remaining.append(part)
            stretches = remaining
    return stretches


def _centres(stretches, half: float) -> list[tuple[float, float]]:
    """Where the centre of something reaching `half` to either side can be."""
    return [(a + half, b - half) for a, b in stretches if b - a >= 2 * half]


def _intersect(first, second) -> list[tuple[float, float]]:
    return [(max(a0, b0), min(a1, b1)) for a0, a1 in first for b0, b1 in second if max(a0, b0) <= min(a1, b1)]


def _pick_by_length(stretches, rng: np.random.Generator) -> int:
    """Index of a stretch, chosen with probability proportional to its length."""
    lengths = np.array([b - a for a, b in stretches]) + 1e-12  # single points stay possible
    return int(rng.choice(len(stretches), p=lengths / lengths.sum()))


def _uniform(stretches, rng: np.random.Generator) -> float | None:
    """A point drawn uniformly from the stretches, or None if there are none."""
    if not stretches:
        return None
    a, b = stretches[_pick_by_length(stretches, rng)]
    return float(rng.uniform(a, b))


# --------------------------------------------------------------------------- generation

@dataclass(frozen=True)
class Attempt:
    style: str | None  # None when no style fits the room
    wall: str | None  # the wall behind the sofa
    outcome: str  # "valid", "invalid", "no position" or "no style"
    failed: tuple[str, ...] = ()  # hard checks an invalid layout breaks


@dataclass(frozen=True)
class Generated:
    layout: Layout | None  # the first valid layout, in canonical form; None if every attempt failed
    style: str | None
    quality: Quality | None
    attempts: tuple[Attempt, ...]


def generate_layout(cond: Condition, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                    config: GeneratorConfig) -> Generated:
    """Up to config.attempts placements for one condition, keeping the first valid one (G0)."""
    options = fitting_styles(cond, catalog, rules, config)
    if not options:
        return Generated(None, None, None, (Attempt(None, None, "no style"),))
    styles = sorted({style for style, _ in options})
    attempts = []
    for _ in range(config.attempts):
        style = styles[rng.integers(len(styles))]
        walls = [wall for s, wall in options if s == style]
        wall = walls[rng.integers(len(walls))]
        items = _place(cond, style, wall, rng, catalog, rules, config)
        if items is None:
            attempts.append(Attempt(style, wall, "no position"))
            continue
        layout = canonicalize(make_layout(catalog, cond.width, cond.depth, cond.door_wall, cond.door_offset,
                                          items, cond.items), catalog)
        reach = reachability(layout, catalog, rules)
        result = check_layout(layout, catalog, rules, reach=reach)
        if result.valid:
            attempts.append(Attempt(style, wall, "valid"))
            return Generated(layout, style, quality_score(layout, catalog, rules, reach=reach), tuple(attempts))
        attempts.append(Attempt(style, wall, "invalid", failed_checks(result)))
    return Generated(None, None, None, tuple(attempts))


@dataclass(frozen=True)
class GeneratedSet:
    layouts: LayoutBatch
    info: pd.DataFrame  # one row per layout: its room, style, attempts used and quality terms
    attempts: pd.DataFrame  # one row per attempt, for every room tried, kept or dropped


def generate_set_a(n_layouts: int, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                   config: GeneratorConfig, rooms: RoomRange | None = None) -> GeneratedSet:
    """Set A: draw rooms (from `rooms`, see sample_condition) until n_layouts of them have a
    valid layout."""
    layouts, info, log = [], [], []
    room = 0
    while len(layouts) < n_layouts:
        if room >= 10 * n_layouts + 100:
            raise RuntimeError(f"only {len(layouts)} of {n_layouts} layouts after {room} rooms; check the settings")
        cond = sample_condition(rng, catalog, rules, config, rooms)
        result = generate_layout(cond, rng, catalog, rules, config)
        log += [{"room": room, "width": cond.width, "depth": cond.depth, "area": cond.area, "items": len(cond.items),
                 "attempt": i, "style": a.style, "wall": a.wall, "outcome": a.outcome, "failed": " ".join(a.failed)}
                for i, a in enumerate(result.attempts, start=1)]
        if result.layout is not None:
            q = result.quality
            info.append({"room": room, "style": result.style, "attempts": len(result.attempts), "quality": q.total,
                         "align": q.align, "relations": q.relations, "circulation": q.circulation, "space": q.space})
            layouts.append(result.layout)
        room += 1
    return GeneratedSet(stack_layouts(layouts, catalog), pd.DataFrame(info), pd.DataFrame(log))


def rejection_summary(attempts: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per room-area bin (by="area") or item count (by="items"): attempts, share rejected,
    rooms tried and rooms dropped after failing every attempt."""
    group = pd.cut(attempts["area"], AREA_BINS, right=False) if by == "area" else attempts["items"]
    rows = attempts.assign(group=group, rejected=attempts["outcome"] != "valid")
    per_attempt = rows.groupby("group", observed=True).agg(attempts=("rejected", "size"), rejected=("rejected", "mean"))
    kept = rows.groupby("room").agg(group=("group", "first"), kept=("rejected", lambda r: not r.all()))
    per_room = kept.groupby("group", observed=True).agg(rooms=("kept", "size"), dropped=("kept", lambda k: int((~k).sum())))
    return per_attempt.join(per_room)


def failed_checks(result: CheckResult) -> tuple[str, ...]:
    flags = (("H1", result.in_room), ("H2", result.no_overlap), ("H3", result.door_clear), ("H4", result.reachable))
    return tuple(name for name, passed in flags if not passed)


def _check_living_room(catalog: RoomCatalog) -> None:
    if catalog.room_type != "living_room":
        raise NotImplementedError(f"no layout styles for {catalog.room_type!r} (only the living room has them)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Set A living-room layouts and report how often attempts fail.")
    parser.add_argument("--layouts", type=int, default=2000, help="layouts to generate (default 2000)")
    parser.add_argument("--seed", type=int, default=None, help="default: the seed in configs/default.yaml")
    parser.add_argument("--figure", type=Path, default=REPORTS_DIR / "figures" / "generator_rejection.png")
    args = parser.parse_args(argv)
    from spacegen.viz import SURFACE, plot_rejection

    seed = load_config()["seed"] if args.seed is None else args.seed
    catalog, rules, config = load_room_catalog("living_room"), load_rules(), load_generator_config()
    start = time.perf_counter()
    result = generate_set_a(args.layouts, np.random.default_rng(seed), catalog, rules, config)
    seconds = time.perf_counter() - start
    log, info = result.attempts, result.info
    print(f"{len(info)} layouts from {log['room'].nunique()} rooms and {len(log)} attempts in {seconds:.1f} s "
          f"({1e3 * seconds / len(log):.1f} ms per attempt); {(log['outcome'] == 'valid').mean():.1%} of attempts kept")
    print("outcomes:", log["outcome"].value_counts().to_dict(), "| failed checks:",
          log.loc[log["failed"] != "", "failed"].value_counts().to_dict())
    print("styles:", info["style"].value_counts(normalize=True).round(3).to_dict())
    print(info[["quality", "align", "relations", "circulation", "space"]].describe().loc[["mean", "min", "max"]].round(3))
    for by in ("area", "items"):
        print(rejection_summary(log, by).round(3).to_string())
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot_rejection(log, config.attempts).savefig(args.figure, facecolor=SURFACE)
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
