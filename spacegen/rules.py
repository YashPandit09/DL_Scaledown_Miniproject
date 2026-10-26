"""The rule engine: the rules file, the door's geometry, and the hard checks H1 to H4.

A layout is valid when it passes all four hard checks:
  H1  every item lies inside the room
  H2  no two items overlap (by more than 0.005 m^2)
  H3  no item stands in the door's clearance zone
  H4  a person can walk from the door to every item that needs access (reachability)

The checker is exact and deterministic, and it alone decides validity: no layout from a
network is called valid without passing it. Every threshold is in configs/rules.yaml, and all
of them are our own design assumptions, not an official standard. The soft quality score is in
spacegen/quality.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

from spacegen import geometry
from spacegen.catalog import RoomCatalog
from spacegen.layout import Layout
from spacegen.paths import CONFIG_DIR

RULES_PATH = CONFIG_DIR / "rules.yaml"

# Walls in the order of the rotation classes: the N wall's outward normal is facing class 0.
WALLS = ("N", "E", "S", "W")

# Walking the walls counter-clockwise: for each wall, its start corner as a fraction of (W, D),
# the direction along the wall, and the inward normal.
_WALL_FRAME = {
    "S": ((0, 0), (1, 0), (0, 1)),
    "E": ((1, 0), (0, 1), (-1, 0)),
    "N": ((1, 1), (-1, 0), (0, -1)),
    "W": ((0, 1), (0, -1), (1, 0)),
}

_QUALITY_TERMS = {"align", "relations", "circulation", "space"}


class RulesError(ValueError):
    """The rules file breaks one of the checks in load_rules()."""


@dataclass(frozen=True)
class DoorRules:
    width: float
    clearance_depth: float
    corner_margin: float  # the door's centre stays this far from a corner


@dataclass(frozen=True)
class HardCheckRules:
    in_room_tol: float
    max_pair_overlap: float
    door_overlap_tol: float


@dataclass(frozen=True)
class ReachabilityRules:
    cell: float
    min_path_width: float
    access_offset: float
    access_tolerance: float


@dataclass(frozen=True)
class QualityRules:
    weights: dict[str, float]
    wall_distance: float
    space_band: tuple[float, float]
    space_falloff: float
    relation_falloff: float


@dataclass(frozen=True)
class RoomRules:
    width: tuple[float, float]  # sampling range of W
    depth: tuple[float, float]  # sampling range of D
    against_wall: tuple[str, ...]
    relations: dict


@dataclass(frozen=True)
class Rules:
    door: DoorRules
    hard_checks: HardCheckRules
    reachability: ReachabilityRules
    f_max: float
    quality: QualityRules
    rooms: dict[str, RoomRules]


def load_rules(path: Path = RULES_PATH) -> Rules:
    """Read and check the rules file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        quality = dict(raw["quality"])
        rules = Rules(
            door=DoorRules(**raw["door"]),
            hard_checks=HardCheckRules(**raw["hard_checks"]),
            reachability=ReachabilityRules(**raw["reachability"]),
            f_max=raw["feasibility"]["f_max"],
            quality=QualityRules(weights=dict(quality.pop("weights")),
                                 space_band=tuple(quality.pop("space_band")), **quality),
            rooms={name: RoomRules(width=tuple(r["width"]), depth=tuple(r["depth"]),
                                   against_wall=tuple(r["against_wall"]), relations=dict(r["relations"]))
                   for name, r in raw["rooms"].items()},
        )
    except (KeyError, TypeError) as e:
        raise RulesError(f"missing or unexpected field ({e})") from e
    _check(rules)
    return rules


def _check(rules: Rules) -> None:
    door = rules.door
    if min(door.width, door.clearance_depth) <= 0:
        raise RulesError("door width and clearance depth must be positive")
    if door.corner_margin < door.width / 2:
        raise RulesError("door.corner_margin must be at least half the door width")
    weights = rules.quality.weights
    if set(weights) != _QUALITY_TERMS or min(weights.values()) < 0 or abs(sum(weights.values()) - 1) > 1e-9:
        raise RulesError(f"quality.weights needs the terms {sorted(_QUALITY_TERMS)}, non-negative, summing to 1")
    _check_range("quality.space_band", rules.quality.space_band)
    if min(rules.quality.space_falloff, rules.quality.relation_falloff) <= 0:
        raise RulesError("quality.space_falloff and quality.relation_falloff must be positive")
    for name, room in rules.rooms.items():
        _check_range(f"{name}.width", room.width)
        _check_range(f"{name}.depth", room.depth)
        if min(room.width[0], room.depth[0]) < 2 * door.corner_margin:
            raise RulesError(f"{name}: the shortest wall must fit a door (at least {2 * door.corner_margin} m)")
        for relation, value in room.relations.items():
            if isinstance(value, list):
                _check_range(f"{name}.relations.{relation}", value)


def _check_range(name: str, value) -> None:
    if len(value) != 2 or not 0 <= value[0] < value[1]:
        raise RulesError(f"{name} must be [low, high] with 0 <= low < high, got {list(value)}")


def footprint_ratio(furniture_area: float, width: float, depth: float, door: DoorRules) -> float:
    """Furniture footprint over the floor outside the door's clearance zone. The pipeline turns a
    request away when this is above rules.f_max: the room is too crowded to furnish."""
    return furniture_area / (width * depth - door.width * door.clearance_depth)


# --------------------------------------------------------------------------- door

@dataclass(frozen=True, eq=False)
class DoorGeometry:
    center: np.ndarray  # middle of the door opening, on the wall
    inward: np.ndarray  # unit vector from the door into the room
    zone_center: np.ndarray  # clearance zone: the door's width along the wall, clearance depth into the room
    zone_size: np.ndarray  # its extent along x and y


def door_geometry(width: float, depth: float, wall: str, offset: float, door: DoorRules) -> DoorGeometry:
    """The door and its clearance zone.

    The door centre lies s = m + (L - 2m) * offset from the wall's start corner (m is the corner
    margin, L the wall's length), so it stays at least m from both corners."""
    if wall not in _WALL_FRAME:
        raise ValueError(f"door wall must be one of {WALLS}, got {wall!r}")
    if not 0.0 <= offset <= 1.0:
        raise ValueError(f"door offset must be in [0, 1], got {offset}")
    length = width if wall in ("S", "N") else depth
    if length < 2 * door.corner_margin:
        raise ValueError(f"the {wall} wall ({length} m) is too short for a door (needs {2 * door.corner_margin} m)")
    start, along, inward = (np.array(v, dtype=float) for v in _WALL_FRAME[wall])
    s = door.corner_margin + (length - 2 * door.corner_margin) * offset
    center = start * np.array([width, depth]) + along * s
    zone_size = np.abs(along) * door.width + np.abs(inward) * door.clearance_depth
    return DoorGeometry(center, inward, center + inward * door.clearance_depth / 2, zone_size)


# --------------------------------------------------------------------------- hard checks

@dataclass(frozen=True)
class CheckResult:
    """Which hard checks a layout passes, with the slots that break each one."""

    items_out_of_room: tuple[int, ...]  # H1
    overlapping_pairs: tuple[tuple[int, int], ...]  # H2
    items_in_door_zone: tuple[int, ...]  # H3
    unreachable_items: tuple[int, ...]  # H4
    reachability_ratio: float  # reachable items / items that need access

    @property
    def in_room(self) -> bool:
        return not self.items_out_of_room

    @property
    def no_overlap(self) -> bool:
        return not self.overlapping_pairs

    @property
    def door_clear(self) -> bool:
        return not self.items_in_door_zone

    @property
    def reachable(self) -> bool:
        return not self.unreachable_items

    @property
    def valid(self) -> bool:
        """All of H1 to H4 hold. (The budget is checked earlier, when the variants are chosen.)"""
        return self.in_room and self.no_overlap and self.door_clear and self.reachable


def check_layout(layout: Layout, catalog: RoomCatalog, rules: Rules,
                 reach: Reachability | None = None) -> CheckResult:
    """Run the hard checks H1 to H4 on the present items.

    Pass `reach` if reachability() was already computed for this layout, to reuse it."""
    _check_room_type(layout, catalog)
    limits = rules.hard_checks
    present, eff = layout.mask, layout.eff_size

    # H1: inside the room
    inside = geometry.inside_room(layout.center, eff, layout.room, tol=limits.in_room_tol)
    out_of_room = np.flatnonzero(present & ~inside)

    # H2: no overlapping pair (the exact intersection area of every pair, upper triangle only)
    pair_area = geometry.pairwise_overlap_area(layout.center, eff, present)
    first, second = np.nonzero(np.triu(pair_area > limits.max_pair_overlap, k=1))

    # H3: the door's clearance zone is free
    door = door_geometry(layout.width, layout.depth, layout.door_wall, layout.door_offset, rules.door)
    zone_area = geometry.overlap_area(layout.center, eff, door.zone_center, door.zone_size)
    in_zone = np.flatnonzero(present & (zone_area > limits.door_overlap_tol))

    # H4: every item that needs access can be walked to from the door
    if reach is None:
        reach = reachability(layout, catalog, rules)
    need_access = sum(catalog.slots[k].needs_access for k in np.flatnonzero(present))

    return CheckResult(
        items_out_of_room=tuple(out_of_room.tolist()),
        overlapping_pairs=tuple(zip(first.tolist(), second.tolist())),
        items_in_door_zone=tuple(in_zone.tolist()),
        unreachable_items=reach.unreachable,
        reachability_ratio=1.0 - len(reach.unreachable) / need_access if need_access else 1.0,
    )


def _check_room_type(layout: Layout, catalog: RoomCatalog) -> None:
    if layout.room_type != catalog.room_type:
        raise ValueError(f"a {layout.room_type} layout needs the {layout.room_type} catalog, got {catalog.room_type}")


# --------------------------------------------------------------------------- reachability (H4)

@dataclass(frozen=True, eq=False)
class Reachability:
    """The walkable grid of one layout and the items that cannot be reached."""

    xs: np.ndarray  # (nx,) cell centres along x
    ys: np.ndarray  # (ny,) cell centres along y
    free: np.ndarray  # (nx, ny) cell centre outside every footprint (free floor)
    passable: np.ndarray  # (nx, ny) at least min_path_width / 2 from every footprint and wall
    reached: np.ndarray  # (nx, ny) passable and connected to the door
    unreachable: tuple[int, ...]  # slots that need access but cannot be walked up to


def reachability(layout: Layout, catalog: RoomCatalog, rules: Rules) -> Reachability:
    """H4: which items a person can walk up to from the door.

    1. Cover the room with a grid of cells of at most `cell` meters (10 cm).
    2. A cell is passable if its centre is at least min_path_width / 2 from every footprint and
       every wall, so a 0.6 m wide person fits. Distances are exact point-to-box distances, and
       the door opening counts as free wall.
    3. The reached region is the group of passable cells connected (4-connectivity, found with
       scipy.ndimage.label) to the cell at the centre of the door's clearance zone.
    4. An item is reachable if a reached cell lies within access_tolerance of its access line:
       the item's front face, moved access_offset in front of it. A symmetric item has an access
       line on every equivalent face, because its stored rotation is arbitrary among them.

    The whole front face counts, not only the point in front of its middle: with a coffee table
    0.35 to 0.50 m in front of a sofa, that point always falls in the narrow gap between them,
    and every such sofa would fail.

    Cells are sampled at their centres, so a corridor narrower than min_path_width always blocks,
    one at least min_path_width + cell wide always passes, and one in between depends on how it
    lines up with the grid."""
    _check_room_type(layout, catalog)
    params = rules.reachability
    nx = max(1, int(np.ceil(layout.width / params.cell - 1e-9)))
    ny = max(1, int(np.ceil(layout.depth / params.cell - 1e-9)))
    xs = (np.arange(nx) + 0.5) * (layout.width / nx)
    ys = (np.arange(ny) + 0.5) * (layout.depth / ny)
    px, py = np.meshgrid(xs, ys, indexing="ij")

    door = door_geometry(layout.width, layout.depth, layout.door_wall, layout.door_offset, rules.door)
    to_items = _distance_to_items(px, py, layout)
    clearance = np.minimum(_distance_to_walls(px, py, layout, door, rules.door.width), to_items)
    passable = clearance >= params.min_path_width / 2

    labels, _ = ndimage.label(passable)  # connected groups of passable cells (4-connectivity, the 2D default)
    start_i = min(int(door.zone_center[0] / layout.width * nx), nx - 1)
    start_j = min(int(door.zone_center[1] / layout.depth * ny), ny - 1)
    # Blocked cells carry label 0, so a blocked start reaches nothing.
    reached = passable & (labels == labels[start_i, start_j])

    reached_points = np.stack([px[reached], py[reached]], axis=-1)
    eff = layout.eff_size
    unreachable = []
    for k in np.flatnonzero(layout.mask):
        slot = catalog.slots[k]
        if not slot.needs_access:
            continue
        lines = _access_lines(layout.center[k], eff[k], int(layout.rot[k]), slot.rot_symmetry, params.access_offset)
        if not any(_near_line(reached_points, *line, params.access_tolerance) for line in lines):
            unreachable.append(int(k))
    return Reachability(xs, ys, to_items > 0, passable, reached, tuple(unreachable))


def _distance_to_items(px: np.ndarray, py: np.ndarray, layout: Layout) -> np.ndarray:
    """Distance from each point to the nearest present footprint (0 inside one)."""
    lo, hi = geometry.box_bounds(layout.center[layout.mask], layout.eff_size[layout.mask])
    if len(lo) == 0:
        return np.full(px.shape, np.inf)
    dx = np.maximum(np.maximum(lo[:, 0] - px[..., None], 0.0), px[..., None] - hi[:, 0])
    dy = np.maximum(np.maximum(lo[:, 1] - py[..., None], 0.0), py[..., None] - hi[:, 1])
    return np.hypot(dx, dy).min(axis=-1)


def _distance_to_walls(px: np.ndarray, py: np.ndarray, layout: Layout, door: DoorGeometry,
                       door_width: float) -> np.ndarray:
    """Distance from each point to the nearest wall, with the door opening left out."""
    distance = {"W": px, "E": layout.width - px, "S": py, "N": layout.depth - py}
    wall = layout.door_wall
    along, middle = (px, door.center[0]) if wall in ("S", "N") else (py, door.center[1])
    to_jamb = door_width / 2 - np.abs(along - middle)  # positive inside the opening
    # Inside the opening the nearest wall point is a door jamb, not the wall line.
    distance[wall] = np.where(to_jamb > 0, np.hypot(distance[wall], to_jamb), distance[wall])
    return np.minimum.reduce(list(distance.values()))


def _access_lines(center: np.ndarray, eff: np.ndarray, rot: int, rot_symmetry: int, offset: float):
    """For every equivalent face: (midpoint, half-length, axis the face runs along)."""
    for r in geometry.equivalent_rotations(rot, rot_symmetry):
        along = 0 if r % 2 == 0 else 1  # a face looking North or South runs along x
        yield geometry.front_point(center, eff, r, offset), eff[along] / 2, along


def _near_line(points: np.ndarray, middle: np.ndarray, half: float, along: int, tol: float) -> bool:
    """Whether any point lies within `tol` of the axis-aligned segment."""
    beyond_end = np.maximum(np.abs(points[:, along] - middle[along]) - half, 0.0)
    across = np.abs(points[:, 1 - along] - middle[1 - along])
    return bool(np.any(np.hypot(beyond_end, across) <= tol))
