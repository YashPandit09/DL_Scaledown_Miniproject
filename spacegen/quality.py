"""The soft quality score S in [0, 1]: how good a layout is, beyond being valid.

S = 0.30 S_align + 0.30 S_rel + 0.25 S_circ + 0.15 S_space, with the weights and every threshold
in configs/rules.yaml (all of them our own assumptions):

- S_align (alignment): share of the against-wall items (TV unit, bookshelf) standing within
  wall_distance of a wall with their back to it. Symmetric items have no back, so they count
  by position only.
- S_rel (relations): mean score of the relations that apply. The sofa and TV unit face each
  other 1.5 to 3.5 m apart; the coffee table stands 0.35 to 0.5 m in front of the sofa; the
  side table beside an end of the sofa; the armchair at 90 degrees to the sofa, facing the
  coffee table. A relation is skipped when one of its items is absent. Each scores 1 inside its
  range and falls linearly to 0 at relation_falloff outside it.
- S_circ (circulation): share of the free floor (cells outside every footprint) that is walkable.
- S_space: 1 while furniture covers 15 to 40% of the floor (space_band), falling linearly to 0
  at space_falloff outside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spacegen import geometry
from spacegen.catalog import RoomCatalog
from spacegen.layout import Layout
from spacegen.rules import WALLS, Reachability, Rules, reachability


@dataclass(frozen=True)
class Quality:
    total: float
    align: float
    relations: float
    circulation: float
    space: float
    relation_scores: dict[str, float]  # one entry per relation that applies


def quality_score(layout: Layout, catalog: RoomCatalog, rules: Rules,
                  reach: Reachability | None = None) -> Quality:
    """The quality score and its four terms. Pass `reach` to reuse an existing reachability grid."""
    room, q = rules.rooms[layout.room_type], rules.quality
    if reach is None:
        reach = reachability(layout, catalog, rules)
    relation_scores = _relation_scores(layout, catalog, room.relations, q.relation_falloff)
    terms = {
        "align": _align_score(layout, catalog, room.against_wall, q.wall_distance),
        "relations": float(np.mean(list(relation_scores.values()))) if relation_scores else 1.0,
        "circulation": float(reach.passable.sum() / max(reach.free.sum(), 1)),
        "space": range_score(furniture_share(layout), *q.space_band, q.space_falloff),
    }
    total = sum(q.weights[name] * value for name, value in terms.items())
    return Quality(total=float(total), relation_scores=relation_scores, **terms)


def range_score(value: float, low: float, high: float, falloff: float) -> float:
    """1 inside [low, high], falling linearly to 0 at `falloff` outside it."""
    outside = max(low - value, value - high, 0.0)
    return max(0.0, 1.0 - outside / falloff)


def furniture_share(layout: Layout) -> float:
    """Footprint area of the present items divided by the room's floor area."""
    eff = layout.eff_size[layout.mask]
    return float((eff[:, 0] * eff[:, 1]).sum() / (layout.width * layout.depth))


def _align_score(layout: Layout, catalog: RoomCatalog, names: tuple[str, ...], wall_distance: float) -> float:
    lo, hi = geometry.box_bounds(layout.center, layout.eff_size)
    aligned = []
    for name in names:
        k = catalog.slot(name).index
        if not layout.mask[k]:
            continue
        to_wall = {"W": lo[k, 0], "S": lo[k, 1], "E": layout.width - hi[k, 0], "N": layout.depth - hi[k, 1]}
        if catalog.slots[k].rot_symmetry == 1:
            behind = WALLS[(int(layout.rot[k]) + 2) % 4]  # an item facing North has its back to the S wall
            aligned.append(to_wall[behind] <= wall_distance)
        else:
            aligned.append(min(to_wall.values()) <= wall_distance)
    return float(np.mean(aligned)) if aligned else 1.0


def _relation_scores(layout: Layout, catalog: RoomCatalog, relations: dict, falloff: float) -> dict[str, float]:
    present = {s.name: s.index for s in catalog.slots if layout.mask[s.index]}
    scores = {}
    for name, value in relations.items():
        if name not in _RELATIONS:
            raise KeyError(f"no scoring rule for relation {name!r}")
        items, score = _RELATIONS[name]
        if all(item in present for item in items):
            low, high = value if isinstance(value, (list, tuple)) else (0.0, value)
            scores[name] = score(layout, *(present[item] for item in items), low, high, falloff)
    return scores


# --------------------------------------------------------------------------- relations

def _seen_from(layout: Layout, viewer: int, item: int) -> tuple[float, float]:
    """Where `item`'s centre lies seen from `viewer`: distance along the viewer's facing, and across it."""
    facing = geometry.facing_vector(int(layout.rot[viewer]), layout.center)
    offset = layout.center[item] - layout.center[viewer]
    return float(offset @ facing), float(offset[0] * facing[1] - offset[1] * facing[0])


def _extents(layout: Layout, item: int, viewer: int) -> tuple[float, float]:
    """`item`'s extent along `viewer`'s facing direction and across it (m)."""
    axis = 1 if layout.rot[viewer] % 2 == 0 else 0
    return float(layout.eff_size[item, axis]), float(layout.eff_size[item, 1 - axis])


def _gap(layout: Layout, i: int, j: int) -> float:
    """Shortest distance between two footprints (0 if they touch or overlap)."""
    lo, hi = geometry.box_bounds(layout.center[[i, j]], layout.eff_size[[i, j]])
    separation = np.maximum(np.maximum(lo[1] - hi[0], lo[0] - hi[1]), 0.0)
    return float(np.hypot(*separation))


def _sofa_tv(layout, sofa, tv, low, high, falloff) -> float:
    """Sofa and TV unit face each other, the TV within the sofa's width; scored front face to front face."""
    along, across = _seen_from(layout, sofa, tv)
    sofa_depth, sofa_width = _extents(layout, sofa, sofa)
    tv_depth, _ = _extents(layout, tv, sofa)
    facing = layout.rot[tv] == (layout.rot[sofa] + 2) % 4 and along > 0 and abs(across) <= sofa_width / 2
    return range_score(along - sofa_depth / 2 - tv_depth / 2, low, high, falloff) if facing else 0.0


def _coffee_table(layout, sofa, table, low, high, falloff) -> float:
    """The coffee table stands in front of the sofa; scored on the gap from the sofa's front face."""
    along, across = _seen_from(layout, sofa, table)
    sofa_depth, sofa_width = _extents(layout, sofa, sofa)
    table_depth, _ = _extents(layout, table, sofa)
    in_front = along > 0 and abs(across) <= sofa_width / 2
    return range_score(along - sofa_depth / 2 - table_depth / 2, low, high, falloff) if in_front else 0.0


def _side_table(layout, sofa, table, low, high, falloff) -> float:
    """The side table stands beside an end of the sofa; scored on the gap to that end."""
    along, across = _seen_from(layout, sofa, table)
    sofa_depth, sofa_width = _extents(layout, sofa, sofa)
    table_depth, table_width = _extents(layout, table, sofa)
    beside = abs(along) < (sofa_depth + table_depth) / 2  # their front-to-back ranges overlap
    return range_score(abs(across) - sofa_width / 2 - table_width / 2, low, high, falloff) if beside else 0.0


def _armchair(layout, sofa, chair, table, low, high, falloff) -> float:
    """The armchair stands at 90 degrees to the sofa, facing the coffee table; scored on its gap to the table."""
    perpendicular = (layout.rot[chair] - layout.rot[sofa]) % 2 == 1
    faces_table = _seen_from(layout, chair, table)[0] > 0
    return range_score(_gap(layout, chair, table), low, high, falloff) if perpendicular and faces_table else 0.0


# Relation name in rules.yaml -> (items it needs, scoring function).
_RELATIONS = {
    "sofa_tv_distance": (("sofa", "tv_unit"), _sofa_tv),
    "coffee_table_gap": (("sofa", "coffee_table"), _coffee_table),
    "side_table_max_gap": (("sofa", "side_table"), _side_table),
    "armchair_max_gap": (("sofa", "armchair", "coffee_table"), _armchair),
}
