"""Rule tests: the rules file, the door geometry and the hard checks H1 to H3."""
import numpy as np
import pytest
import yaml

from tests.layouts import ARMCHAIR, BOOKSHELF, SIDE_TABLE, SOFA
from spacegen.geometry import box_bounds
from spacegen.layout import make_layout
from spacegen.rules import (RULES_PATH, WALLS, DoorRules, RulesError, check_layout, door_geometry,
                            load_rules)

# --------------------------------------------------------------------------- rules file


def test_rules_file_holds_the_spec_values(rules):
    assert (rules.door.width, rules.door.clearance_depth, rules.door.corner_margin) == (0.9, 0.9, 0.65)
    limits = rules.hard_checks
    assert (limits.in_room_tol, limits.max_pair_overlap, limits.door_overlap_tol) == (1e-3, 0.005, 1e-4)
    assert rules.reachability.min_path_width == 0.6
    assert rules.quality.weights == {"align": 0.30, "relations": 0.30, "circulation": 0.25, "space": 0.15}
    room = rules.rooms["living_room"]
    assert (room.width, room.depth) == ((3.5, 7.0), (3.0, 6.0))


def test_rules_name_only_catalog_items(rules, catalog):
    names = {s.name for s in catalog.slots}
    for room in rules.rooms.values():
        assert set(room.against_wall) <= names


def _broken_rules_file(tmp_path, change):
    raw = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("change, message", [
    (lambda r: r["quality"]["weights"].update(space=0.5), "summing to 1"),
    (lambda r: r["door"].update(corner_margin=0.4), "half the door width"),
    (lambda r: r["rooms"]["living_room"].update(depth=[1.2, 6.0]), "must fit a door"),
    (lambda r: r["rooms"]["living_room"]["relations"].update(coffee_table_gap=[0.5, 0.35]), "low < high"),
    (lambda r: r["hard_checks"].pop("max_pair_overlap"), "missing or unexpected"),
    (lambda r: r["door"].update(colour="red"), "missing or unexpected"),
    (lambda r: r["quality"].update(relation_falloff=0), "must be positive"),
])
def test_broken_rules_files_fail_loudly(tmp_path, change, message):
    with pytest.raises(RulesError, match=message):
        load_rules(_broken_rules_file(tmp_path, change))


# --------------------------------------------------------------------------- door

@pytest.mark.parametrize("wall, offset, center", [
    ("S", 0.0, (0.65, 0.0)),  # start corner (0, 0), walking towards +x
    ("E", 0.0, (5.0, 0.65)),  # start corner (W, 0), towards +y
    ("N", 0.0, (4.35, 3.0)),  # start corner (W, D), towards -x
    ("W", 0.0, (0.0, 2.35)),  # start corner (0, D), towards -y
    ("S", 1.0, (4.35, 0.0)),
    ("S", 0.5, (2.5, 0.0)),
    ("E", 0.5, (5.0, 1.5)),
])
def test_door_centre_walks_the_walls_counter_clockwise(rules, wall, offset, center):
    np.testing.assert_allclose(door_geometry(5.0, 3.0, wall, offset, rules.door).center, center)


@pytest.mark.parametrize("wall", WALLS)
@pytest.mark.parametrize("offset", [0.0, 0.5, 1.0])
def test_door_stays_on_its_wall_away_from_the_corners(rules, wall, offset):
    width, depth = 5.0, 3.0
    door = door_geometry(width, depth, wall, offset, rules.door)
    on_wall_axis, along_axis = (1, 0) if wall in ("N", "S") else (0, 1)
    assert door.center[on_wall_axis] == {"N": depth, "S": 0.0, "E": width, "W": 0.0}[wall]
    along, length = door.center[along_axis], (width, depth)[along_axis]
    to_corner = min(along, length - along)
    assert to_corner >= 0.65 - 1e-9  # the centre
    assert to_corner - rules.door.width / 2 >= 0.2 - 1e-9  # both edges


@pytest.mark.parametrize("wall", WALLS)
def test_clearance_zone_lies_in_front_of_the_door_inside_the_room(wall):
    # unequal width and depth, so a zone turned the wrong way would show
    rules = DoorRules(width=1.0, clearance_depth=0.6, corner_margin=0.7)
    door = door_geometry(5.0, 3.0, wall, 0.3, rules)
    np.testing.assert_allclose(door.zone_size, (1.0, 0.6) if wall in ("N", "S") else (0.6, 1.0))
    np.testing.assert_allclose(door.zone_center, door.center + door.inward * 0.3)
    lo, hi = box_bounds(door.zone_center, door.zone_size)
    assert np.all(lo >= -1e-9) and np.all(hi <= np.array([5.0, 3.0]) + 1e-9)


@pytest.mark.parametrize("wall, offset, depth", [("X", 0.5, 3.0), ("N", -0.1, 3.0), ("N", 1.1, 3.0),
                                                 ("E", 0.5, 1.2)])
def test_door_rejects_bad_input(rules, wall, offset, depth):
    with pytest.raises(ValueError):
        door_geometry(5.0, depth, wall, offset, rules.door)


# --------------------------------------------------------------------------- hard checks

def test_hand_built_living_room_passes(good_layout, catalog, rules):
    result = check_layout(good_layout, catalog, rules)
    assert result.valid
    assert (result.items_out_of_room, result.overlapping_pairs, result.items_in_door_zone,
            result.unreachable_items) == ((), (), (), ())
    assert result.reachability_ratio == 1.0


@pytest.mark.parametrize("x, passes", [(4.8505, True), (4.9, False)])  # 0.5 mm and 50 mm outside
def test_h1_allows_only_the_tolerance(good_layout, catalog, rules, x, passes):
    result = check_layout(good_layout.with_item(BOOKSHELF, center=(x, 1.5)), catalog, rules)
    assert result.in_room == passes
    assert result.items_out_of_room == (() if passes else (BOOKSHELF,))


@pytest.mark.parametrize("area, passes", [(0.004, True), (0.006, False)])
def test_h2_allows_only_a_small_overlap(good_layout, catalog, rules, area, passes):
    # The side table's y-range lies inside the sofa's, so the overlap is 0.45 m times its x-overlap.
    x = 1.75 - 0.225 + area / 0.45
    result = check_layout(good_layout.with_item(SIDE_TABLE, center=(x, 3.775)), catalog, rules)
    assert result.no_overlap == passes
    assert result.overlapping_pairs == (() if passes else ((SOFA, SIDE_TABLE),))


@pytest.mark.parametrize("x, passes", [(1.3, True), (0.8, False)])  # touching the zone, inside it
def test_h3_keeps_the_door_zone_clear(good_layout, catalog, rules, x, passes):
    result = check_layout(good_layout.with_item(ARMCHAIR, center=(x, 2.0)), catalog, rules)
    assert result.door_clear == passes
    assert result.items_in_door_zone == (() if passes else (ARMCHAIR,))
    assert result.in_room and result.no_overlap


@pytest.mark.parametrize("wall", WALLS)
def test_h3_works_for_every_door_wall(catalog, rules, wall):
    door = door_geometry(5.0, 4.0, wall, 0.5, rules.door)
    x, y = door.zone_center
    layout = make_layout(catalog, 5.0, 4.0, wall, 0.5, {"side_table": (x, y, 0)})
    assert check_layout(layout, catalog, rules).items_in_door_zone == (SIDE_TABLE,)


def test_absent_items_are_ignored(good_layout, catalog, rules):
    blocking = good_layout.with_item(ARMCHAIR, center=(0.8, 2.0))
    assert not check_layout(blocking, catalog, rules).valid
    assert check_layout(blocking.without(ARMCHAIR), catalog, rules).valid
