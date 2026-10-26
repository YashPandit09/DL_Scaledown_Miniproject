"""Reachability tests: who can walk up to which item from the door."""
import numpy as np
import pytest

from tests.layouts import ARMCHAIR, BOOKSHELF, COFFEE_TABLE, SIDE_TABLE, SOFA
from spacegen import geometry
from spacegen.layout import make_layout
from spacegen.rules import WALLS, check_layout, door_geometry, reachability


def _nearest_reached(reach, point):
    px, py = np.meshgrid(reach.xs, reach.ys, indexing="ij")
    return np.hypot(px[reach.reached] - point[0], py[reach.reached] - point[1]).min()


def _start_cell(reach, layout, rules):
    door = door_geometry(layout.width, layout.depth, layout.door_wall, layout.door_offset, rules.door)
    return np.argmin(np.abs(reach.xs - door.zone_center[0])), np.argmin(np.abs(reach.ys - door.zone_center[1]))


@pytest.mark.parametrize("wall", WALLS)
@pytest.mark.parametrize("offset", [0.0, 1.0])  # doors as close to a corner as allowed
def test_empty_room_is_walkable_from_any_door(catalog, rules, wall, offset):
    layout = make_layout(catalog, 5.0, 4.0, wall, offset, {})
    reach = reachability(layout, catalog, rules)
    assert reach.passable[_start_cell(reach, layout, rules)]
    assert reach.reached.sum() == reach.passable.sum() > 0.5 * reach.passable.size
    assert check_layout(layout, catalog, rules).reachability_ratio == 1.0


def test_every_item_of_the_hand_built_room_is_reached(good_layout, catalog, rules):
    assert reachability(good_layout, catalog, rules).unreachable == ()


def test_sofa_is_reached_along_its_front_face(good_layout, catalog, rules):
    """A single access point (in front of the middle of the face) would wrongly fail the classic arrangement."""
    reach = reachability(good_layout, catalog, rules)
    middle_point = geometry.front_point(good_layout.center[SOFA], good_layout.eff_size[SOFA],
                                        int(good_layout.rot[SOFA]), rules.reachability.access_offset)
    # the point in front of the middle of the sofa sits in the gap to the coffee table...
    assert _nearest_reached(reach, middle_point) > rules.reachability.access_tolerance
    # ...but the ends of the sofa's front face are open
    assert SOFA not in reach.unreachable


def test_door_boxed_in_by_furniture(catalog, rules):
    # Door in the middle of the west wall (zone x 0 to 0.9, y 1.55 to 2.45), fenced in by a
    # sofa in front and an armchair and a bookshelf on either side, 0.1 m gaps between them.
    items = {
        "sofa": (1.35, 2.0, 1),  # x 0.9 to 1.8, y 0.95 to 3.05, facing east
        "armchair": (0.4, 2.85, 0),  # x 0 to 0.8, y 2.45 to 3.25
        "bookshelf": (0.4, 1.4, 0),  # x 0 to 0.8, y 1.25 to 1.55, facing north into the pocket
        "coffee_table": (3.5, 2.0, 0),
    }
    layout = make_layout(catalog, 5.0, 4.0, "W", 0.5, items)
    result = check_layout(layout, catalog, rules)
    assert result.in_room and result.no_overlap and result.door_clear
    assert set(result.unreachable_items) == {SOFA, ARMCHAIR, COFFEE_TABLE}
    assert result.reachability_ratio == pytest.approx(1 / 4)  # only the bookshelf
    reach = reachability(layout, catalog, rules)
    px, _ = np.meshgrid(reach.xs, reach.ys, indexing="ij")
    assert px[reach.reached].max() < 0.9  # the reached region never leaves the pocket


def test_item_walled_off_behind_the_sofa(good_layout, catalog, rules):
    # The sofa stands 0.1 m in front of the bookshelf, covering its whole width.
    layout = good_layout.without(ARMCHAIR).with_item(SOFA, center=(4.15, 1.5), rot=3)
    assert reachability(layout, catalog, rules).unreachable == (BOOKSHELF,)


@pytest.mark.parametrize("gap, reached", [(0.55, False), (0.75, True)])
def test_corridors_narrower_than_the_minimum_path_block(catalog, rules, gap, reached):
    # A sofa runs from the north wall down to `gap` above the south wall, splitting the room;
    # the bookshelf on the far side can only be reached through that corridor.
    depth = gap + 2.1
    items = {"sofa": (2.45, gap + 1.05, 1), "bookshelf": (4.85, depth / 2, 3)}
    layout = make_layout(catalog, 5.0, depth, "W", 0.5, items)
    expected = () if reached else (SOFA, BOOKSHELF)
    assert reachability(layout, catalog, rules).unreachable == expected


def test_symmetric_item_against_a_wall_is_reached_from_another_face(catalog, rules):
    # The side table's stored rotation (0) faces the north wall it stands against.
    layout = make_layout(catalog, 4.0, 3.5, "S", 0.5, {"side_table": (2.0, 3.275, 0)})
    assert reachability(layout, catalog, rules).unreachable == ()


@pytest.mark.parametrize("rot, reached", [(0, False), (2, True)])
def test_item_with_a_front_must_face_the_room(catalog, rules, rot, reached):
    # The armchair has one real front: facing the north wall it cannot be used.
    layout = make_layout(catalog, 4.0, 3.5, "S", 0.5, {"armchair": (2.0, 3.1, rot)})
    assert reachability(layout, catalog, rules).unreachable == (() if reached else (ARMCHAIR,))


def test_coffee_table_is_reached_from_its_open_side(good_layout, catalog, rules):
    reach = reachability(good_layout, catalog, rules)
    north_face = geometry.front_point(good_layout.center[COFFEE_TABLE], good_layout.eff_size[COFFEE_TABLE],
                                      0, rules.reachability.access_offset)
    # its north side faces the sofa across a 0.40 m gap; its south side is open
    assert _nearest_reached(reach, north_face) > rules.reachability.access_tolerance
    assert COFFEE_TABLE not in reach.unreachable
    assert SIDE_TABLE not in reach.unreachable


def test_layout_and_catalog_must_match(good_layout, catalog, rules):
    from dataclasses import replace
    with pytest.raises(ValueError):
        reachability(replace(good_layout, room_type="bedroom"), catalog, rules)
