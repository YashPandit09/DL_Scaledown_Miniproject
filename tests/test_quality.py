"""Quality score tests: each term and each relation behaves as designed."""
import dataclasses

import pytest

from spacegen.layout import make_layout
from spacegen.quality import _RELATIONS, furniture_share, quality_score, range_score
from spacegen.rules import reachability
from tests.layouts import ARMCHAIR, COFFEE_TABLE, SIDE_TABLE, TV_UNIT


def test_hand_built_room_scores_full_marks_except_circulation(good_layout, catalog, rules):
    q = quality_score(good_layout, catalog, rules)
    assert (q.align, q.relations, q.space) == (1.0, 1.0, 1.0)
    assert set(q.relation_scores.values()) == {1.0} and len(q.relation_scores) == 4
    assert 0.0 < q.circulation < 1.0
    weights = rules.quality.weights
    assert q.total == pytest.approx(weights["align"] + weights["relations"]
                                    + weights["circulation"] * q.circulation + weights["space"])


@pytest.mark.parametrize("value, expected", [
    (0.40, 1.0), (0.35, 1.0), (0.50, 1.0),  # inside the range, ends included
    (0.60, 0.8), (0.25, 0.8),  # 0.1 m outside, with falloff 0.5
    (1.00, 0.0), (-1.0, 0.0),  # beyond the falloff
])
def test_range_score(value, expected):
    assert range_score(value, 0.35, 0.50, 0.5) == pytest.approx(expected)


def test_every_configured_relation_has_a_scoring_rule(rules):
    for room in rules.rooms.values():
        assert set(room.relations) <= set(_RELATIONS)


# --------------------------------------------------------------------------- relations

def _relation(layout, catalog, rules, name):
    return quality_score(layout, catalog, rules).relation_scores.get(name)


@pytest.mark.parametrize("change, expected", [
    (dict(center=(2.8, 1.65)), 0.5),  # 1.25 m front to front: 0.25 m short of 1.5
    (dict(rot=2), 0.0),  # TV turned to face the same way as the sofa
    (dict(center=(4.5, 0.2)), 0.0),  # TV beyond the end of the sofa, not in front of it
])
def test_sofa_and_tv(good_layout, catalog, rules, change, expected):
    layout = good_layout.with_item(TV_UNIT, **change)
    assert _relation(layout, catalog, rules, "sofa_tv_distance") == pytest.approx(expected)


@pytest.mark.parametrize("y, expected", [(2.425, 1.0), (2.225, 0.8), (1.5, 0.0)])  # gaps 0.40, 0.60, 1.325
def test_coffee_table_gap(good_layout, catalog, rules, y, expected):
    layout = good_layout.with_item(COFFEE_TABLE, center=(2.8, y))
    assert _relation(layout, catalog, rules, "coffee_table_gap") == pytest.approx(expected)


@pytest.mark.parametrize("center, expected", [
    ((1.475, 3.775), 1.0),  # 0.05 m from the sofa's end
    ((1.125, 3.775), 0.5),  # 0.40 m away: 0.25 m beyond the 0.15 m limit
    ((1.475, 2.6), 0.0),  # in front of the sofa, not beside it
])
def test_side_table_beside_the_sofa(good_layout, catalog, rules, center, expected):
    layout = good_layout.with_item(SIDE_TABLE, center=center)
    assert _relation(layout, catalog, rules, "side_table_max_gap") == pytest.approx(expected)


@pytest.mark.parametrize("rot, expected", [(3, 1.0), (1, 0.0), (2, 0.0)])  # faces the table / away / parallel
def test_armchair_turned_towards_the_coffee_table(good_layout, catalog, rules, rot, expected):
    layout = good_layout.with_item(ARMCHAIR, rot=rot)
    assert _relation(layout, catalog, rules, "armchair_max_gap") == pytest.approx(expected)


def test_armchair_gap_to_the_coffee_table(catalog, rules):
    # a 7 x 6 m room: the armchair faces the coffee table 1.45 m away, 0.25 m beyond 1.2
    items = {"sofa": (3.5, 5.55, 2), "tv_unit": (3.5, 0.2, 0), "coffee_table": (3.5, 4.425, 0),
             "armchair": (5.85, 4.6, 3)}
    layout = make_layout(catalog, 7.0, 6.0, "W", 0.5, items)
    assert _relation(layout, catalog, rules, "armchair_max_gap") == pytest.approx(0.5)


def test_relations_with_an_absent_item_are_skipped(good_layout, catalog, rules):
    without_side_table = quality_score(good_layout.without(SIDE_TABLE), catalog, rules)
    assert "side_table_max_gap" not in without_side_table.relation_scores
    without_table = quality_score(good_layout.without(COFFEE_TABLE), catalog, rules)
    assert set(without_table.relation_scores) == {"sofa_tv_distance", "side_table_max_gap"}
    assert without_table.relations == 1.0


# --------------------------------------------------------------------------- alignment, space, circulation

@pytest.mark.parametrize("change, expected", [
    (dict(center=(2.8, 0.5)), 0.5),  # TV 0.30 m off the wall (limit 0.10)
    (dict(rot=2), 0.5),  # TV against the south wall but facing it
])
def test_alignment(good_layout, catalog, rules, change, expected):
    assert quality_score(good_layout.with_item(TV_UNIT, **change), catalog, rules).align == pytest.approx(expected)


def test_symmetric_items_align_by_position_only(good_layout, catalog, rules):
    room = dataclasses.replace(rules.rooms["living_room"], against_wall=("side_table",))
    rules = dataclasses.replace(rules, rooms={"living_room": room})
    for rot in range(4):  # the side table touches the north wall whichever way it is stored
        assert quality_score(good_layout.with_item(SIDE_TABLE, rot=rot), catalog, rules).align == 1.0
    assert quality_score(good_layout.with_item(SIDE_TABLE, center=(2.0, 2.0)), catalog, rules).align == 0.0


@pytest.mark.parametrize("width, depth, expected_share", [
    (5.0, 4.0, 4.1225 / 20.0),  # inside the 15-40% band
    (7.0, 6.0, 4.1225 / 42.0),  # below it
    (3.5, 3.0, 4.1225 / 10.5),  # just inside
])
def test_space_score_follows_the_furniture_share(catalog, rules, width, depth, expected_share):
    from tests.layouts import GOOD_ITEMS
    layout = make_layout(catalog, width, depth, "W", 0.5, GOOD_ITEMS)  # positions do not matter here
    share = furniture_share(layout)
    assert share == pytest.approx(expected_share)
    low, high = rules.quality.space_band
    expected = 1.0 if low <= share <= high else range_score(share, low, high, rules.quality.space_falloff)
    assert quality_score(layout, catalog, rules).space == pytest.approx(expected)
    if share < low:
        assert expected == pytest.approx(share / low)  # falls linearly to 0 for an empty room


def test_circulation_is_the_walkable_share_of_free_floor(good_layout, catalog, rules):
    reach = reachability(good_layout, catalog, rules)
    q = quality_score(good_layout, catalog, rules, reach=reach)
    assert q.circulation == pytest.approx(reach.passable.sum() / reach.free.sum())
    # removing items frees floor and walkways
    assert quality_score(good_layout.without(ARMCHAIR), catalog, rules).circulation > q.circulation
