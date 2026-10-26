"""Metric tests: layout distance, diversity and per-sample scores."""
import numpy as np
import pytest

from spacegen import geometry
from spacegen.layout import canonicalize
from spacegen.metrics import diversity, layout_distance, score_layouts
from spacegen.quality import quality_score
from tests.layouts import ARMCHAIR, COFFEE_TABLE, SIDE_TABLE, SOFA


def test_identical_layouts_are_zero_apart(good_layout):
    assert layout_distance(good_layout, good_layout) == 0.0


def test_moving_one_of_six_items_by_a_metre(good_layout):
    moved = good_layout.with_item(ARMCHAIR, center=good_layout.center[ARMCHAIR] + (0.6, 0.8))
    assert layout_distance(good_layout, moved) == pytest.approx(1.0 / 6)


def test_a_turned_item_costs_half_a_metre_on_its_slot(good_layout, catalog):
    turned = good_layout.with_item(SOFA, rot=0)
    assert layout_distance(good_layout, turned) == pytest.approx(0.5 / 6)
    # turns that look the same are no difference in canonical form; a quarter turn of the coffee table is
    same = canonicalize(good_layout.with_item(COFFEE_TABLE, rot=2).with_item(SIDE_TABLE, rot=3), catalog)
    assert layout_distance(good_layout, same) == 0.0
    quarter = canonicalize(good_layout.with_item(COFFEE_TABLE, rot=1), catalog)
    assert layout_distance(good_layout, quarter) == pytest.approx(0.5 / 6)


def test_only_slots_in_both_layouts_count(good_layout):
    other = good_layout.without(SIDE_TABLE).with_item(SOFA, center=good_layout.center[SOFA] + (1.0, 0.0))
    assert layout_distance(good_layout, other) == pytest.approx(1.0 / 5)


def test_diversity_is_the_mean_over_pairs(good_layout):
    a = good_layout
    b = a.with_item(ARMCHAIR, center=a.center[ARMCHAIR] + (0.6, 0.8))
    c = a.with_item(SOFA, rot=0)
    assert diversity([a]) is None and diversity([]) is None
    assert diversity([a, b]) == pytest.approx(layout_distance(a, b))
    expected = np.mean([layout_distance(a, b), layout_distance(a, c), layout_distance(b, c)])
    assert diversity([a, b, c]) == pytest.approx(expected)


def test_scores_per_sample(good_layout, catalog, rules):
    overlapping = good_layout.with_item(SIDE_TABLE, center=good_layout.center[SIDE_TABLE] + (0.2, 0.0))
    scores = score_layouts([good_layout, overlapping], catalog, rules)
    assert scores["valid"].tolist() == [True, False]
    assert scores["overlap"][0] == 0.0
    pair = geometry.overlap_area(overlapping.center[SIDE_TABLE], overlapping.eff_size[SIDE_TABLE],
                                 overlapping.center[SOFA], overlapping.eff_size[SOFA])
    assert scores["overlap"][1] == pytest.approx(pair) and pair > 0
    assert scores["reachability"][0] == 1.0
    assert scores["quality"][0] == pytest.approx(quality_score(good_layout, catalog, rules).total)
    assert len(score_layouts([], catalog, rules)) == 0
