"""Set B tests: each perturbation does what it says, and labels come from the checker."""
import dataclasses

import numpy as np
import pytest

from spacegen import geometry
from spacegen.generator import load_generator_config
from spacegen.perturb import (force_overlap, generate_set_b, jitter, label_summary, load_set_b_config,
                              near_miss_door, near_miss_overlap, random_placement, turn)
from spacegen.rules import check_layout, door_geometry
from tests.layouts import COFFEE_TABLE, SIDE_TABLE, SOFA

CONFIG = load_generator_config()
SET_B = load_set_b_config()


def _moved(before, after):
    return np.flatnonzero(np.abs(after.center - before.center).max(axis=1) > 0)


def _overlaps(layout):
    """{(i, j): area} for every overlapping pair."""
    area = np.triu(geometry.pairwise_overlap_area(layout.center, layout.eff_size, layout.mask), k=1)
    return {(int(i), int(j)): area[i, j] for i, j in zip(*np.nonzero(area))}


def test_shares_cover_every_type_and_sum_to_one():
    shares = SET_B.shares()
    assert sum(shares.values()) == pytest.approx(1.0) and shares["clean"] == 0.5
    assert shares["rotation"] == shares["random"] == shares["overlap"] == pytest.approx(0.1)
    assert shares["jitter_0.1"] == shares["jitter_0.3"] == shares["jitter_0.6"] == pytest.approx(0.1 / 3)
    assert shares["near_miss_overlap"] == shares["near_miss_door"] == pytest.approx(0.05)


@pytest.mark.parametrize("sigma", [0.1, 0.6])
def test_jitter_moves_some_items_by_about_sigma(good_layout, sigma):
    rng = np.random.default_rng(0)
    steps, moved_counts = [], []
    for _ in range(300):
        after = jitter(good_layout, sigma, 0.5, rng)
        moved = _moved(good_layout, after)
        moved_counts.append(len(moved))
        steps.append(after.center[moved] - good_layout.center[moved])
    assert min(moved_counts) >= 1 and np.mean(moved_counts) == pytest.approx(3.0, abs=0.3)  # half of 6 items
    assert np.concatenate(steps).std() == pytest.approx(sigma, rel=0.1)


def test_turn_changes_the_look_of_one_or_two_items(good_layout, catalog):
    rng = np.random.default_rng(0)
    for _ in range(50):
        after = turn(good_layout, rng, catalog)
        changed = np.flatnonzero(after.rot != good_layout.rot)
        assert 1 <= len(changed) <= 2 and SIDE_TABLE not in changed
        np.testing.assert_array_equal(after.center, good_layout.center)
        if COFFEE_TABLE in changed:  # a half turn would look the same
            assert after.rot[COFFEE_TABLE] % 2 != good_layout.rot[COFFEE_TABLE] % 2


def test_random_placement_keeps_every_item_in_the_room(good_layout, catalog, rules):
    rng = np.random.default_rng(0)
    for _ in range(50):
        after = random_placement(good_layout, rng)
        assert check_layout(after, catalog, rules).in_room
        assert len(_moved(good_layout, after)) == 6


def test_forced_overlap_puts_one_item_inside_another(good_layout, catalog, rules):
    rng = np.random.default_rng(0)
    for _ in range(50):
        after = force_overlap(good_layout, rng)
        assert len(_moved(good_layout, after)) == 1
        assert not check_layout(after, catalog, rules).no_overlap


def test_overlap_near_miss_creates_one_small_overlap(good_layout, catalog, rules):
    rng = np.random.default_rng(0)
    low, high = SET_B.near_miss_area
    under = []
    for _ in range(40):
        after = near_miss_overlap(good_layout, rng, catalog, rules, SET_B)
        assert len(_moved(good_layout, after)) == 1
        (pair, area), = _overlaps(after).items()  # exactly one overlapping pair
        assert low - 1e-12 <= area <= high + 1e-12
        result = check_layout(after, catalog, rules)
        assert result.in_room and result.door_clear and result.reachable
        assert result.valid == (area <= rules.hard_checks.max_pair_overlap)  # the overlap alone decides
        under.append(area <= rules.hard_checks.max_pair_overlap)
    assert 0.3 <= np.mean(under) <= 0.7  # log-uniform: half below the tolerance


def test_door_near_miss_ends_within_the_gap_of_the_door_zone(good_layout, catalog, rules):
    rng = np.random.default_rng(0)
    door = door_geometry(5.0, 4.0, "W", 0.5, rules.door)
    for _ in range(40):
        after = near_miss_door(good_layout, rng, catalog, rules, SET_B)
        (k,) = _moved(good_layout, after)
        lo, hi = geometry.box_bounds(after.center[k], after.eff_size[k])
        zone_lo, zone_hi = geometry.box_bounds(door.zone_center, door.zone_size)
        separation = np.maximum(zone_lo - hi, lo - zone_hi).max()  # > 0 clear of the zone, < 0 inside it
        assert abs(separation) <= SET_B.near_miss_gap + 1e-9
        assert _overlaps(after) == {}  # the door zone is the only thing it gets near
        in_zone = geometry.overlap_area(after.center[k], after.eff_size[k], door.zone_center, door.zone_size)
        result = check_layout(after, catalog, rules)
        assert result.reachable and result.door_clear == (in_zone <= rules.hard_checks.door_overlap_tol)


def test_set_b_has_exact_shares_and_checker_labels(catalog, rules):
    result = generate_set_b(200, np.random.default_rng(0), catalog, rules, CONFIG, SET_B)
    counts = result.info["perturbation"].value_counts().to_dict()
    assert counts == {"clean": 100, "rotation": 20, "random": 20, "overlap": 20, "near_miss_overlap": 10,
                      "near_miss_door": 10, "jitter_0.1": 7, "jitter_0.3": 7, "jitter_0.6": 6}
    for i in range(0, 200, 7):  # labels match the checker on the stored (canonical) layouts
        assert result.info["valid"][i] == check_layout(result.layouts.layout(i, catalog), catalog, rules).valid
    assert result.info.loc[result.info["perturbation"] == "clean", "valid"].all()
    assert not result.info.loc[result.info["perturbation"] == "overlap", "valid"].any()
    assert 0.5 <= result.info["valid"].mean() <= 0.75
    assert result.info["quality"].between(0, 1).all()
    summary = label_summary(result.info)
    assert summary["samples"].sum() == 200 and summary.loc["clean", "valid"] == 1.0


def test_set_b_is_reproducible(catalog, rules):
    no_clean = dataclasses.replace(SET_B, clean_share=0.0)
    first, second = (generate_set_b(30, np.random.default_rng(3), catalog, rules, CONFIG, no_clean) for _ in range(2))
    np.testing.assert_array_equal(first.layouts.center, second.layouts.center)
    assert first.info.equals(second.info)
    assert (first.layouts.center[:, SOFA] != 0).any()
