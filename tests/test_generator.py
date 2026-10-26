"""Generator tests: Set A layouts are valid, canonical and follow the rules."""
import dataclasses

import numpy as np
import pytest

from spacegen import geometry
from spacegen.generator import (Condition, WallFrame, fitting_styles, generate_layout, generate_set_a,
                                load_generator_config, optional_probability, rejection_summary, sample_furniture)
from spacegen.layout import is_canonical
from spacegen.rules import WALLS, check_layout
from tests.layouts import ARMCHAIR, SOFA, TV_UNIT

CONFIG = load_generator_config()


@pytest.fixture(scope="module")
def generated(catalog, rules):
    return generate_set_a(150, np.random.default_rng(0), catalog, rules, CONFIG)


def _condition(catalog, width, depth, *optional):
    """Sofa and TV unit (default variants), plus the named optional items."""
    names = ("sofa", "tv_unit") + optional
    return Condition(width, depth, "S", 0.5, {n: catalog.slot(n).variant().id for n in names})


def _gap_behind(layout, k):
    """Distance from item k's back to the wall behind it."""
    lo, hi = geometry.box_bounds(layout.center[k], layout.eff_size[k])
    behind = WALLS[(int(layout.rot[k]) + 2) % 4]  # an item facing North has its back to the S wall
    return {"W": lo[0], "S": lo[1], "E": layout.width - hi[0], "N": layout.depth - hi[1]}[behind]


# --------------------------------------------------------------------------- generated layouts

def test_every_layout_passes_the_hard_checks_and_is_canonical(generated, catalog, rules):
    assert len(generated.layouts) == len(generated.info) == 150
    for i in range(len(generated.layouts)):
        layout = generated.layouts.layout(i, catalog)
        assert check_layout(layout, catalog, rules).valid
        assert is_canonical(layout, catalog)


def test_every_layout_meets_the_alignment_and_relation_rules(generated):
    assert (generated.info["align"] == 1).all() and (generated.info["relations"] == 1).all()
    assert generated.info["quality"].min() > 0.8


def test_each_style_has_its_arrangement(generated, catalog):
    assert set(generated.info["style"]) == {"a", "b", "c"}
    for i, style in enumerate(generated.info["style"]):
        layout = generated.layouts.layout(i, catalog)
        assert layout.rot[TV_UNIT] == (layout.rot[SOFA] + 2) % 4  # sofa and TV unit face each other
        assert _gap_behind(layout, TV_UNIT) <= CONFIG.wall_gap + 1e-9
        if style == "b":  # floating, with the walkway behind
            assert _gap_behind(layout, SOFA) >= CONFIG.walkway - 1e-9
        else:
            assert _gap_behind(layout, SOFA) <= CONFIG.wall_gap + 1e-9
        if style == "c":  # the armchair's back against the side wall
            assert _gap_behind(layout, ARMCHAIR) <= CONFIG.wall_gap + 1e-9
            assert (layout.rot[ARMCHAIR] - layout.rot[SOFA]) % 2 == 1


def test_the_same_seed_gives_the_same_layouts(catalog, rules):
    first, second = (generate_set_a(20, np.random.default_rng(7), catalog, rules, CONFIG) for _ in range(2))
    np.testing.assert_array_equal(first.layouts.center, second.layouts.center)
    np.testing.assert_array_equal(first.layouts.rot, second.layouts.rot)
    assert first.attempts.equals(second.attempts)


# --------------------------------------------------------------------------- the attempt log

def test_attempt_log(generated):
    log = generated.attempts
    assert log.groupby("room").size().max() <= CONFIG.attempts
    assert set(log["outcome"]) <= {"valid", "invalid", "no position", "no style"}
    kept = log[log["outcome"] == "valid"]
    assert len(kept) == len(generated.layouts)
    last = log.groupby("room")["attempt"].max()
    assert (kept["attempt"].to_numpy() == last[kept["room"]].to_numpy()).all()  # a room stops at its valid layout
    assert (log.loc[log["outcome"] == "invalid", "failed"] != "").all()  # failures name their checks


def test_rejection_summary_adds_up(generated):
    for by in ("area", "items"):
        table = rejection_summary(generated.attempts, by)
        assert table["attempts"].sum() == len(generated.attempts)
        assert table["rooms"].sum() == generated.attempts["room"].nunique()
        assert table["rejected"].between(0, 1).all()


# --------------------------------------------------------------------------- styles and rooms

def test_styles_that_fit_a_room(catalog, rules):
    # 5 x 4 m: along y (sofa on the S or N wall) the sofa-TV distance is 4.0 - 0.9 - 0.4 = 2.7 m,
    # so a wall sofa fits; along x it would be 3.7 m, beyond 3.5, so only a floating sofa fits
    plain = fitting_styles(_condition(catalog, 5.0, 4.0), catalog, rules, CONFIG)
    assert sorted(plain) == [("a", "N"), ("a", "S"), ("b", "E"), ("b", "N"), ("b", "S"), ("b", "W")]
    group = fitting_styles(_condition(catalog, 5.0, 4.0, "armchair", "coffee_table"), catalog, rules, CONFIG)
    assert sorted(group) == sorted(plain + [("c", "N"), ("c", "S")])
    # 7 x 6 m: too deep for a wall sofa either way (6.0 - 1.3 = 4.7 m)
    big = fitting_styles(_condition(catalog, 7.0, 6.0, "armchair", "coffee_table"), catalog, rules, CONFIG)
    assert {style for style, _ in big} == {"b"}


def test_a_room_with_no_fitting_style_is_dropped_at_once(catalog, rules):
    tiny = _condition(catalog, 2.5, 2.5)  # 2.5 - 1.3 = 1.2 m between sofa and TV at most
    assert fitting_styles(tiny, catalog, rules, CONFIG) == []
    result = generate_layout(tiny, np.random.default_rng(0), catalog, rules, CONFIG)
    assert result.layout is None and [a.outcome for a in result.attempts] == ["no style"]


def test_optional_items_become_more_likely_in_larger_rooms(catalog):
    assert optional_probability(10.5, CONFIG) == pytest.approx(0.25)
    assert optional_probability(20.25, CONFIG) == pytest.approx(0.5)  # halfway between 10.5 and 30 m^2
    assert optional_probability(42.0, CONFIG) == pytest.approx(0.75)
    rng = np.random.default_rng(0)
    for area, mean in ((10.5, 1.0), (42.0, 3.0)):
        draws = [sample_furniture(rng, area, catalog, CONFIG) for _ in range(4000)]
        assert all({"sofa", "tv_unit"} <= set(d) for d in draws)
        assert np.mean([len(d) - 2 for d in draws]) == pytest.approx(mean, abs=0.1)
        assert len({frozenset(d) for d in draws}) == 16  # every combination of the 4 optional items occurs
        assert {d["sofa"] for d in draws} == {"sofa_3seater", "sofa_2seater"}


@pytest.mark.parametrize("wall", WALLS)
def test_wall_frame(wall):
    frame = WallFrame(wall, 5.0, 4.0)
    x, y = frame.to_room(1.0, 0.25)  # 1 m along the wall, 0.25 m into the room
    assert {"S": y, "N": 4.0 - y, "W": x, "E": 5.0 - x}[wall] == pytest.approx(0.25)
    t0, t1, n0, n1 = frame.to_wall(np.array([x - 0.1, y - 0.2]), np.array([x + 0.1, y + 0.2]))
    assert ((t0 + t1) / 2, (n0 + n1) / 2) == pytest.approx((1.0, 0.25))
    along = np.subtract(frame.to_room(1.1, 0.25), (x, y)) / 0.1
    away = np.subtract(frame.to_room(1.0, 0.35), (x, y)) / 0.1
    np.testing.assert_allclose(geometry.facing_vector(frame.rotation("+t"), along), along, atol=1e-9)
    np.testing.assert_allclose(geometry.facing_vector(frame.rotation("+n"), away), away, atol=1e-9)
    assert frame.rotation("-n") == (frame.rotation("+n") + 2) % 4


def test_only_the_living_room_has_styles(catalog, rules):
    bedroom = dataclasses.replace(catalog, room_type="bedroom")
    with pytest.raises(NotImplementedError, match="bedroom"):
        fitting_styles(_condition(catalog, 4.0, 4.0), bedroom, rules, CONFIG)
