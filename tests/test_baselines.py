"""Baseline tests: B1, B2 and G0 return layouts for any condition."""
import dataclasses

import numpy as np
import pytest

from spacegen import geometry
from spacegen.baselines import (GeneratorBaseline, StatisticalBaseline, UniformBaseline, load_baseline_config)
from spacegen.dataset import stack_layouts
from spacegen.generator import Condition, generate_set_a, load_generator_config, sample_condition
from spacegen.layout import is_canonical
from spacegen.rules import check_layout
from tests.layouts import COFFEE_TABLE, SOFA

CONFIG = load_baseline_config()
GENERATOR = load_generator_config()


@pytest.fixture(scope="module")
def train(catalog, rules):
    return generate_set_a(300, np.random.default_rng(0), catalog, rules, GENERATOR).layouts


@pytest.fixture(scope="module")
def conditions(catalog, rules):
    rng = np.random.default_rng(1)
    return [sample_condition(rng, catalog, rules, GENERATOR) for _ in range(30)]


def _matches(layout, cond, catalog):
    return Condition.of(layout, catalog) == cond


@pytest.mark.parametrize("make", ["B1", "B2", "G0"])
def test_every_baseline_returns_layouts_for_any_condition(make, catalog, rules, train, conditions):
    baseline = {"B1": lambda: UniformBaseline(catalog),
                "B2": lambda: StatisticalBaseline(catalog, CONFIG).fit(train),
                "G0": lambda: GeneratorBaseline(catalog, rules, GENERATOR)}[make]()
    out_of_range = Condition(7.6, 6.4, "N", 0.3, {"sofa": "sofa_3seater", "tv_unit": "tv_unit_large"})
    rng = np.random.default_rng(2)
    for cond in conditions[:10] + [out_of_range]:
        samples = baseline.sample(cond, 3, rng)
        assert samples.attempts >= 3 and len(samples.layouts) <= 3
        assert all(_matches(layout, cond, catalog) and is_canonical(layout, catalog) for layout in samples.layouts)
    if make != "G0":
        assert len(samples.layouts) == 3  # raw samples, unfiltered
    else:
        assert all(check_layout(layout, catalog, rules).valid for layout in samples.layouts)


def test_b1_keeps_items_inside_the_room_facing_every_way(catalog, rules, conditions):
    rng = np.random.default_rng(0)
    samples = UniformBaseline(catalog).sample(conditions[0], 200, rng)
    assert all(check_layout(layout, catalog, rules).in_room for layout in samples.layouts[:40])
    assert {int(layout.rot[SOFA]) for layout in samples.layouts} == {0, 1, 2, 3}


def test_b2_reproduces_where_training_rooms_put_each_item(catalog, good_layout):
    rng = np.random.default_rng(0)
    copies = [good_layout.with_item(SOFA, center=good_layout.center[SOFA] + rng.normal(0, 0.02, 2)) for _ in range(60)]
    # without out-of-room redraws, which would cut off the draws pushing the wall sofa outwards
    plain = dataclasses.replace(CONFIG, b2_redraw_outside=False)
    b2 = StatisticalBaseline(catalog, plain).fit(stack_layouts(copies, catalog))
    samples = b2.sample(Condition.of(good_layout, catalog), 100, rng).layouts
    sofa = np.array([layout.center[SOFA] for layout in samples])
    np.testing.assert_allclose(sofa.mean(axis=0), good_layout.center[SOFA], atol=0.03)
    assert sofa.std(axis=0) == pytest.approx(np.hypot(0.02, CONFIG.b2_noise * good_layout.room), rel=0.3)
    assert {int(layout.rot[SOFA]) for layout in samples} == {2}  # always facing South, as in training
    assert {int(layout.rot[COFFEE_TABLE]) for layout in samples} == {0}  # canonical


def test_b2_redraws_reduce_overlaps(catalog, rules, train, conditions):
    def overlapping(config):
        b2 = StatisticalBaseline(catalog, config).fit(train)
        rng = np.random.default_rng(3)
        layouts = [layout for cond in conditions for layout in b2.sample(cond, 5, rng).layouts]
        return np.mean([geometry.total_overlap_area(lay.center, lay.eff_size, lay.mask) > 0 for lay in layouts])

    assert overlapping(CONFIG) < overlapping(dataclasses.replace(CONFIG, b2_redraws=0))


def test_b2_borrows_from_wider_groups_when_a_bin_is_empty(catalog, good_layout):
    b2 = StatisticalBaseline(catalog, CONFIG).fit(stack_layouts([good_layout] * 30, catalog))  # door on W only
    other = dataclasses.replace(Condition.of(good_layout, catalog), door_wall="E")
    assert len(b2.sample(other, 5, np.random.default_rng(0)).layouts) == 5
    with pytest.raises(RuntimeError, match="fit"):
        StatisticalBaseline(catalog, CONFIG).sample(other, 1, np.random.default_rng(0))


def test_g0_counts_every_attempt(catalog, rules, conditions):
    samples = GeneratorBaseline(catalog, rules, GENERATOR).sample(conditions[0], 5, np.random.default_rng(0))
    assert len(samples.layouts) == 5 and samples.attempts >= 5


def test_samples_are_reproducible(catalog, train, conditions):
    b2 = StatisticalBaseline(catalog, CONFIG).fit(train)
    first, second = (b2.sample(conditions[3], 4, np.random.default_rng(9)).layouts for _ in range(2))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.center, b.center)


def test_b2_can_also_redraw_items_that_stick_out_of_the_room(catalog, rules, train, conditions):
    def outside_share(redraw_outside):
        b2 = StatisticalBaseline(catalog, dataclasses.replace(CONFIG, b2_redraw_outside=redraw_outside)).fit(train)
        rng = np.random.default_rng(4)
        layouts = [layout for cond in conditions for layout in b2.sample(cond, 4, rng).layouts]
        return np.mean([not check_layout(layout, catalog, rules).in_room for layout in layouts])

    assert outside_share(True) < 0.05 < outside_share(False)
