"""Pipeline tests: request checks, variants, top 3, end to end."""
import dataclasses

import numpy as np
import pytest
import torch

from spacegen.dataset import minibatches, training_tensors
from spacegen.generator import generate_set_a, load_generator_config
from spacegen.latent_opt import LatentOptConfig
from spacegen.metrics import layout_distance
from spacegen.models.cvae import CVAE, CVAEConfig, cvae_loss
from spacegen.models.evaluator import Evaluator
from spacegen.pipeline import (PipelineConfig, Request, check_request, choose_variants, generate, load_pipeline_config,
                               select_diverse)
from spacegen.raster import RasterConfig
from spacegen.rules import check_layout
from spacegen.seed import set_seed
from tests.layouts import ARMCHAIR, SOFA

ALL_OPEN = {"sofa": None, "tv_unit": None, "coffee_table": None, "bookshelf": None, "armchair": None,
            "side_table": None}
CROWDED = {**ALL_OPEN, "sofa": "sofa_3seater", "tv_unit": "tv_unit_large"}  # the larger variants, fixed


def test_config_file_holds_the_spec_values():
    assert load_pipeline_config() == PipelineConfig()
    assert (PipelineConfig().candidates, PipelineConfig().top_k, PipelineConfig().tau_div) == (64, 3, 0.3)


def test_request_checks(catalog, rules):
    errors, warnings = check_request(Request(5.0, 4.0, "W", 0.5, {"sofa": None, "tv_unit": None}), catalog, rules)
    assert errors == [] and warnings == []
    errors, _ = check_request(Request(5.0, 4.0, "W", 0.5, {"sofa": "sofa_9seater", "piano": None}), catalog, rules)
    assert errors == ["unknown item 'piano'", "the tv unit is required", "unknown variant 'sofa_9seater' for the sofa"]
    errors, _ = check_request(Request(5.0, 1.2, "W", 0.5, {"sofa": None, "tv_unit": None}), catalog, rules)
    assert any("too short for a door" in error for error in errors)
    _, warnings = check_request(Request(7.5, 6.5, "N", 0.5, {"sofa": None, "tv_unit": None}), catalog, rules)
    assert warnings and "outside the training ranges" in warnings[0]


def test_open_variants_get_the_largest_footprint_that_fits(catalog, rules):
    cond, message = choose_variants(Request(5.0, 4.0, "W", 0.5, ALL_OPEN), catalog, rules)
    assert message is None and cond.items["sofa"] == "sofa_3seater" and cond.items["tv_unit"] == "tv_unit_large"
    # with every optional item, 60,000 INR rules out the 3-seater (60,500 INR with the standard TV unit);
    # of the rest, the 2-seater with the large TV unit has the largest footprint
    cond, _ = choose_variants(dataclasses.replace(Request(5.0, 4.0, "W", 0.5, ALL_OPEN), budget=60_000), catalog, rules)
    prices = sum(catalog.slot(n).variant(v).price for n, v in cond.items.items())
    assert prices == 54_500 and (cond.items["sofa"], cond.items["tv_unit"]) == ("sofa_2seater", "tv_unit_large")
    fixed, _ = choose_variants(Request(5.0, 4.0, "W", 0.5, {"sofa": "sofa_2seater", "tv_unit": None}), catalog, rules)
    assert fixed.items == {"sofa": "sofa_2seater", "tv_unit": "tv_unit_large"}  # requested variants stay


def test_infeasible_requests_are_turned_away_with_a_suggestion(catalog, rules):
    crowded = Request(3.5, 3.0, "W", 0.5, CROWDED)  # 4.24 m^2 of furniture: 44% of the free floor
    cond, message = choose_variants(crowded, catalog, rules)
    assert cond is None and "remove an optional item" in message
    poor = Request(5.0, 4.0, "W", 0.5, {"sofa": None, "tv_unit": None}, budget=20_000)
    cond, message = choose_variants(poor, catalog, rules)
    assert cond is None and "over the budget" in message and "28,000" in message


def test_top_k_picks_are_far_enough_apart(good_layout):
    near = good_layout.with_item(ARMCHAIR, center=good_layout.center[ARMCHAIR] + (0.3, 0.0))  # 0.05 m away on average
    far = good_layout.with_item(SOFA, rot=0).with_item(ARMCHAIR, center=(1.0, 1.0))
    picks, tau = select_diverse([good_layout, near, far], 2, 0.3, 0.25, 2)
    assert picks == [0, 2] and tau == 0.3
    # three picks are asked for but only two are 0.3 m apart: tau shrinks twice, then gives up
    picks, tau = select_diverse([good_layout, near, far], 3, 0.3, 0.25, 2)
    assert picks == [0, 2] and tau == pytest.approx(0.3 * 0.75 ** 2)
    assert layout_distance(good_layout, near) < tau


@pytest.fixture(scope="module")
def models(catalog, rules):
    """A tiny CVAE trained for two seconds on 200 generated layouts (an untrained decoder barely
    depends on z, so latent optimization cannot move its items), and an untrained evaluator."""
    set_seed(0)
    data = training_tensors(generate_set_a(200, np.random.default_rng(0), catalog, rules,
                                           load_generator_config()).layouts, catalog)
    cvae = CVAE(CVAEConfig(hidden=64))
    optimizer = torch.optim.Adam(cvae.parameters(), lr=3e-3)
    shuffle = torch.Generator().manual_seed(0)
    for epoch in range(60):
        for index in minibatches(len(data), 50, shuffle, drop_last=True):
            batch = data[index]
            loss = cvae_loss(cvae(batch.x, batch.c), batch, beta=0.01 * min(1.0, epoch / 20), config=cvae.config)
            optimizer.zero_grad()
            loss.total.backward()
            optimizer.step()
    return cvae.eval(), Evaluator(pixels=32).eval()


def test_the_pipeline_returns_three_valid_layouts_end_to_end(models, catalog, rules):
    """Smoke test: small untrained models still give three valid, distinct layouts."""
    cvae, evaluator = models
    request = Request(6.0, 5.0, "S", 0.3, {"sofa": None, "tv_unit": None})
    small = RasterConfig(canvas=8.0, pixels=32, front_strip=0.1)
    result = generate(request, catalog, rules, cvae, np.random.default_rng(0), evaluator, small,
                      PipelineConfig(), LatentOptConfig(steps=60))
    assert result.message is None and len(result.top) == 3 and result.valid >= 3
    for candidate in result.top:
        assert check_layout(candidate.layout, catalog, rules).valid
        assert 0 <= candidate.quality <= 1 and 0 <= candidate.evaluator_valid <= 1
    first, second, third = (c.layout for c in result.top)
    assert min(layout_distance(first, second), layout_distance(first, third), layout_distance(second, third)) >= result.tau_used
    assert set(result.seconds) == {"check", "sample", "checks", "rank"}


def test_without_an_evaluator_the_rule_score_ranks(models, catalog, rules):
    cvae, _ = models
    request = Request(6.0, 5.0, "S", 0.3, {"sofa": None, "tv_unit": None})
    result = generate(request, catalog, rules, cvae, np.random.default_rng(0), config=PipelineConfig(latent_opt=False))
    qualities = [c.quality for c in result.top]
    assert qualities == sorted(qualities, reverse=True) and all(c.evaluator_score is None for c in result.top)


def test_turned_away_requests_return_a_message(models, catalog, rules):
    cvae, _ = models
    result = generate(Request(3.5, 3.0, "W", 0.5, CROWDED), catalog, rules, cvae, np.random.default_rng(0))
    assert result.condition is None and result.top == [] and "remove an optional item" in result.message
    result = generate(Request(5.0, 4.0, "W", 0.5, {"piano": None}), catalog, rules, cvae, np.random.default_rng(0))
    assert result.condition is None and "unknown item" in result.message
