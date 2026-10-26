"""Latent optimization tests: the loss, its gradients, and the optimizer."""
import dataclasses

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from spacegen.generator import Condition, generate_set_a, load_generator_config
from spacegen.latent_opt import (LatentOptConfig, Problem, constraint_terms, initial_rotations, load_latent_opt_config,
                                 optimize, snap_pins, total_loss)
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.pipeline import condition_batch, problem_for
from spacegen.seed import set_seed
from tests.layouts import SIDE_TABLE, SOFA

CONFIG = LatentOptConfig()


def test_config_file_holds_the_spec_values():
    config = load_latent_opt_config()
    assert config == LatentOptConfig()
    assert (config.steps, config.lr, config.margin) == (150, 0.05, 0.05)
    assert (config.lambda_overlap, config.lambda_room, config.lambda_door, config.lambda_pin,
            config.lambda_anchor) == (10.0, 10.0, 10.0, 20.0, 0.05)


def _problem(layout, catalog, rules, n=1, dtype=torch.float32):
    cond = Condition.of(layout, catalog)
    problem = problem_for(cond, condition_batch(cond, n, catalog), catalog, rules, "cpu")
    return Problem(*(t.to(dtype) for t in (problem.c, problem.zone_center, problem.zone_size)))


def _normalized(layout, dtype=torch.float32):
    return torch.as_tensor(layout.center / layout.room, dtype=dtype)[None]


def test_a_valid_layout_has_no_constraint_loss(good_layout, catalog, rules):
    problem = _problem(good_layout, catalog, rules)
    terms = constraint_terms(_normalized(good_layout), torch.as_tensor(good_layout.rot)[None], problem, 6, margin=0.0)
    assert all(value.item() == pytest.approx(0.0, abs=1e-6) for value in terms.values())
    # with a 10 cm margin, items closer than that (the side table 5 cm from the sofa) are pushed apart
    with_margin = constraint_terms(_normalized(good_layout), torch.as_tensor(good_layout.rot)[None], problem, 6, 0.10)
    assert with_margin["overlap"].item() > 0


def test_each_term_reacts_to_its_violation(good_layout, catalog, rules):
    problem = _problem(good_layout, catalog, rules)
    rot = torch.as_tensor(good_layout.rot)[None]

    def terms_with(slot, center):
        return constraint_terms(_normalized(good_layout.with_item(slot, center=center)), rot, problem, 6, 0.0)

    assert terms_with(SIDE_TABLE, (4.9, 3.775))["room"].item() > 0  # half outside the east wall... and overlapping
    assert terms_with(SIDE_TABLE, (0.4, 2.0))["door"].item() > 0  # in the door's clearance zone
    assert terms_with(SIDE_TABLE, (2.8, 3.55))["overlap"].item() > 0  # on the sofa


def test_gradients_match_finite_differences_including_a_contained_item(good_layout, catalog, rules):
    """Finite differences against autograd, with a box wholly inside another."""
    contained = good_layout.with_item(SIDE_TABLE, center=(2.6, 3.5))  # inside the sofa's footprint
    # move every item off the kinks: the hand-built room has items exactly against walls, where the
    # out-of-room term changes slope (finite differences there average the two one-sided slopes)
    contained = dataclasses.replace(contained, center=contained.center - np.array([0.013, 0.017]))
    problem = _problem(contained, catalog, rules, dtype=torch.float64)
    rot = torch.as_tensor(contained.rot)[None]
    z0 = torch.zeros(1, 2, dtype=torch.float64)
    config = dataclasses.replace(CONFIG, margin=0.07)  # no gap equals the margin either

    def loss(positions):
        return total_loss(constraint_terms(positions, rot, problem, 6, config.margin), z0, z0, config)[0]

    positions = _normalized(contained, torch.float64).requires_grad_(True)
    assert gradcheck(loss, (positions,))
    (gradient,) = torch.autograd.grad(loss(positions).sum(), positions)
    assert gradient[0, SIDE_TABLE].abs().sum() > 0  # the contained side table is pushed out


@pytest.fixture(scope="module")
def setting(catalog, rules):
    """An untrained CVAE and 64 candidates for one generated room: plenty of overlaps to repair."""
    layout = generate_set_a(1, np.random.default_rng(3), catalog, rules, load_generator_config()).layouts.layout(0, catalog)
    set_seed(0)
    model = CVAE(CVAEConfig(hidden=64)).eval()
    problem = _problem(layout, catalog, rules, n=64)
    z0 = torch.randn(64, model.config.latent, generator=torch.Generator().manual_seed(1))
    return model, problem, z0


def test_optimization_lowers_the_loss_for_most_candidates(setting):
    model, problem, z0 = setting
    result = optimize(model, z0, problem, CONFIG)
    # The final loss below the initial one for at least 90% of the candidates that
    # needed repair; candidates already within tolerance keep z0, so theirs cannot fall
    needed = result.initial >= CONFIG.tolerance
    assert needed.any() and (result.final[needed] < result.initial[needed]).float().mean() >= 0.9
    assert (result.final <= result.initial + 1e-6).float().mean() >= 0.9
    assert (result.steps <= CONFIG.steps).all() and result.positions.shape == (64, 6, 2)


def test_rotations_stay_fixed_and_the_decoder_stays_frozen(setting):
    model, problem, z0 = setting
    before = [p.detach().clone() for p in model.parameters()]
    result = optimize(model, z0, problem, dataclasses.replace(CONFIG, steps=20))
    assert torch.equal(result.rot, initial_rotations(model, z0, problem))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_candidates_that_meet_the_constraints_stop(setting):
    model, problem, z0 = setting
    loose = dataclasses.replace(CONFIG, tolerance=float("inf"))  # everyone is done at once
    result = optimize(model, z0, problem, loose)
    assert (result.steps == 0).all() and torch.equal(result.z, z0)


def test_the_anchor_keeps_z_near_its_start(setting):
    model, problem, z0 = setting
    free = optimize(model, z0, problem, dataclasses.replace(CONFIG, lambda_anchor=0.0, steps=60))
    held = optimize(model, z0, problem, dataclasses.replace(CONFIG, lambda_anchor=50.0, steps=60))
    assert (held.z - z0).norm(dim=1).mean() < 0.5 * (free.z - z0).norm(dim=1).mean()


def test_pins_set_the_facing_pull_the_item_and_are_snapped(setting):
    model, problem, z0 = setting
    k = 6
    pin_mask = torch.zeros(64, k, dtype=torch.bool)
    pin_mask[:, SOFA] = True
    pin_center = torch.zeros(64, k, 2)
    pin_center[:, SOFA] = torch.tensor([2.0, 1.5])
    pin_rot = torch.full((64, k), -1)
    pin_rot[:, SOFA] = 1  # face East
    pinned = dataclasses.replace(problem, pin_mask=pin_mask, pin_center=pin_center, pin_rot=pin_rot)
    result = optimize(model, z0, pinned, dataclasses.replace(CONFIG, steps=80))
    assert (result.rot[:, SOFA] == 1).all()
    room = problem.c[:, :2] * 8
    start, _ = model.decode(z0, problem.c)
    distance = lambda positions: ((positions[:, SOFA] * room - pin_center[:, SOFA]) ** 2).sum(-1).sqrt()
    assert distance(result.positions).mean() < distance(start.detach()).mean()
    snapped = snap_pins(result.positions, pinned)
    torch.testing.assert_close(snapped[:, SOFA] * room, pin_center[:, SOFA])
