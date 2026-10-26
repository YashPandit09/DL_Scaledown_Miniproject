"""CVAE tests: shapes, the masked loss, KL, and a tiny-batch overfit."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as distribution_kl

from spacegen.dataset import split_targets, training_tensors, unpack_conditions
from spacegen.generator import generate_set_a, load_generator_config
from spacegen.models.cvae import (ACTIVATIONS, CVAE, CVAEConfig, CVAEOutput, active_units, cvae_loss,
                                  decoded_overlap, kl_divergence, load_cvae_config, position_error, reparameterize)
from spacegen.seed import set_seed
from tests.layouts import COFFEE_TABLE, SIDE_TABLE, SOFA, TV_UNIT


@pytest.fixture(scope="module")
def batch(catalog, rules):
    return generate_set_a(64, np.random.default_rng(0), catalog, rules, load_generator_config()).layouts


@pytest.fixture(scope="module")
def data(batch, catalog):
    return training_tensors(batch, catalog)


def test_config_file_holds_the_spec_defaults():
    assert load_cvae_config() == CVAEConfig()
    assert (CVAEConfig().latent, CVAEConfig().hidden, CVAEConfig().depth) == (16, 256, 2)
    for bad in (dict(activation="swish"), dict(position_head="tanh"), dict(position_loss="l3"), dict(depth=0)):
        with pytest.raises(ValueError):
            CVAEConfig(**bad)


def test_shapes_and_parameter_count(data):
    set_seed(0)
    model = CVAE()
    out = model(data.x, data.c)
    assert out.positions.shape == (64, 6, 2) and out.rot_logits.shape == (64, 6, 4)
    assert out.mu.shape == out.logvar.shape == out.z.shape == (64, 16)
    assert ((out.positions > 0) & (out.positions < 1)).all()  # Sigmoid head
    assert sum(p.numel() for p in model.parameters()) == 177_732  # about 0.18 M parameters


def test_condition_vectors_unpack_to_meters(batch, data, catalog):
    room, mask, size = unpack_conditions(data.c, 6)
    np.testing.assert_allclose(room.numpy(), batch.room, atol=1e-5)
    np.testing.assert_array_equal(mask.numpy(), batch.mask)
    np.testing.assert_allclose(size.numpy(), batch.sizes(catalog), atol=1e-5)


def test_kl_matches_torch_distributions():
    gen = torch.Generator().manual_seed(0)
    mu, logvar = torch.randn(8, 16, generator=gen), torch.randn(8, 16, generator=gen)
    expected = distribution_kl(Normal(mu, torch.exp(0.5 * logvar)), Normal(0.0, 1.0)).sum(dim=1)
    torch.testing.assert_close(kl_divergence(mu, logvar), expected)
    assert torch.equal(kl_divergence(torch.zeros(2, 4), torch.zeros(2, 4)), torch.zeros(2))  # the prior itself


def test_reparameterization_passes_gradients_to_mu_and_logvar():
    mu = torch.randn(4, 3, requires_grad=True)
    logvar = torch.randn(4, 3, requires_grad=True)
    reparameterize(mu, logvar, torch.Generator().manual_seed(5)).sum().backward()
    eps = torch.randn(4, 3, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(mu.grad, torch.ones(4, 3))  # dz/dmu = 1
    torch.testing.assert_close(logvar.grad, 0.5 * torch.exp(0.5 * logvar.detach()) * eps)  # dz/dsigma = eps


def _output(data, positions, logits):
    zeros = torch.zeros(len(data), 16)
    return CVAEOutput(positions, logits, zeros, zeros, zeros)


def test_absent_items_and_unlearned_rotations_do_not_count(data):
    gen = torch.Generator().manual_seed(1)
    positions = torch.rand(len(data), 6, 2, generator=gen, requires_grad=True)
    logits = torch.randn(len(data), 6, 4, generator=gen, requires_grad=True)
    config = CVAEConfig()
    loss = cvae_loss(_output(data, positions, logits), data, beta=0.0, config=config)
    loss.total.backward()
    absent = data.mask == 0
    assert absent.any() and (positions.grad[absent] == 0).all() and (logits.grad[absent] == 0).all()
    assert (logits.grad[:, SIDE_TABLE] == 0).all()  # a side table looks the same any way round
    assert (positions.grad[:, SOFA] != 0).any() and (logits.grad[:, SOFA] != 0).any()


def test_loss_terms_follow_their_definitions(data):
    gen = torch.Generator().manual_seed(2)
    positions, logits = torch.rand(len(data), 6, 2, generator=gen), torch.randn(len(data), 6, 4, generator=gen)
    config = CVAEConfig(lambda_rot=0.5)
    loss = cvae_loss(_output(data, positions, logits), data, beta=0.3, config=config)
    target_positions, target_rotations = split_targets(data.x)
    position = (data.mask * ((positions - target_positions) ** 2).sum(-1)).sum(1).mean()
    ce = F.cross_entropy(logits.reshape(-1, 4), target_rotations.argmax(-1).reshape(-1), reduction="none")
    rotation = (data.rot_mask * ce.view(len(data), 6)).sum(1).mean()
    torch.testing.assert_close(loss.position, position)
    torch.testing.assert_close(loss.rotation, rotation)
    torch.testing.assert_close(loss.total, position + 0.5 * rotation)  # mu = logvar = 0: KL is 0


@pytest.mark.parametrize("kind, expected", [("mse", 0.04), ("mae", 0.2), ("huber", 0.05 * (0.2 - 0.025))])
def test_position_losses(kind, expected):
    predicted, target = torch.full((1, 1, 2), 0.7), torch.full((1, 1, 2), 0.5)
    error = position_error(predicted, target, CVAEConfig(position_loss=kind, huber_delta=0.05))
    assert error.item() == pytest.approx(2 * expected)  # summed over u and v


@pytest.mark.parametrize("options", [
    *({"activation": name} for name in ACTIVATIONS),
    {"batch_norm": False, "depth": 6},  # E3b's deep network without BatchNorm
    {"dropout": 0.0, "latent": 4},
    {"position_head": "linear", "position_loss": "huber"},
    {"lambda_overlap": 1.0},
])
def test_every_option_builds_and_trains_a_step(data, options):
    model = CVAE(CVAEConfig(**options))
    loss = cvae_loss(model(data.x, data.c), data, beta=0.1, config=model.config)
    loss.total.backward()
    assert torch.isfinite(loss.total) and all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_linear_head_is_clamped_only_when_generating(data):
    model = CVAE(CVAEConfig(position_head="linear")).eval()
    with torch.no_grad():
        model.head_pos.bias.fill_(3.0)  # push every position far beyond the room
    assert (model(data.x, data.c).positions > 1).any()  # training sees the error outside the room
    positions, _ = model.generate(data.c)
    assert positions.max() <= 1.0 and positions.min() >= 0.0


def test_overlap_term_is_zero_apart_and_positive_on_top(data):
    c = data.c[:1]
    apart = torch.tensor([[[0.5, 0.85], [0.5, 0.06], [0.5, 0.6], [0.95, 0.3], [0.85, 0.65], [0.1, 0.9]]])
    rotations = torch.tensor([[2, 0, 0, 3, 3, 0]])
    on_top = apart.clone()
    on_top[0, TV_UNIT] = on_top[0, SOFA] + torch.tensor([0.02, 0.01])  # dropped onto the sofa (not exactly
    on_top.requires_grad_(True)  # on its centre, where the push is 0 by symmetry)
    config = CVAEConfig(lambda_overlap=1.0)
    overlap = decoded_overlap(on_top, c, rotations, config)
    overlap.sum().backward()
    assert overlap.item() > 0 and on_top.grad[0, TV_UNIT].abs().sum() > 0
    room, mask, size = unpack_conditions(c, 6)
    if mask[0, COFFEE_TABLE] and mask[0, SOFA]:
        assert decoded_overlap(apart, c, rotations, config).item() < overlap.item()


def test_active_units_count_dimensions_that_carry_information():
    mu = torch.zeros(100, 16)
    mu[:, :3] = torch.randn(100, 3, generator=torch.Generator().manual_seed(0))  # three informative dimensions
    assert active_units(mu, torch.zeros(100, 16)) == 3
    assert active_units(torch.zeros(100, 16), torch.zeros(100, 16)) == 0  # collapse: q(z|x) = prior


def test_the_same_seed_gives_the_same_model(data):
    set_seed(0)
    first = CVAE()
    set_seed(0)
    second = CVAE()
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_generate_draws_z_from_the_prior(data):
    set_seed(0)
    model = CVAE().eval()
    first, _ = model.generate(data.c, torch.Generator().manual_seed(1))
    again, _ = model.generate(data.c, torch.Generator().manual_seed(1))
    other, _ = model.generate(data.c, torch.Generator().manual_seed(2))
    assert torch.equal(first, again) and not torch.equal(first, other)


def test_tiny_batch_overfit(data):
    """The loss on 64 samples goes near zero in a few
    hundred steps (dropout off, no KL term, so the model can memorize)."""
    set_seed(0)
    model = CVAE(CVAEConfig(dropout=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(500):
        loss = cvae_loss(model(data.x, data.c), data, beta=0.0, config=model.config)
        first = first if first is not None else loss.total.item()
        optimizer.zero_grad()
        loss.total.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.c, sample=False)
    target_positions, target_rotations = split_targets(data.x)
    mse = (((out.positions - target_positions) ** 2) * data.mask[..., None]).sum() / (2 * data.mask.sum())
    right = (out.rot_logits.argmax(-1) == target_rotations.argmax(-1)).float()
    assert loss.total.item() < 0.01 * first
    assert mse.item() < 1e-4
    assert (right * data.rot_mask).sum() == data.rot_mask.sum()  # every learned rotation right
