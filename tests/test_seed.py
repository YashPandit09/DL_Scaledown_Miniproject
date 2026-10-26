"""Reproducibility: the same seed gives identical results on the same machine."""
import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spacegen.seed import set_seed

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")),
]


def _weights_after_training(device: str) -> torch.Tensor:
    """Train a small CVAE-like MLP and a small CNN together; return all their weights."""
    set_seed(123)
    mlp = nn.Sequential(
        nn.Linear(41, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 36)
    ).to(device)
    cnn = nn.Sequential(
        nn.Conv2d(4, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(8 * 16 * 16, 2),
    ).to(device)
    params = [*mlp.parameters(), *cnn.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-2)
    for _ in range(10):
        out = mlp(torch.randn(32, 41, device=device))
        loss = F.mse_loss(torch.sigmoid(out[:, :12]), torch.rand(32, 12, device=device))
        loss = loss + F.cross_entropy(out[:, 12:].reshape(-1, 4), torch.randint(0, 4, (32 * 6,), device=device))
        scores = cnn(torch.rand(16, 4, 32, 32, device=device))
        labels = torch.randint(0, 2, (16,), device=device).float()
        loss = loss + F.binary_cross_entropy_with_logits(scores[:, 0], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.cat([p.detach().flatten().cpu() for p in params])


@pytest.mark.parametrize("device", DEVICES)
def test_same_seed_gives_bit_identical_weights(device):
    assert torch.equal(_weights_after_training(device), _weights_after_training(device))


def test_seed_repeats_python_numpy_and_torch_draws():
    def draws():
        rng = set_seed(7)
        return random.random(), np.random.rand(), rng.random(), torch.rand(1).item()

    assert draws() == draws()


def test_cublas_workspace_is_configured():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8")
