"""Machine check: versions, GPU and VRAM, determinism, and the training speed of an evaluator-sized CNN.

Run it on each machine before training:  python run.py check-env
"""
from __future__ import annotations

import platform
import time

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from spacegen.seed import set_seed

TRAIN_SAMPLES = 42_000  # 70% of the 60k Set B layouts (the training split)
BATCH = 256
RASTER = 128  # pixels per side of the evaluator input
BUDGET_S = 30 * 60  # our target: evaluator training within 30 minutes


def evaluator_sized_cnn() -> nn.Sequential:
    """A CNN with the layer shapes of the evaluator (spacegen/models/evaluator.py), for timing."""
    layers: list[nn.Module] = []
    for c_in, c_out in [(4, 16), (16, 32), (32, 64), (64, 64)]:
        layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
    layers += [nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 2)]
    return nn.Sequential(*layers)


def _setup(device: str, batch: int):
    set_seed(0)
    model = evaluator_sized_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(batch, 4, RASTER, RASTER, device=device)
    y = torch.randint(0, 2, (batch,), device=device).float()
    return model, optimizer, x, y


def _step(model, optimizer, x, y) -> None:
    out = model(x)
    loss = F.binary_cross_entropy_with_logits(out[:, 0], y) + F.mse_loss(out[:, 1], y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def weights_after_training(device: str, steps: int = 5, batch: int = 32) -> torch.Tensor:
    """All weights after a few training steps from seed 0."""
    model, optimizer, x, y = _setup(device, batch)
    for _ in range(steps):
        _step(model, optimizer, x, y)
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def samples_per_second(device: str, steps: int, warmup: int = 2) -> float:
    """Training throughput at the evaluator's input size and batch, with deterministic kernels."""
    model, optimizer, x, y = _setup(device, BATCH)
    for i in range(warmup + steps):
        if i == warmup:
            _sync(device)
            start = time.perf_counter()
        _step(model, optimizer, x, y)
    _sync(device)
    return BATCH * steps / (time.perf_counter() - start)


def main() -> None:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, torch {torch.__version__}")
    devices = ["cpu"]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB VRAM, CUDA {torch.version.cuda}")
        devices.insert(0, "cuda")
    else:
        print("GPU: none (CPU only)")

    print(f"\nEvaluator-sized training ({RASTER}x{RASTER} rasters, batch {BATCH}, {TRAIN_SAMPLES:,} samples per epoch):")
    for device in devices:
        identical = torch.equal(weights_after_training(device), weights_after_training(device))
        sps = samples_per_second(device, steps=20 if device == "cuda" else 3)
        epoch_s = TRAIN_SAMPLES / sps
        line = (f"  {device:4s} deterministic={identical}  {sps:7,.0f} samples/s  "
                f"{epoch_s:5.0f} s per epoch  {BUDGET_S / epoch_s:4.0f} epochs in 30 min")
        if device == "cuda":
            line += f"  peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        print(line)


if __name__ == "__main__":
    main()
