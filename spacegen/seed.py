"""One call that seeds every random number generator and sets deterministic mode."""
from __future__ import annotations

import random

import numpy as np
import torch

import spacegen


def set_seed(seed: int, deterministic: bool = True) -> np.random.Generator:
    """Seed Python, NumPy and PyTorch (CPU and every GPU); return a NumPy Generator.

    With deterministic=True, two runs with the same seed on the same machine and software
    versions give bit-identical results: PyTorch then uses only deterministic kernels and
    raises an error for an operation that has none, instead of silently varying.
    """
    if deterministic and spacegen.CUBLAS_SET_TOO_LATE:
        raise RuntimeError(
            "CUDA was used before spacegen was imported, so CUBLAS_WORKSPACE_CONFIG came "
            "too late and GPU results may vary. Import spacegen before any CUDA work."
        )
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
