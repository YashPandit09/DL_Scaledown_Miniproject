"""The conditional VAE (CVAE): learns to generate furniture layouts for a given room.

Notation: x is a layout (where each item stands and which way it faces), c is the condition
(the room, its door and the furniture in it), z is a latent vector of 16 numbers.

  encoder  q(z | x, c):  [x ; c] -> hidden layers -> mu, logvar      (used in training only)
  decoder  p(x | z, c):  [z ; c] -> hidden layers -> positions (K, 2), rotation logits (K, 4)

Training uses the reparameterization trick, z = mu + sigma * eps with eps ~ N(0, I): the
randomness sits in eps, so z stays differentiable in mu and sigma and the reconstruction loss
can train the encoder. To generate, the encoder is not used at all: z ~ N(0, I) goes straight
into the decoder, and each new z gives a different layout for the same room.

Each hidden layer is Linear -> BatchNorm -> activation -> Dropout. The position head ends in a
Sigmoid, because a position is a fraction of the room's width or depth, in [0, 1]. The rotation
head gives 4 logits per item (facing N, E, S or W); the softmax happens inside F.cross_entropy.

The loss of one layout, with m_k = 1 when item k is in the room and 0 when it is not:

  reconstruction = sum_k m_k * position_error_k + lambda_rot * sum_k r_k * cross_entropy_k
  KL             = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))
  loss           = mean(reconstruction) + beta * mean(KL)    (+ lambda_overlap * mean(overlap))

r_k equals m_k except for the side table, which looks the same in every facing, so there is no
rotation to learn for it. The overlap term is off (lambda_overlap = 0) except in experiment E2.

Every setting the experiments vary is a field of CVAEConfig (configs/default.yaml, section cvae).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from spacegen import geometry
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import TrainingTensors, condition_dim, split_targets, target_dim, unpack_conditions

# The hidden-layer activations compared in experiments E3a and E3b.
ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.01), "elu": nn.ELU, "gelu": nn.GELU,
               "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
POSITION_LOSSES = ("mse", "mae", "huber")  # compared in experiment E2


@dataclass(frozen=True)
class CVAEConfig:
    num_slots: int = 6  # K: the furniture slots of a living room
    latent: int = 16  # size of z
    hidden: int = 256  # units per hidden layer
    depth: int = 2  # hidden layers in the encoder and in the decoder
    activation: str = "relu"  # a key of ACTIVATIONS
    batch_norm: bool = True
    dropout: float = 0.1
    position_head: str = "sigmoid"  # or "linear" (E7), clamped to [0, 1] only when generating
    position_loss: str = "mse"  # or "mae" or "huber" (E2); the final models use "mae"
    huber_delta: float = 0.05  # where the Huber loss turns from quadratic to linear
    lambda_rot: float = 1.0  # weight of the rotation cross-entropy
    lambda_overlap: float = 0.0  # weight of the optional overlap term (E2 tries 1.0)
    overlap_margin: float = 0.0  # 0, because training layouts may touch

    def __post_init__(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {self.activation!r}")
        if self.position_head not in ("sigmoid", "linear"):
            raise ValueError(f"position_head must be sigmoid or linear, got {self.position_head!r}")
        if self.position_loss not in POSITION_LOSSES:
            raise ValueError(f"position_loss must be one of {POSITION_LOSSES}, got {self.position_loss!r}")
        if self.depth < 1 or not 0 <= self.dropout < 1:
            raise ValueError("depth must be at least 1 and dropout in [0, 1)")


def load_cvae_config(path: Path = DEFAULT_CONFIG, **overrides) -> CVAEConfig:
    return CVAEConfig(**{**load_config(path)["cvae"], **overrides})


def _hidden_stack(in_features: int, config: CVAEConfig) -> nn.Sequential:
    """`depth` hidden layers, each Linear -> BatchNorm -> activation -> Dropout.

    BatchNorm or Dropout is left out when switched off (E3b, E6). The order of the layers fixes
    the parameter names in a saved model (encoder.0.weight, encoder.1.running_mean, ...)."""
    layers: list[nn.Module] = []
    for i in range(config.depth):
        layers.append(nn.Linear(in_features if i == 0 else config.hidden, config.hidden))
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden))
        layers.append(ACTIVATIONS[config.activation]())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
    return nn.Sequential(*layers)


@dataclass
class CVAEOutput:
    positions: torch.Tensor  # (B, K, 2) normalized centres (u, v)
    rot_logits: torch.Tensor  # (B, K, 4)
    mu: torch.Tensor  # (B, latent)
    logvar: torch.Tensor  # (B, latent)
    z: torch.Tensor  # (B, latent)


class CVAE(nn.Module):
    def __init__(self, config: CVAEConfig = CVAEConfig()):
        super().__init__()
        self.config = config
        k = config.num_slots
        # Encoder: the layout and the room (36 + 25 = 61 inputs) -> the mean and log-variance of z.
        self.encoder = _hidden_stack(target_dim(k) + condition_dim(k), config)
        self.to_mu = nn.Linear(config.hidden, config.latent)
        self.to_logvar = nn.Linear(config.hidden, config.latent)
        # Decoder: z and the room (16 + 25 = 41 inputs) -> a layout.
        self.decoder = _hidden_stack(config.latent + condition_dim(k), config)
        self.head_pos = nn.Linear(config.hidden, 2 * k)  # (u, v) of each item
        self.head_rot = nn.Linear(config.hidden, 4 * k)  # 4 facing logits per item

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z | x, c): mean and log-variance, each (B, latent)."""
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(x | z, c): positions (B, K, 2) and rotation logits (B, K, 4).

        A linear head's positions are returned unclamped, so the loss still sees how far they
        leave the room."""
        h = self.decoder(torch.cat([z, c], dim=1))
        positions = self.head_pos(h)
        if self.config.position_head == "sigmoid":
            positions = torch.sigmoid(positions)
        k = self.config.num_slots
        return positions.view(-1, k, 2), self.head_rot(h).view(-1, k, 4)

    def forward(self, x: torch.Tensor, c: torch.Tensor, sample: bool = True,
                generator: torch.Generator | None = None) -> CVAEOutput:
        """Training pass: encode, draw z (or take z = mu when sample is False), decode."""
        mu, logvar = self.encode(x, c)
        z = reparameterize(mu, logvar, generator) if sample else mu
        positions, logits = self.decode(z, c)
        return CVAEOutput(positions, logits, mu, logvar, z)

    @torch.no_grad()
    def generate(self, c: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """New layouts for rooms c: z drawn from the prior N(0, I) and decoded, positions clamped
        to [0, 1]. The encoder is not used."""
        z = torch.randn(c.shape[0], self.config.latent, generator=generator, device=c.device, dtype=c.dtype)
        positions, logits = self.decode(z, c)
        return positions.clamp(0.0, 1.0), logits


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """z = mu + sigma * eps with sigma = exp(logvar / 2) and eps ~ N(0, I).

    Sampling z directly would have no gradient with respect to mu and sigma; written this way,
    dz/dmu = 1 and dz/dsigma = eps. The network predicts log(sigma^2), so sigma is always positive."""
    eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu + torch.exp(0.5 * logvar) * eps


def kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, 1)) in closed form, for every sample and latent dimension: (B, latent)."""
    return -0.5 * (1 + logvar - mu ** 2 - logvar.exp())


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL of each sample's posterior to the prior N(0, I), summed over the latent dimensions: (B,)."""
    return kl_per_dimension(mu, logvar).sum(dim=1)


def active_units(mu: torch.Tensor, logvar: torch.Tensor, threshold: float = 0.01) -> int:
    """Latent dimensions the model uses: those whose KL, averaged over the samples, exceeds
    `threshold` nats. 0 active units means posterior collapse: the decoder ignores z."""
    return int((kl_per_dimension(mu, logvar).mean(dim=0) > threshold).sum())


def position_error(predicted: torch.Tensor, target: torch.Tensor, config: CVAEConfig) -> torch.Tensor:
    """Error of each item's position, summed over u and v: (B, K). Squared (MSE), absolute (MAE)
    or Huber, as config.position_loss says."""
    if config.position_loss == "mse":
        error = (predicted - target) ** 2
    elif config.position_loss == "mae":
        error = (predicted - target).abs()
    else:
        error = F.huber_loss(predicted, target, reduction="none", delta=config.huber_delta)
    return error.sum(dim=-1)


def decoded_overlap(positions: torch.Tensor, c: torch.Tensor, rotations: torch.Tensor, config: CVAEConfig) -> torch.Tensor:
    """The optional overlap term: sum over item pairs of the squared penetration depth, for the
    decoded positions placed in meters with the true rotations. (B,)."""
    room, mask, size = unpack_conditions(c, config.num_slots)
    center = positions * room[:, None, :]
    return geometry.overlap_penalty(center, geometry.effective_size(size, rotations), margin=config.overlap_margin,
                                    mask=mask > 0.5)


@dataclass
class LossTerms:
    total: torch.Tensor  # what backward() is called on
    recon: torch.Tensor  # batch means from here on
    position: torch.Tensor
    rotation: torch.Tensor
    kl: torch.Tensor
    overlap: torch.Tensor


def cvae_loss(output: CVAEOutput, batch: TrainingTensors, beta: float, config: CVAEConfig) -> LossTerms:
    """The masked CVAE loss of a mini-batch (see the module docstring)."""
    target_positions, target_rotations = split_targets(batch.x)
    classes = target_rotations.argmax(dim=-1)  # the true facing of each item, 0 to 3

    # 1. Position error, counted only for the items that are in the room (mask 1).
    position = (batch.mask * position_error(output.positions, target_positions, config)).sum(dim=1)

    # 2. Rotation: cross-entropy over the four facings, only where the facing is learned.
    cross_entropy = F.cross_entropy(output.rot_logits.reshape(-1, 4), classes.reshape(-1),
                                    reduction="none").view_as(classes)
    rotation = (batch.rot_mask * cross_entropy).sum(dim=1)
    recon = position + config.lambda_rot * rotation

    # 3. KL to the prior, weighted by beta (it rises from 0 during KL annealing).
    kl = kl_divergence(output.mu, output.logvar)
    total = recon.mean() + beta * kl.mean()

    # 4. The optional overlap term (only in experiment E2).
    overlap = torch.zeros_like(recon)
    if config.lambda_overlap > 0:
        overlap = decoded_overlap(output.positions, batch.c, classes, config)
        total = total + config.lambda_overlap * overlap.mean()
    return LossTerms(total, recon.mean(), position.mean(), rotation.mean(), kl.mean(), overlap.mean())
