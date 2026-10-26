"""The CNN evaluator: looks at a layout as an image and predicts (1) whether it is valid and
(2) its rule quality score.

The input is a 4-channel 128 x 128 image of the layout (spacegen/raster.py): the room, the
furniture, the fronts of the items and the door's clearance zone, at 6.25 cm per pixel.

  4 x [Conv 3x3 (padding 1) -> BatchNorm -> ReLU -> MaxPool 2]   128 -> 64 -> 32 -> 16 -> 8 pixels
  Flatten (64 channels x 8 x 8 = 4096 values) -> Dropout(0.3) -> Linear(4096, 128) -> ReLU
  head_valid: Linear(128, 1), a logit: P(valid) = sigmoid(logit)
  head_score: Linear(128, 1), the predicted quality score

Loss: BCE on the validity logit + lambda_score * MSE on the score, over every layout (the
quality score is defined for invalid layouts too).

The rule checker, not this network, decides whether a layout is valid. The pipeline uses the
CNN only to rank the layouts that have already passed the checker.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from spacegen.config import DEFAULT_CONFIG, load_config


@dataclass(frozen=True)
class EvaluatorConfig:
    channels: tuple[int, ...] = (16, 32, 64, 64)  # output channels of the four conv blocks
    hidden: int = 128  # units of the dense layer
    dropout: float = 0.3
    lambda_score: float = 1.0  # weight of the score regression next to the BCE
    batch: int = 256  # training mini-batch size
    lr: float = 1e-3  # Adam learning rate


def load_evaluator_config(path: Path = DEFAULT_CONFIG, **overrides) -> EvaluatorConfig:
    raw = {**load_config(path)["evaluator"], **overrides}
    return EvaluatorConfig(**{**raw, "channels": tuple(raw["channels"])})


def evaluator_config_from_record(values: dict) -> EvaluatorConfig:
    """The EvaluatorConfig of a training run, from the "evaluator" entry of its run.json.

    Only the settings EvaluatorConfig has are read. The saved evaluator's record also lists
    three training options that no experiment changed and that were removed later (a Huber
    score loss, its delta and a BCE class weight); they do not affect the network itself."""
    known = {field.name for field in dataclasses.fields(EvaluatorConfig)}
    values = {key: value for key, value in values.items() if key in known}
    return EvaluatorConfig(**{**values, "channels": tuple(values["channels"])})


class Evaluator(nn.Module):
    def __init__(self, config: EvaluatorConfig = EvaluatorConfig(), pixels: int = 128, in_channels: int = 4):
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        for c_in, c_out in zip((in_channels, *config.channels), config.channels):  # 4->16, 16->32, 32->64, 64->64
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
        side = pixels // 2 ** len(config.channels)  # each block halves the image: 128 px -> 8 px
        if side < 1 or pixels % 2 ** len(config.channels):
            raise ValueError(f"{pixels} px cannot be halved {len(config.channels)} times")
        self.features = nn.Sequential(*blocks, nn.Flatten())  # (B, 64 * 8 * 8)
        self.dense = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.channels[-1] * side * side, config.hidden),
                                   nn.ReLU())
        self.head_valid = nn.Linear(config.hidden, 1)
        self.head_score = nn.Linear(config.hidden, 1)

    def forward(self, raster: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Validity logits (B,) and predicted quality scores (B,) for images (B, 4, P, P)."""
        h = self.dense(self.features(raster))
        return self.head_valid(h).squeeze(1), self.head_score(h).squeeze(1)


@dataclass
class EvaluatorLoss:
    total: torch.Tensor
    bce: torch.Tensor
    score: torch.Tensor


def evaluator_loss(valid_logit: torch.Tensor, score: torch.Tensor, valid: torch.Tensor, quality: torch.Tensor,
                   config: EvaluatorConfig) -> EvaluatorLoss:
    """BCE on the validity logit plus lambda_score times the MSE of the score, as batch means.

    The BCE takes the logit itself (binary_cross_entropy_with_logits), which is numerically
    stable: a sigmoid followed by a log can give log(0)."""
    bce = F.binary_cross_entropy_with_logits(valid_logit, valid)
    score_loss = F.mse_loss(score, quality)
    return EvaluatorLoss(bce + config.lambda_score * score_loss, bce, score_loss)
