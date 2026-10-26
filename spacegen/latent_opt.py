"""Latent optimization (M2): repair a generated layout by moving its latent vector z.

The CVAE decoder turns z into a layout. A layout decoded from a random z often breaks a rule:
two items overlap, an item sticks out of the room or stands in front of the door. Instead of
throwing it away, we run gradient descent on z itself, with the decoder frozen, to make the
decoded layout break the rules as little as possible:

  L(z) = lambda_overlap * sum over item pairs of pen_ij^2       (how deep two items overlap)
       + lambda_room    * sum over items of out_i               (how far an item leaves the room)
       + lambda_door    * sum over items of pen(i, door zone)^2  (how deep it stands in the door zone)
       + lambda_pin     * sum over pinned items of ||p_k - p_k*||^2   (distance from its pin, m)
       + lambda_anchor  * 0.5 * ||z - z0||^2                    (stay near the starting point z0)

Only z changes; the decoder's weights do not. The 64 candidates of a request are optimized at
once, as a batch of independent problems (the decoder is in eval() mode, so BatchNorm does not
mix them).

- Rotations are discrete, so they are not optimized: every item keeps the decoder's facing at
  z0, or the facing the user asked for when the item is pinned.
- The anchor keeps each candidate near its own start, so the 64 candidates stay different. A
  prior term ||z||^2 would instead pull all of them towards z = 0, the same layout.
- A candidate stops once its constraint terms (the anchor not counted) fall below `tolerance`.
- Adam does not lower the loss at every step, so success is judged by the final loss against
  the initial one (Optimized.final and Optimized.initial).
- The rule checker still decides afterwards. A repaired layout can still fail it, for example
  when an item cannot be reached: reachability is not differentiable, so it is not in L(z).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from spacegen import geometry
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import unpack_conditions
from spacegen.models.cvae import CVAE


@dataclass(frozen=True)
class LatentOptConfig:
    steps: int = 150  # Adam steps (E8 tries 0 to 200)
    lr: float = 0.05
    lambda_overlap: float = 10.0
    lambda_room: float = 10.0
    lambda_door: float = 10.0
    lambda_pin: float = 20.0
    lambda_anchor: float = 0.05
    margin: float = 0.05  # items closer than this (m) already count as overlapping
    tolerance: float = 1e-4  # a candidate stops once its constraint terms fall below this


def load_latent_opt_config(path: Path = DEFAULT_CONFIG, **overrides) -> LatentOptConfig:
    return LatentOptConfig(**{**load_config(path)["latent_opt"], **overrides})


@dataclass(frozen=True)
class Problem:
    """What the candidates must satisfy, as tensors on the decoder's device (B candidates, K slots)."""
    c: torch.Tensor  # (B, 7 + 3K) condition vectors: the room, its door and the furniture
    zone_center: torch.Tensor  # (B, 2) centre of the door's clearance zone (m)
    zone_size: torch.Tensor  # (B, 2) its extent along x and y (m)
    pin_mask: torch.Tensor | None = None  # (B, K) True for pinned items
    pin_center: torch.Tensor | None = None  # (B, K, 2) where they are pinned (m)
    pin_rot: torch.Tensor | None = None  # (B, K) the facing asked for, or -1 to keep the decoder's


@dataclass
class Optimized:
    z: torch.Tensor  # (B, latent) the returned latent codes
    positions: torch.Tensor  # (B, K, 2) normalized centres decoded from z (unclamped)
    rot: torch.Tensor  # (B, K) the fixed rotation classes
    initial: torch.Tensor  # (B,) L(z) at z0
    final: torch.Tensor  # (B,) L(z) at the returned z
    steps: torch.Tensor  # (B,) steps each candidate took


def constraint_terms(positions: torch.Tensor, rot: torch.Tensor, problem: Problem, num_slots: int,
                     margin: float) -> dict[str, torch.Tensor]:
    """The four constraint terms of L(z) before weighting, one value per candidate (B,)."""
    room, mask, size = unpack_conditions(problem.c, num_slots)
    present = mask > 0.5
    center = positions * room[:, None, :]  # normalized (u, v) -> meters
    eff = geometry.effective_size(size, rot)  # footprint along x and y after rotation
    terms = {
        "overlap": geometry.overlap_penalty(center, eff, margin=margin, mask=present),
        "room": (geometry.out_of_room(center, eff, room) * present).sum(dim=1),
        "door": (geometry.penetration_depth(center, eff, problem.zone_center[:, None, :], problem.zone_size[:, None, :],
                                            margin=margin) ** 2 * present).sum(dim=1),
    }
    if problem.pin_mask is not None:
        terms["pin"] = (((center - problem.pin_center) ** 2).sum(dim=-1) * problem.pin_mask).sum(dim=1)
    else:
        terms["pin"] = torch.zeros_like(terms["room"])
    return terms


def total_loss(terms: dict[str, torch.Tensor], z: torch.Tensor, z0: torch.Tensor,
               config: LatentOptConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """(L(z) per candidate, its constraint part alone) from the terms of constraint_terms()."""
    constraint = (config.lambda_overlap * terms["overlap"] + config.lambda_room * terms["room"]
                  + config.lambda_door * terms["door"] + config.lambda_pin * terms["pin"])
    loss = constraint + config.lambda_anchor * 0.5 * ((z - z0) ** 2).sum(dim=1)  # + the anchor
    return loss, constraint


def initial_rotations(model: CVAE, z0: torch.Tensor, problem: Problem) -> torch.Tensor:
    """The decoder's most likely facing (arg-max of the logits) at z0, with pinned facings written over it."""
    with torch.no_grad():
        _, logits = model.decode(z0, problem.c)
    rot = logits.argmax(dim=-1)
    if problem.pin_rot is not None:
        rot = torch.where(problem.pin_rot >= 0, problem.pin_rot, rot)
    return rot


def optimize(model: CVAE, z0: torch.Tensor, problem: Problem, config: LatentOptConfig) -> Optimized:
    """Gradient descent with Adam on z, starting from z0 (B, latent), with the decoder frozen."""
    model.eval()  # BatchNorm uses its running statistics, so every candidate is decoded on its own
    k = model.config.num_slots
    rot = initial_rotations(model, z0, problem)  # fixed for the whole optimization

    def evaluate(z: torch.Tensor):
        positions, _ = model.decode(z, problem.c)
        terms = constraint_terms(positions, rot, problem, k, config.margin)
        loss, constraint = total_loss(terms, z, z0, config)
        return positions, loss, constraint

    z = z0.detach().clone().requires_grad_(True)  # the only tensor that is optimized
    optimizer = torch.optim.Adam([z], lr=config.lr)
    done = torch.zeros(len(z0), dtype=torch.bool, device=z0.device)  # candidates that have stopped
    kept = z0.detach().clone()  # the z of each stopped candidate, from the step it stopped
    steps = torch.zeros(len(z0), dtype=torch.long, device=z0.device)
    initial = None
    for step in range(config.steps + 1):
        _, loss, constraint = evaluate(z)
        if step == 0:
            initial = loss.detach()
        # A candidate whose constraint terms are (almost) zero is repaired: keep its z from now on.
        newly_done = (constraint.detach() < config.tolerance) & ~done
        kept[newly_done] = z.detach()[newly_done]
        done = done | newly_done
        if done.all() or step == config.steps:
            break
        # The gradient of the summed loss with respect to z alone: each candidate's loss depends
        # only on its own row of z, so row i receives exactly dL_i / dz_i. The decoder's weights
        # get no gradient and never change.
        (gradient,) = torch.autograd.grad(loss.sum(), z)
        z.grad = gradient * (~done)[:, None]  # no gradient for stopped candidates
        optimizer.step()  # Adam's momentum still moves stopped rows, which is why `kept` holds their z
        steps += (~done).long()
    with torch.no_grad():
        final_z = torch.where(done[:, None], kept, z.detach())
        positions, final, _ = evaluate(final_z)
    return Optimized(final_z, positions, rot, initial, final, steps)


def snap_pins(positions: torch.Tensor, problem: Problem) -> torch.Tensor:
    """Pinned items placed exactly on their requested centres (positions are normalized)."""
    if problem.pin_mask is None:
        return positions
    room, _, _ = unpack_conditions(problem.c, positions.shape[1])
    pinned = problem.pin_center / room[:, None, :]
    return torch.where(problem.pin_mask[..., None], pinned, positions)
