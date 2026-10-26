"""Box geometry shared by the rule checker and the differentiable losses.

Every piece of furniture is an axis-aligned box: a centre (x, y) and an effective size, its
extent along x and along y after rotation.

Conventions:
- Units are meters. The room is [0, W] x [0, D] with the origin at the bottom-left corner.
- Rotation class r is the direction the item's front faces: 0 = North (+y), 1 = East (+x),
  2 = South (-y), 3 = West (-x). Classes 1 and 3 swap the catalog width and depth.

Each function works on NumPy arrays (the rule checker, the generator) and on PyTorch tensors
(the losses, which need gradients) and returns the same kind. The small helpers below (_relu,
_minimum, ...) pick the NumPy or the PyTorch version of an operation. This way each formula is
written once, and the checker and the losses can never disagree about the geometry. Inputs
broadcast; the last axis of a centre or size is (x, y), and functions that take a whole layout
expect an item axis before it: centre and size (..., K, 2), room (W, D) as (..., 2).

Two measures of overlap:
- overlap_area, the exact intersection area, is what the checker uses (H2, H3).
- penetration_depth, how far two boxes must move apart to stop overlapping, is what the losses
  use: the area has zero gradient when one box lies inside the other along an axis, while the
  penetration depth always pushes the boxes apart.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]

# Unit vector of the front direction for each rotation class: N, E, S, W.
_FACING = ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))

# |t| is smoothed as sqrt(t^2 + SMOOTH_EPS) inside the penetration depth, so its gradient stays finite.
SMOOTH_EPS = 1e-6


# --------------------------------------------------------------------------- NumPy or PyTorch

def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def _relu(x):
    return torch.relu(x) if _is_torch(x) else np.maximum(x, 0.0)


def _minimum(a, b):
    return torch.minimum(a, b) if _is_torch(a) else np.minimum(a, b)


def _maximum(a, b):
    return torch.maximum(a, b) if _is_torch(a) else np.maximum(a, b)


def _where(cond, a, b):
    return torch.where(cond, a, b) if _is_torch(a) else np.where(cond, a, b)


def _sqrt(x):
    return torch.sqrt(x) if _is_torch(x) else np.sqrt(x)


def _sum(x, axis):
    return x.sum(dim=axis) if _is_torch(x) else x.sum(axis=axis)


def _all(x, axis):
    return x.all(dim=axis) if _is_torch(x) else x.all(axis=axis)


def _like(values, like: Array) -> Array:
    """`values` as an array of the same kind, dtype and device as `like`."""
    if _is_torch(like):
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)
    return np.asarray(values, dtype=like.dtype)


def _as_rot(rot, like: Array) -> Array:
    """Rotation classes as an integer array of the same kind (and device) as `like`."""
    if _is_torch(like):
        return torch.as_tensor(rot, device=like.device)
    return np.asarray(rot)


# --------------------------------------------------------------------------- rotation

def effective_size(size: Array, rot) -> Array:
    """Extent along (x, y) of an item with catalog size (w, d) at rotation class `rot`.

    Facing North or South the width lies along x; facing East or West (odd classes) width and
    depth swap."""
    rot = _as_rot(rot, size)
    odd = (rot % 2 == 1)[..., None]
    return _where(odd, size[..., [1, 0]], size)


def facing_vector(rot, like: Array) -> Array:
    """Unit vector in the direction the item's front faces, same kind of array as `like`."""
    return _like(_FACING, like)[_as_rot(rot, like)]


def front_point(center: Array, eff_size: Array, rot, offset=0.0) -> Array:
    """The point `offset` meters in front of the middle of the item's front face.

    With offset 0.35 m this is where a person stands to use the item (the reachability check)."""
    rot = _as_rot(rot, center)
    half_depth = _where(rot % 2 == 0, eff_size[..., 1], eff_size[..., 0]) / 2
    return center + facing_vector(rot, center) * (half_depth + offset)[..., None]


def equivalent_rotations(rot: int, rot_symmetry: int) -> list[int]:
    """Rotation classes that give the same footprint and look.

    rot_symmetry 1: only `rot`; 2: `rot` and its 180 degree turn; 4: all four classes."""
    if rot_symmetry not in (1, 2, 4):
        raise ValueError(f"rot_symmetry must be 1, 2 or 4, got {rot_symmetry}")
    step = 4 // rot_symmetry
    return [(rot + k * step) % 4 for k in range(rot_symmetry)]


def canonical_rotation(rot, rot_symmetry):
    """The stored rotation class: `rot` reduced modulo 4 / rot_symmetry (elementwise).

    A coffee table (symmetry 2) keeps only classes 0 and 1; a side table (symmetry 4) is always
    0. So layouts that look the same get the same training target."""
    return rot % (4 // rot_symmetry)


# --------------------------------------------------------------------------- the room

def box_bounds(center: Array, size: Array) -> tuple[Array, Array]:
    """Lower-left and upper-right corners of boxes given by centre and effective size."""
    return center - size / 2, center + size / 2


def inside_room(center: Array, size: Array, room, tol: float = 0.0) -> Array:
    """H1: whether each footprint lies inside [0, W] x [0, D], allowing `tol` meters.

    Returns a boolean array with one entry per item, shape (..., K)."""
    room = _like(room, center)[..., None, :]
    lo, hi = box_bounds(center, size)
    return _all(lo >= -tol, -1) & _all(hi <= room + tol, -1)


def out_of_room(center: Array, size: Array, room) -> Array:
    """How far each footprint sticks out of the room, summed over the four walls (m), (..., K).

    The out-of-room term of latent optimization: 0, with zero gradient, for an item fully inside
    the room."""
    room = _like(room, center)[..., None, :]
    lo, hi = box_bounds(center, size)
    return _sum(_relu(-lo) + _relu(hi - room), -1)


# --------------------------------------------------------------------------- overlap (checker)

def overlap_area(c1: Array, s1: Array, c2: Array, s2: Array) -> Array:
    """Exact intersection area of two axis-aligned boxes in m^2 (broadcasts).

    Used by the checker (H2, H3) and the metrics, but not as a loss: when one box lies inside
    the other along an axis, the overlap along that axis equals the smaller size wherever the
    box is, so its gradient is 0 and an optimizer would get no push."""
    lo = _maximum(c1 - s1 / 2, c2 - s2 / 2)
    hi = _minimum(c1 + s1 / 2, c2 + s2 / 2)
    extent = _relu(hi - lo)  # overlap along x and along y
    return extent[..., 0] * extent[..., 1]


def pairwise_overlap_area(center: Array, size: Array, mask: Array | None = None) -> Array:
    """Intersection area of every pair of items, (..., K, K), 0 on the diagonal.

    `mask` (..., K) marks the items that are present; pairs with an absent item are 0."""
    return overlap_area(*_pairs(center, size)) * _pair_mask(center, mask)


def total_overlap_area(center: Array, size: Array, mask: Array | None = None) -> Array:
    """Sum of intersection areas over pairs i < j, one value per layout (m^2)."""
    return _sum(pairwise_overlap_area(center, size, mask), (-2, -1)) / 2


# --------------------------------------------------------------------------- penetration (losses)

def penetration_depth(c1: Array, s1: Array, c2: Array, s2: Array, margin: float = 0.0) -> Array:
    """The smallest shift that separates two boxes by at least `margin` (m).

    Per axis p = ReLU((s1 + s2)/2 + margin - |c1 - c2|), and the depth is the smaller of the
    two axes, the easier way out. While the boxes are closer than `margin`, its gradient with
    respect to c1 is -sign(c1 - c2) on that axis, even when one box lies inside the other, so it
    always pushes them apart. |t| is smoothed as sqrt(t^2 + SMOOTH_EPS), so the gradient stays
    finite when two centres coincide (it is exactly 0 there, by symmetry)."""
    d = c1 - c2
    per_axis = _relu((s1 + s2) / 2 + margin - _sqrt(d * d + SMOOTH_EPS))
    return _minimum(per_axis[..., 0], per_axis[..., 1])


def pairwise_penetration(center: Array, size: Array, margin: float = 0.0,
                         mask: Array | None = None) -> Array:
    """Penetration depth of every pair of items, (..., K, K), 0 on the diagonal."""
    return penetration_depth(*_pairs(center, size), margin=margin) * _pair_mask(center, mask)


def overlap_penalty(center: Array, size: Array, margin: float = 0.0,
                    mask: Array | None = None) -> Array:
    """The overlap loss: sum over pairs i < j of pen_ij^2, one value per layout."""
    pen = pairwise_penetration(center, size, margin, mask)
    return _sum(pen * pen, (-2, -1)) / 2  # every pair appears twice in the K x K matrix


# --------------------------------------------------------------------------- pair helpers

def _pairs(center: Array, size: Array):
    """Views that broadcast item i against item j: (..., K, 1, 2) and (..., 1, K, 2)."""
    return (center[..., :, None, :], size[..., :, None, :],
            center[..., None, :, :], size[..., None, :, :])


def _pair_mask(center: Array, mask: Array | None) -> Array:
    """1 for pairs of two different present items, else 0; shape (..., K, K)."""
    k = center.shape[-2]
    off_diagonal = _like(1.0 - np.eye(k), center)
    if mask is None:
        return off_diagonal
    m = _like(mask, center)
    return m[..., :, None] * m[..., None, :] * off_diagonal
