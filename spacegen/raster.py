"""Layouts as 4-channel images, the input of the CNN evaluator.

Each pixel holds the exact fraction of its area that a shape covers (not just 0 or 1), so
moving an item by a centimetre changes the pixel values, and a near-miss overlap is visible.
The canvas has a fixed physical size (8 m, 128 pixels) with the room at its origin, so a meter
spans the same number of pixels in every room. Channels:

  0  room        coverage of the room rectangle
  1  furniture   coverage summed over the items, so an overlap shows as values above 1
  2  fronts      a thin band inside each item's front face; symmetric items get a band on
                 every equivalent face, so their arbitrary stored rotation does not show
  3  door zone   coverage of the door clearance zone

Pixel [i, j] covers x in [i s, (i+1) s] and y in [j s, (j+1) s], with s = canvas / pixels.

An axis-aligned box covers a pixel by (its overlap with the pixel's column) times (its
overlap with the pixel's row), so each channel is a sum of outer products of 1-D overlaps:
one batched matrix product (einsum) on the GPU. That is fast enough to build the images of
each training mini-batch on the fly instead of storing 60,000 of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from spacegen import geometry
from spacegen.catalog import RoomCatalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch, stack_layouts
from spacegen.layout import Layout
from spacegen.rules import WALLS, Rules, door_geometry

CHANNELS = ("room", "furniture", "fronts", "door zone")


@dataclass(frozen=True)
class RasterConfig:
    canvas: float  # meters per side
    pixels: int  # pixels per side
    front_strip: float  # thickness of the band marking an item's front (m)

    @property
    def pixel(self) -> float:
        """Pixel size in meters."""
        return self.canvas / self.pixels


def load_raster_config(path: Path = DEFAULT_CONFIG) -> RasterConfig:
    return RasterConfig(**load_config(path)["raster"])


def rasterize(center: torch.Tensor, size: torch.Tensor, rot: torch.Tensor, mask: torch.Tensor,
              room: torch.Tensor, zone_center: torch.Tensor, zone_size: torch.Tensor,
              rot_symmetry: torch.Tensor, config: RasterConfig) -> torch.Tensor:
    """(B, 4, P, P) rasters of B layouts with K slots each.

    center, size: (B, K, 2) item centres and catalog sizes (w, d); rot: (B, K) rotation
    classes; mask: (B, K) present items; room: (B, 2) as (W, D); zone_center, zone_size:
    (B, 2) door clearance zones; rot_symmetry: (K,) per slot.
    """
    edges = torch.arange(config.pixels + 1, dtype=center.dtype, device=center.device) * config.pixel
    eff = geometry.effective_size(size, rot)
    present = mask.to(center.dtype)
    one = torch.ones_like(room[:, :1])  # weight of a single box per layout

    def cover(lo, hi, weight):
        return _coverage(lo, hi, weight, edges, config.pixel)

    channels = [
        cover(torch.zeros_like(room)[:, None], room[:, None], one),
        cover(center - eff / 2, center + eff / 2, present),
        cover(*_front_strips(center, eff, rot, present, rot_symmetry, config.front_strip)),
        cover((zone_center - zone_size / 2)[:, None], (zone_center + zone_size / 2)[:, None], one),
    ]
    return torch.stack(channels, dim=1)


@dataclass(frozen=True)
class RasterInputs:
    """Layouts as tensors on one device, ready for rasterize(): built once per dataset, so that
    training rasterizes mini-batches by indexing (Set B as precomputed rasters would need 16 GB)."""
    center: torch.Tensor  # (N, K, 2)
    size: torch.Tensor  # (N, K, 2) catalog (w, d)
    rot: torch.Tensor  # (N, K)
    mask: torch.Tensor  # (N, K) bool
    room: torch.Tensor  # (N, 2)
    zone_center: torch.Tensor  # (N, 2) door clearance zone
    zone_size: torch.Tensor  # (N, 2)
    rot_symmetry: torch.Tensor  # (K,)

    def __len__(self) -> int:
        return self.center.shape[0]

    def rasterize(self, index, config: RasterConfig) -> torch.Tensor:
        """(len(index), 4, P, P) rasters of the layouts at `index`."""
        return rasterize(self.center[index], self.size[index], self.rot[index], self.mask[index], self.room[index],
                         self.zone_center[index], self.zone_size[index], self.rot_symmetry, config)


def raster_inputs(batch: LayoutBatch, catalog: RoomCatalog, rules: Rules, device: str | torch.device = "cpu",
                  dtype: torch.dtype = torch.float32) -> RasterInputs:
    """A batch of layouts as RasterInputs on `device`, with each room's door clearance zone."""
    if batch.room_type != catalog.room_type:
        raise ValueError(f"layouts are {batch.room_type}, catalog is for {catalog.room_type}")
    zones = [door_geometry(float(w), float(d), WALLS[wall], float(o), rules.door)
             for (w, d), wall, o in zip(batch.room, batch.door_wall, batch.door_offset)]

    def tensor(values, kind=dtype):
        return torch.as_tensor(np.asarray(values), dtype=kind, device=device)

    return RasterInputs(tensor(batch.center), tensor(batch.sizes(catalog)), tensor(batch.rot, torch.long),
                        tensor(batch.mask, torch.bool), tensor(batch.room),
                        tensor([z.zone_center for z in zones]), tensor([z.zone_size for z in zones]),
                        tensor([s.rot_symmetry for s in catalog.slots], torch.long))


def rasterize_layouts(layouts: Sequence[Layout], catalog: RoomCatalog, rules: Rules, config: RasterConfig,
                      device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Rasters of Layout objects, as (len(layouts), 4, P, P) on `device`."""
    if any(layout.room_type != catalog.room_type for layout in layouts):
        raise ValueError(f"every layout must be a {catalog.room_type}")
    inputs = raster_inputs(stack_layouts(layouts, catalog), catalog, rules, device, dtype)
    return inputs.rasterize(slice(None), config)


def _coverage(lo: torch.Tensor, hi: torch.Tensor, weight: torch.Tensor, edges: torch.Tensor,
              pixel: float) -> torch.Tensor:
    """Weighted sum of box coverages: lo, hi (B, N, 2), weight (B, N) -> (B, P, P).

    Pixel [i, j] of box n gets (overlap with column i) x (overlap with row j), each as a
    fraction of a pixel; the einsum forms these outer products and sums them over the boxes."""
    def along(axis: int) -> torch.Tensor:  # (B, N, P) overlap of each box with each pixel column or row
        overlap = torch.minimum(hi[..., axis, None], edges[1:]) - torch.maximum(lo[..., axis, None], edges[:-1])
        return overlap.clamp(min=0) / pixel

    return torch.einsum("bni,bnj->bij", along(0) * weight[..., None], along(1))


def _front_strips(center: torch.Tensor, eff: torch.Tensor, rot: torch.Tensor, present: torch.Tensor,
                  rot_symmetry: torch.Tensor, thickness: float):
    """Channel 2, which shows the CNN which way each item faces.

    For every item and each of its four sides, a thin band (`thickness` deep) just inside that
    side. A band counts (weight 1) only on the front face, or on every face equivalent to it for
    a symmetric item; the other bands get weight 0. Building all four sides and weighting them
    keeps the code free of loops over the items. Returns lo, hi (B, 4K, 2) and weight (B, 4K).
    """
    side = torch.arange(4, device=rot.device)
    faces = (rot[..., None] + side) % 4  # (B, K, 4) direction of each side, starting with the front
    step = 4 // rot_symmetry.to(rot.device)  # equivalent faces are every `step` sides apart
    equivalent = (side % step[:, None] == 0).to(center.dtype)  # (K, 4)
    looks_north_south = faces % 2 == 0
    eff4 = eff[:, :, None, :].expand(-1, -1, 4, -1)
    depth = torch.where(looks_north_south, eff4[..., 1], eff4[..., 0])  # extent along the face's direction
    length = torch.where(looks_north_south, eff4[..., 0], eff4[..., 1])  # extent along the face itself
    band = torch.clamp(depth, max=thickness)
    middle = center[:, :, None, :] + geometry.facing_vector(faces, center) * ((depth - band) / 2)[..., None]
    size = torch.where(looks_north_south[..., None], torch.stack([length, band], -1), torch.stack([band, length], -1))
    weight = present[:, :, None] * equivalent
    return (middle - size / 2).flatten(1, 2), (middle + size / 2).flatten(1, 2), weight.flatten(1, 2)
