"""Layouts as vectors for the CVAE, and the dataset held in memory.

A living room has K = 6 furniture slots (sofa, TV unit, coffee table, bookshelf, armchair, side
table); a room uses some of them. Every layout becomes two vectors:

Condition c, 7 + 3K = 25 values, what the user gives:
    W/8, D/8 | door wall one-hot (N, E, S, W) | door offset o | presence masks m_k |
    widths w_k/3 | depths d_k/3        (catalog sizes before rotation; 0 for absent slots)
Target x, 6K = 36 values, what the CVAE must produce, slot by slot:
    u_k = x_k / W, v_k = y_k / D (the centre as a fraction of the room), then the one-hot
    rotation class r_k (4 values). Absent slots are all zeros and are masked out of the loss.

Dividing by the room size puts every position in [0, 1] whatever the room, the range of the
Sigmoid head, and "against the east wall" is always u close to 1. The divisions by 8 and 3 bring
the room and furniture sizes to about the same range as the other inputs.

A dataset is stored as a LayoutBatch: stacked NumPy arrays in meters, in canonical form. The
vectors are computed from it, and for training the whole dataset goes to the GPU once as
TrainingTensors (about 10 MB); mini-batches are drawn by indexing, without a DataLoader.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spacegen import geometry
from spacegen.catalog import RoomCatalog
from spacegen.layout import Layout, canonicalize
from spacegen.rules import WALLS

ROOM_SCALE = 8.0  # W/8 and D/8: the raster canvas, so rooms up to 8 m (beyond the training range) still fit
SIZE_SCALE = 3.0  # w/3 and d/3: no furniture is longer than 3 m


def condition_dim(num_slots: int) -> int:
    return 7 + 3 * num_slots


def target_dim(num_slots: int) -> int:
    return 6 * num_slots


@dataclass(frozen=True, eq=False)
class LayoutBatch:
    """N layouts of one room type as stacked arrays, in meters (the dataset storage format).

    It can also carry rooms alone: decode_targets() takes the rooms and items from a batch and
    puts in new centres and rotations."""
    room_type: str
    room: np.ndarray  # (N, 2) W, D
    door_wall: np.ndarray  # (N,) index into WALLS: 0 N, 1 E, 2 S, 3 W
    door_offset: np.ndarray  # (N,) o in [0, 1]
    mask: np.ndarray  # (N, K) True where the slot is present
    variant: np.ndarray  # (N, K) index into the slot's variants; -1 for absent slots
    center: np.ndarray  # (N, K, 2) item centres; 0 for absent slots
    rot: np.ndarray  # (N, K) rotation classes; 0 for absent slots

    def __post_init__(self):
        n, k = self.mask.shape
        expected = {"room": (n, 2), "door_wall": (n,), "door_offset": (n,), "variant": (n, k),
                    "center": (n, k, 2), "rot": (n, k)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")

    def __len__(self) -> int:
        return len(self.room)

    def sizes(self, catalog: RoomCatalog) -> np.ndarray:
        """(N, K, 2) catalog (w, d) of every item; 0 for absent slots."""
        slots = np.arange(catalog.num_slots)
        return np.where(self.mask[..., None], _size_table(catalog)[slots, np.maximum(self.variant, 0)], 0.0)

    def layout(self, i: int, catalog: RoomCatalog) -> Layout:
        """Layout i as a Layout object, for the checker, plots and JSON."""
        row = self.subset([i])
        ids = tuple(catalog.slots[k].variants[v].id if m else None
                    for k, (v, m) in enumerate(zip(row.variant[0], row.mask[0])))
        return Layout(self.room_type, float(row.room[0, 0]), float(row.room[0, 1]), WALLS[row.door_wall[0]],
                      float(row.door_offset[0]), row.center[0], row.rot[0], row.sizes(catalog)[0], row.mask[0], ids)

    def subset(self, index) -> LayoutBatch:
        """The layouts at `index` (index array, boolean mask or slice), as a new batch."""
        arrays = {f.name: getattr(self, f.name)[index] for f in dataclasses.fields(self) if f.name != "room_type"}
        return LayoutBatch(self.room_type, **arrays)


_STORED = ("room", "door_wall", "door_offset", "mask", "variant", "center", "rot")


def save_layouts(path: Path, batch: LayoutBatch) -> None:
    """Write a batch to a compressed .npz file."""
    np.savez_compressed(path, room_type=np.array(batch.room_type), **{name: getattr(batch, name) for name in _STORED})


def load_layouts(path: Path) -> LayoutBatch:
    with np.load(path) as data:
        return LayoutBatch(str(data["room_type"]), **{name: data[name] for name in _STORED})


def stack_layouts(layouts: Sequence[Layout], catalog: RoomCatalog) -> LayoutBatch:
    """Stack layouts of one room type into a batch, in canonical form."""
    if not layouts:
        raise ValueError("no layouts to stack")
    canonical = [canonicalize(layout, catalog) for layout in layouts]
    variant_index = [{v.id: i for i, v in enumerate(slot.variants)} for slot in catalog.slots]
    return LayoutBatch(
        room_type=catalog.room_type,
        room=np.array([(lay.width, lay.depth) for lay in canonical]),
        door_wall=np.array([WALLS.index(lay.door_wall) for lay in canonical]),
        door_offset=np.array([lay.door_offset for lay in canonical]),
        mask=np.stack([lay.mask for lay in canonical]),
        variant=np.array([[-1 if v is None else variant_index[k][v] for k, v in enumerate(lay.variant_ids)]
                          for lay in canonical]),
        center=np.stack([lay.center for lay in canonical]),
        rot=np.stack([lay.rot for lay in canonical]))


# --------------------------------------------------------------------------- vectors

def encode_conditions(batch: LayoutBatch, catalog: RoomCatalog, dtype=np.float32) -> np.ndarray:
    """(N, 7 + 3K) condition vectors c."""
    size = batch.sizes(catalog) / SIZE_SCALE
    return np.concatenate([
        batch.room / ROOM_SCALE,  # W/8, D/8
        np.eye(len(WALLS))[batch.door_wall],  # door wall, one-hot
        batch.door_offset[:, None],  # where the door is along its wall
        batch.mask,  # which furniture is in the room
        size[..., 0],  # widths / 3
        size[..., 1],  # depths / 3
    ], axis=1).astype(dtype)


def encode_targets(batch: LayoutBatch, catalog: RoomCatalog, dtype=np.float32) -> np.ndarray:
    """(N, 6K) target vectors x: per slot (u, v) and the one-hot canonical rotation."""
    uv = batch.center / batch.room[:, None, :]
    rot = geometry.canonical_rotation(batch.rot, _rot_symmetry(catalog))
    per_slot = np.concatenate([uv, np.eye(4)[rot]], axis=2) * batch.mask[..., None]  # absent slots: all zeros
    return per_slot.reshape(len(batch), -1).astype(dtype)


def unpack_conditions(c: torch.Tensor, num_slots: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read back from condition vectors (tensors): the room (W, D) in meters (..., 2), the presence
    masks (..., K) and the catalog sizes (w, d) in meters (..., K, 2)."""
    k = num_slots
    widths, depths = c[..., 7 + k:7 + 2 * k], c[..., 7 + 2 * k:7 + 3 * k]
    return c[..., :2] * ROOM_SCALE, c[..., 7:7 + k], torch.stack([widths, depths], dim=-1) * SIZE_SCALE


def split_targets(x):
    """Target vectors (..., 6K) as positions (..., K, 2) and rotation scores (..., K, 4).

    Works on NumPy arrays and on tensors alike."""
    per_slot = x.reshape(*x.shape[:-1], -1, 6)
    return per_slot[..., :2], per_slot[..., 2:]


def decode_targets(uv, rot_scores, batch: LayoutBatch, catalog: RoomCatalog) -> LayoutBatch:
    """Layouts from targets or decoder outputs, for the rooms and items of `batch`.

    `uv` (N, K, 2) are normalized centres; `rot_scores` (N, K, 4) are one-hot targets,
    probabilities or logits: the most likely class is taken and reduced to canonical form.
    Tensors are accepted and moved to the CPU. Absent slots come back zero-filled."""
    uv, rot_scores = _numpy(uv), _numpy(rot_scores)
    rot = geometry.canonical_rotation(rot_scores.argmax(axis=-1), _rot_symmetry(catalog))
    present = batch.mask
    return dataclasses.replace(batch, center=np.where(present[..., None], uv * batch.room[:, None, :], 0.0),
                               rot=np.where(present, rot, 0))


# --------------------------------------------------------------------------- tensors for training

@dataclass(frozen=True)
class TrainingTensors:
    """The CVAE's data on one device; index it with a tensor of rows to get a mini-batch."""
    x: torch.Tensor  # (N, 6K) targets
    c: torch.Tensor  # (N, 7 + 3K) conditions
    mask: torch.Tensor  # (N, K) 1 where the slot is present: masks the position loss
    rot_mask: torch.Tensor  # (N, K) 1 where the rotation is learned: present, and not the side table

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index) -> TrainingTensors:
        return TrainingTensors(self.x[index], self.c[index], self.mask[index], self.rot_mask[index])


def training_tensors(batch: LayoutBatch, catalog: RoomCatalog, device="cpu",
                     dtype=torch.float32) -> TrainingTensors:
    """Encode a dataset once and move it to `device` (about 10 MB for 30k living rooms)."""
    learned = _rot_symmetry(catalog) < 4  # a side table looks the same any way round: nothing to learn

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    return TrainingTensors(x=tensor(encode_targets(batch, catalog, np.float64)),
                           c=tensor(encode_conditions(batch, catalog, np.float64)),
                           mask=tensor(batch.mask), rot_mask=tensor(batch.mask & learned))


def minibatches(n: int, batch_size: int, generator: torch.Generator | None = None, device="cpu",
                drop_last: bool = False) -> Iterator[torch.Tensor]:
    """Row indices for one epoch in shuffled order.

    The order is drawn on the CPU, so a seed gives the same batches on every device. Use
    drop_last when a final batch of one sample would break BatchNorm."""
    order = torch.randperm(n, generator=generator).to(device)
    stop = n - n % batch_size if drop_last else n
    for start in range(0, stop, batch_size):
        yield order[start:start + batch_size]


# --------------------------------------------------------------------------- helpers

def _rot_symmetry(catalog: RoomCatalog) -> np.ndarray:
    return np.array([s.rot_symmetry for s in catalog.slots])


def _size_table(catalog: RoomCatalog) -> np.ndarray:
    """(K, V, 2) catalog (w, d) of variant v of slot k; zero-padded where a slot has fewer variants."""
    table = np.zeros((catalog.num_slots, max(len(s.variants) for s in catalog.slots), 2))
    for s in catalog.slots:
        table[s.index, :len(s.variants)] = [(v.w, v.d) for v in s.variants]
    return table


def _numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)
