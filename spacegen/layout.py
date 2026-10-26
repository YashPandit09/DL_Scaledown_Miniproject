"""A furniture layout as plain arrays, one row per catalog slot, in meters.

Canonical form: a coffee table looks the same after a half turn, and a side table after any
quarter turn. If the data stored either facing at random, two identical layouts would have
different training targets, and the rotation head would learn noise. canonicalize() therefore
stores every symmetric item's rotation in one standard way (spacegen.geometry.canonical_rotation).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from spacegen import geometry
from spacegen.catalog import RoomCatalog


@dataclass(frozen=True, eq=False)
class Layout:
    room_type: str
    width: float  # W, extent along x
    depth: float  # D, extent along y
    door_wall: str  # N, E, S or W
    door_offset: float  # o in [0, 1]: where the door is along its wall
    center: np.ndarray  # (K, 2) item centres
    rot: np.ndarray  # (K,) rotation classes 0 to 3
    size: np.ndarray  # (K, 2) catalog (w, d) of each item; 0 for absent slots
    mask: np.ndarray  # (K,) True where the slot is present
    variant_ids: tuple  # (K,) catalog variant ids; None for absent slots

    @property
    def room(self) -> np.ndarray:
        """(W, D)."""
        return np.array([self.width, self.depth])

    @property
    def eff_size(self) -> np.ndarray:
        """(K, 2) footprint extents along x and y after rotation."""
        return geometry.effective_size(self.size, self.rot)

    def with_item(self, slot: int, center=None, rot: int | None = None) -> Layout:
        """A copy with one present item moved and/or turned."""
        if not self.mask[slot]:
            raise ValueError(f"slot {slot} is absent")
        new_center, new_rot = self.center.copy(), self.rot.copy()
        if center is not None:
            new_center[slot] = center
        if rot is not None:
            new_rot[slot] = _check_rot(rot)
        return dataclasses.replace(self, center=new_center, rot=new_rot)

    def without(self, slot: int) -> Layout:
        """A copy with one slot marked absent, its row zero-filled like every absent slot."""
        center, rot, size, mask = self.center.copy(), self.rot.copy(), self.size.copy(), self.mask.copy()
        center[slot], rot[slot], size[slot], mask[slot] = 0.0, 0, 0.0, False
        ids = list(self.variant_ids)
        ids[slot] = None
        return dataclasses.replace(self, center=center, rot=rot, size=size, mask=mask, variant_ids=tuple(ids))


def make_layout(catalog: RoomCatalog, width: float, depth: float, door_wall: str,
                door_offset: float, items: dict[str, tuple[float, float, int]],
                variants: dict[str, str] | None = None) -> Layout:
    """Build a layout from {slot name: (x, y, rot)}; slots not named are absent.

    `variants` maps slot names to variant ids; other present slots get their default variant."""
    variants = variants or {}
    unused = set(variants) - set(items)
    if unused:
        raise KeyError(f"variants given for slots that are not in the layout: {sorted(unused)}")
    k = catalog.num_slots
    center, size = np.zeros((k, 2)), np.zeros((k, 2))
    rot, mask = np.zeros(k, dtype=np.int64), np.zeros(k, dtype=bool)
    ids: list[str | None] = [None] * k
    for name, (x, y, r) in items.items():
        slot = catalog.slot(name)
        variant = slot.variant(variants.get(name))
        center[slot.index] = (x, y)
        rot[slot.index] = _check_rot(r)
        size[slot.index] = (variant.w, variant.d)
        mask[slot.index] = True
        ids[slot.index] = variant.id
    return Layout(catalog.room_type, float(width), float(depth), door_wall, float(door_offset),
                  center, rot, size, mask, tuple(ids))


def canonicalize(layout: Layout, catalog: RoomCatalog) -> Layout:
    """The canonical form of a layout, the only form used as a training target.

    A symmetric item keeps its rotation modulo 4 / rot_symmetry, so a coffee table turned by 180
    degrees, or a side table turned any way, gives the same layout. The footprint does not change:
    a half turn keeps the extents, and the catalog requires rot_symmetry 4 items to be square.
    Absent slots are zero-filled."""
    if layout.room_type != catalog.room_type:
        raise ValueError(f"layout is a {layout.room_type}, catalog is for {catalog.room_type}")
    symmetry = np.array([s.rot_symmetry for s in catalog.slots])
    present = layout.mask
    return dataclasses.replace(
        layout,
        center=np.where(present[:, None], layout.center, 0.0),
        rot=np.where(present, geometry.canonical_rotation(layout.rot, symmetry), 0),
        size=np.where(present[:, None], layout.size, 0.0),
        variant_ids=tuple(v if m else None for v, m in zip(layout.variant_ids, present)))


def is_canonical(layout: Layout, catalog: RoomCatalog) -> bool:
    """True if canonicalize() would leave the layout unchanged."""
    canonical = canonicalize(layout, catalog)
    return (np.array_equal(canonical.rot, layout.rot) and np.array_equal(canonical.center, layout.center)
            and np.array_equal(canonical.size, layout.size) and canonical.variant_ids == layout.variant_ids)


def layout_to_dict(layout: Layout, catalog: RoomCatalog, door_width: float,
                   metrics: dict | None = None, meta: dict | None = None) -> dict:
    """The layout JSON shared by the pipeline, the app and the exports."""
    items = []
    for k in np.flatnonzero(layout.mask):
        v = catalog.slot(int(k)).variant(layout.variant_ids[k])
        items.append({"slot": int(k), "id": v.id, "w": v.w, "d": v.d, "h": v.h,
                      "x": float(layout.center[k, 0]), "y": float(layout.center[k, 1]),
                      "rotation": int(layout.rot[k]), "price": v.price})
    data = {
        "room": {"type": layout.room_type, "width": layout.width, "depth": layout.depth},
        "door": {"wall": layout.door_wall, "offset": layout.door_offset, "width": door_width},
        "items": items,
    }
    if metrics is not None:
        data["metrics"] = metrics
    if meta is not None:
        data["meta"] = meta
    return data


def layout_from_dict(data: dict, catalog: RoomCatalog) -> Layout:
    """Read a layout JSON; item sizes must match the catalog variants they name."""
    if data["room"]["type"] != catalog.room_type:
        raise ValueError(f"layout is a {data['room']['type']}, catalog is for {catalog.room_type}")
    items, variants = {}, {}
    for item in data["items"]:
        slot = catalog.slot(item["slot"])
        if slot.name in items:
            raise ValueError(f"slot {slot.index} ({slot.name}) appears twice")
        variant = slot.variant(item["id"])
        if not np.allclose((item["w"], item["d"]), (variant.w, variant.d)):
            raise ValueError(f"{item['id']}: size {item['w']} x {item['d']} differs from the catalog")
        items[slot.name] = (item["x"], item["y"], item["rotation"])
        variants[slot.name] = variant.id
    return make_layout(catalog, data["room"]["width"], data["room"]["depth"], data["door"]["wall"],
                       data["door"]["offset"], items, variants)


def _check_rot(rot: int) -> int:
    if rot not in range(4):
        raise ValueError(f"rotation class must be 0, 1, 2 or 3, got {rot}")
    return int(rot)
