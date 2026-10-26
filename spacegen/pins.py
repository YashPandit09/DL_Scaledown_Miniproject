"""Pinned furniture (experiment E12): the user fixes an item, the method arranges the rest.

A Pin is a requested centre in meters and, optionally, a facing. Every method treats it the same
way in the end: the pinned item is put exactly on its spot (snap), so the pin holds by
construction, and the rule checker then decides whether the layout around it is valid.

  M2      latent optimization pulls the item towards its pin (lambda_pin) with its facing fixed,
          then the item is snapped (spacegen.pipeline.LatentOptSampler)
  M1      the CVAE's sample, snapped (spacegen.pipeline.CVAESampler)
  G0-pin  a generator layout with the pinned item moved onto its spot (PinnedSampler)
  B1, B2  the pinned item placed first, the others sampled; B2 redraws them against it
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spacegen import geometry
from spacegen.baselines import Samples
from spacegen.catalog import RoomCatalog
from spacegen.generator import Condition
from spacegen.layout import Layout, canonicalize
from spacegen.rules import Rules, door_geometry


@dataclass(frozen=True)
class Pin:
    x: float  # requested centre (m)
    y: float
    facing: int | None = None  # rotation class 0 to 3 (N, E, S, W); None leaves it to the method


def snap(layout: Layout, pins: dict[str, Pin], catalog: RoomCatalog) -> Layout:
    """The layout with every pinned item on its spot, with its facing if one was asked for, in
    canonical form (a symmetric item's facing is reduced like any other)."""
    for name, pin in pins.items():
        layout = layout.with_item(catalog.slot(name).index, center=(pin.x, pin.y), rot=pin.facing)
    return canonicalize(layout, catalog)


def displacement(layout: Layout, pins: dict[str, Pin], catalog: RoomCatalog) -> float:
    """Mean distance (m) of the pinned items from their pins, before snapping."""
    return float(np.mean([np.hypot(*(layout.center[catalog.slot(name).index] - (pin.x, pin.y)))
                          for name, pin in pins.items()]))


def pin_errors(pins: dict[str, Pin], cond: Condition, catalog: RoomCatalog, rules: Rules) -> list[str]:
    """Why a pin cannot be honoured at all: the item is not in the room's furniture, or on its spot
    it would stick out of the room or stand in the door's clearance zone (with the facing asked
    for; without one, in every facing)."""
    errors = []
    room = np.array([cond.width, cond.depth])
    door = door_geometry(cond.width, cond.depth, cond.door_wall, cond.door_offset, rules.door)
    for name, pin in pins.items():
        label = name.replace("_", " ")
        if name not in cond.items:
            errors.append(f"the pinned {label} is not among the furniture")
            continue
        if pin.facing is not None and pin.facing not in (0, 1, 2, 3):
            errors.append(f"the pinned {label}'s facing must be 0 to 3 (N, E, S, W)")
            continue
        variant = catalog.slot(name).variant(cond.items[name])
        center, size = np.array([pin.x, pin.y]), np.array([variant.w, variant.d])
        facings = (0, 1) if pin.facing is None else (pin.facing,)
        fits = [bool(geometry.inside_room(center, geometry.effective_size(size, rot), room,
                                          tol=rules.hard_checks.in_room_tol)) for rot in facings]
        clear = [float(geometry.overlap_area(center, geometry.effective_size(size, rot), door.zone_center,
                                             door.zone_size)) <= rules.hard_checks.door_overlap_tol for rot in facings]
        if not any(fits):
            errors.append(f"the pinned {label} would stick out of the room at ({pin.x:g}, {pin.y:g}) m")
        elif not any(inside and free for inside, free in zip(fits, clear)):
            errors.append(f"the pinned {label} would stand in the door's clearance zone")
    return errors


class PinnedSampler:
    """A sampler with pinned items. B1 and B2 take the pins themselves (the pinned item is placed
    first). Any other sampler's layouts are sampled as usual and the pinned items then moved onto
    their spots: for the generator that is G0-pin (E12). `last_displacement` holds,
    per layout of the last call, how far the pinned items were from their pins before the snap."""

    def __init__(self, base, pins: dict[str, Pin], catalog: RoomCatalog, takes_pins: bool = False):
        self.base, self.pins, self.catalog, self.takes_pins = base, pins, catalog, takes_pins
        self.name = f"{getattr(base, 'name', 'sampler')}-pin"
        self.last_displacement = np.zeros(0)

    def sample(self, cond: Condition, n: int, rng: np.random.Generator) -> Samples:
        if self.takes_pins:
            samples = self.base.sample(cond, n, rng, pins=self.pins)
            self.last_displacement = np.zeros(len(samples.layouts))
            return samples
        samples = self.base.sample(cond, n, rng)
        self.last_displacement = np.array([displacement(layout, self.pins, self.catalog) for layout in samples.layouts])
        return Samples([snap(layout, self.pins, self.catalog) for layout in samples.layouts], samples.attempts)
