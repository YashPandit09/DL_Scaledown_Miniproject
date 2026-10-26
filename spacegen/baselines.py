"""The methods without a neural network: B1 uniform random, B2 statistical sampler, G0 the generator.

Every baseline samples layouts for a Condition (the room, its door and the items with their
variants): sample(cond, n, rng) returns Samples(layouts, attempts). B1 and B2 return n raw
layouts, unfiltered (attempts = n). G0 returns up to n valid layouts and counts every
placement attempt, so for all three the raw valid rate is valid layouts / attempts; for G0
that is its acceptance per attempt.

B1  every item placed uniformly where it fits inside the room, facing a random way (the spec's
    "centre uniform in the room", kept inside so that B1 does not fail on the walls alone).
B2  fitted on Set A training layouts, no neural network: each item's (u, v, rotation) is drawn
    from a training layout with the same door wall and a room in the same width and depth bins,
    plus N(0, b2_noise^2) on u and v. Items are placed one after another in slot order, and one
    that overlaps an item already placed, or (b2_redraw_outside, on by default) sticks out of the
    room, is drawn again, up to b2_redraws times. Without that second condition B2 fails mostly
    because u and v taken from a larger training room put wall items through a smaller room's wall.
G0  generate_layout(), the Set A generator itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spacegen import geometry
from spacegen.catalog import RoomCatalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch
from spacegen.generator import Condition, GeneratorConfig, generate_layout
from spacegen.layout import Layout, canonicalize, make_layout
from spacegen.rules import WALLS, Rules


@dataclass(frozen=True)
class BaselineConfig:
    b2_bins: int  # width bins and depth bins, equal shares of the training rooms
    b2_min_examples: int  # fewer examples in a bin: borrow from wider groups
    b2_noise: float  # sigma of the noise on u and v
    b2_redraws: int  # draws again while an item overlaps one already placed
    b2_redraw_outside: bool  # ... or sticks out of the room


def load_baseline_config(path: Path = DEFAULT_CONFIG) -> BaselineConfig:
    return BaselineConfig(**load_config(path)["baselines"])


@dataclass(frozen=True)
class Samples:
    layouts: list[Layout]  # in canonical form; raw (unchecked) for B1 and B2, valid for G0
    attempts: int  # samples drawn; for G0 every placement attempt


class UniformBaseline:
    """B1: every item uniformly where it fits inside the room, facing a random way."""

    name = "B1"

    def __init__(self, catalog: RoomCatalog):
        self.catalog = catalog

    def sample(self, cond: Condition, n: int, rng: np.random.Generator, pins: dict | None = None) -> Samples:
        """`pins`: {slot name: Pin}; a pinned item stands on its spot, with its facing if given."""
        return Samples([self._one(cond, rng, pins or {}) for _ in range(n)], n)

    def _one(self, cond: Condition, rng: np.random.Generator, pins: dict) -> Layout:
        items = {}
        for name, variant_id in cond.items.items():
            variant = self.catalog.slot(name).variant(variant_id)
            rot = int(rng.integers(4))
            half = geometry.effective_size(np.array([variant.w, variant.d]), rot) / 2
            room = np.array([cond.width, cond.depth])
            items[name] = (*rng.uniform(half, np.maximum(room - half, half)), rot)
            if name in pins:
                items[name] = (pins[name].x, pins[name].y, rot if pins[name].facing is None else pins[name].facing)
        return _layout(cond, items, self.catalog)


class StatisticalBaseline:
    """B2: per-item (u, v, rotation) drawn from training layouts of similar rooms, placed one
    after another with redraws on overlap. Fit with fit(train)."""

    name = "B2"

    def __init__(self, catalog: RoomCatalog, config: BaselineConfig):
        self.catalog, self.config = catalog, config
        self.width_edges = self.depth_edges = None
        self.examples: dict = {}  # (slot, door wall, width bin, depth bin) -> (M, 3) rows of u, v, rotation

    def fit(self, train: LayoutBatch) -> StatisticalBaseline:
        """Collect each slot's (u, v, rotation) by door wall and room-size bin."""
        shares = np.linspace(0, 1, self.config.b2_bins + 1)[1:-1]
        self.width_edges = np.quantile(train.room[:, 0], shares)
        self.depth_edges = np.quantile(train.room[:, 1], shares)
        width_bin = np.digitize(train.room[:, 0], self.width_edges)
        depth_bin = np.digitize(train.room[:, 1], self.depth_edges)
        uv = train.center / train.room[:, None, :]
        self.examples = {}
        for k in range(self.catalog.num_slots):
            present = train.mask[:, k]
            rows = np.column_stack([uv[present, k], train.rot[present, k]])
            keys = np.column_stack([train.door_wall[present], width_bin[present], depth_bin[present]])
            for key in {tuple(row) for row in keys.tolist()}:
                self.examples[(k, *key)] = rows[(keys == key).all(axis=1)]
            for wall in range(len(WALLS)):  # pooled over room sizes, then over everything
                self.examples[(k, wall, None, None)] = rows[keys[:, 0] == wall]
            self.examples[(k, None, None, None)] = rows
        return self

    def sample(self, cond: Condition, n: int, rng: np.random.Generator, pins: dict | None = None) -> Samples:
        """`pins`: {slot name: Pin}; pinned items are placed first, on their spots, and the
        others are drawn again while they overlap them."""
        if self.width_edges is None:
            raise RuntimeError("fit() B2 on training layouts first")
        return Samples([self._one(cond, rng, pins or {}) for _ in range(n)], n)

    def _one(self, cond: Condition, rng: np.random.Generator, pins: dict) -> Layout:
        room = np.array([cond.width, cond.depth])
        wall = WALLS.index(cond.door_wall)
        sizes = (int(np.digitize(cond.width, self.width_edges)), int(np.digitize(cond.depth, self.depth_edges)))
        placed_center, placed_eff, items = [], [], {}
        order = sorted(self.catalog.slots, key=lambda slot: slot.name not in pins)  # pinned first, then slot order
        for slot in order:  # slot order: sofa first
            if slot.name not in cond.items:
                continue
            variant = slot.variant(cond.items[slot.name])
            pool = self._pool(slot.index, wall, sizes)
            if slot.name in pins:
                pin = pins[slot.name]
                rot = pool[rng.integers(len(pool))][2] if pin.facing is None else pin.facing
                center = np.array([pin.x, pin.y], dtype=float)
                eff = geometry.effective_size(np.array([variant.w, variant.d]), int(rot))
            for _ in range(0 if slot.name in pins else self.config.b2_redraws + 1):
                u, v, rot = pool[rng.integers(len(pool))]
                center = (np.array([u, v]) + rng.normal(0, self.config.b2_noise, size=2)) * room
                eff = geometry.effective_size(np.array([variant.w, variant.d]), int(rot))
                outside = self.config.b2_redraw_outside and not geometry.inside_room(center, eff, room).all()
                overlaps = bool(placed_center) and geometry.overlap_area(
                    center, eff, np.array(placed_center), np.array(placed_eff)).max() > 0
                if not (outside or overlaps):
                    break
            placed_center.append(center)
            placed_eff.append(eff)
            items[slot.name] = (*center, int(rot))
        return _layout(cond, items, self.catalog)

    def _pool(self, slot: int, wall: int, sizes: tuple[int, int]) -> np.ndarray:
        for key in ((slot, wall, *sizes), (slot, wall, None, None), (slot, None, None, None)):
            rows = self.examples.get(key)
            if rows is not None and len(rows) >= self.config.b2_min_examples:
                return rows
        rows = self.examples.get((slot, None, None, None))
        if rows is None or len(rows) == 0:
            raise ValueError(f"no training layout has a {self.catalog.slots[slot].name}")
        return rows


class GeneratorBaseline:
    """G0: the Set A generator, keeping the first valid layout of each run of attempts."""

    name = "G0"

    def __init__(self, catalog: RoomCatalog, rules: Rules, config: GeneratorConfig):
        self.catalog, self.rules, self.config = catalog, rules, config

    def sample(self, cond: Condition, n: int, rng: np.random.Generator) -> Samples:
        layouts, attempts = [], 0
        for _ in range(n):
            result = generate_layout(cond, rng, self.catalog, self.rules, self.config)
            attempts += len(result.attempts)
            if result.layout is not None:
                layouts.append(result.layout)
        return Samples(layouts, attempts)


def _layout(cond: Condition, items: dict, catalog: RoomCatalog) -> Layout:
    return canonicalize(make_layout(catalog, cond.width, cond.depth, cond.door_wall, cond.door_offset, items,
                                    cond.items), catalog)
