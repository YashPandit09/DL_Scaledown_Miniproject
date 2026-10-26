"""Train/validation/test splits and the held-out test sets.

Training data never contains a room from a held-out region: Set A and Set B draw their rooms
from training_rooms(), which draws such rooms again. Each held-out set is drawn with the same
generator, restricted to its region (configs/default.yaml, splits.held_out):

  interpolation       floor area 22 to 26 m^2: a gap inside the training ranges
  unseen_combination  floor area above 32 m^2: each W and D occurs in training, not together
  out_of_range        W 7 to 8 m and D 6 to 7 m: beyond the training ranges

Set A and Set B are each split 70/15/15 by layout (split_indices). The diversity reference
holds several generator layouts (G0) for each of a sample of test rooms: how different the
layouts of one room can be, the yardstick for the other methods' diversity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spacegen.catalog import RoomCatalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch, stack_layouts
from spacegen.generator import (Condition, GeneratedSet, GeneratorConfig, Region, RoomRange, generate_layout,
                                generate_set_a)
from spacegen.rules import RoomRules, Rules

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    fractions: tuple[float, float, float]  # train, validation, test
    held_out: dict[str, Region]  # regions that training never sees
    held_out_rooms: int  # rooms per held-out set
    diversity_rooms: int  # test rooms in the diversity reference ...
    diversity_layouts: int  # ... with this many generator layouts each


def load_split_config(path: Path = DEFAULT_CONFIG) -> SplitConfig:
    raw = load_config(path)["splits"]
    fractions = tuple(raw["fractions"])
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"splits.fractions must be three positive shares summing to 1, got {list(fractions)}")
    held_out = {name: Region(**{key: tuple(value) for key, value in spec.items()})
                for name, spec in raw["held_out"].items()}
    return SplitConfig(fractions, held_out, raw["held_out_rooms"], raw["diversity_rooms"], raw["diversity_layouts"])


def training_rooms(room: RoomRules, split: SplitConfig) -> RoomRange:
    """The room type's sampling ranges, minus every held-out region."""
    return RoomRange(room.width, room.depth, exclude=tuple(split.held_out.values()))


def held_out_rooms(name: str, room: RoomRules, split: SplitConfig) -> RoomRange:
    """Rooms of one held-out region: drawn from its own W and D ranges where it sets them,
    otherwise from the training ranges, and kept only inside the region."""
    region = split.held_out[name]
    width = region.width if np.isfinite(region.width).all() else room.width
    depth = region.depth if np.isfinite(region.depth).all() else room.depth
    return RoomRange(width, depth, within=region)


def in_region(batch: LayoutBatch, region: Region) -> np.ndarray:
    """(N,) True for the layouts whose room lies in the region."""
    return np.array([region.contains(width, depth) for width, depth in batch.room], dtype=bool)


def leaks(batch: LayoutBatch, split: SplitConfig) -> dict[str, int]:
    """How many of the batch's rooms fall in each held-out region (all 0 for training data)."""
    return {name: int(in_region(batch, region).sum()) for name, region in split.held_out.items()}


def split_indices(n: int, fractions, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Row indices of the train, validation and test splits: a random partition of range(n)."""
    order = rng.permutation(n)
    ends = np.round(np.cumsum(fractions)[:-1] * n).astype(int)
    return {name: np.sort(part) for name, part in zip(SPLITS, np.split(order, ends))}


def generate_held_out(name: str, n_layouts: int, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                      config: GeneratorConfig, split: SplitConfig) -> GeneratedSet:
    """A held-out test set: rooms of one region, each with one generator layout as a reference."""
    return generate_set_a(n_layouts, rng, catalog, rules, config,
                          held_out_rooms(name, rules.rooms[catalog.room_type], split))


def diversity_reference(test: LayoutBatch, n_rooms: int, n_layouts: int, rng: np.random.Generator,
                        catalog: RoomCatalog, rules: Rules, config: GeneratorConfig) -> GeneratedSet:
    """Up to n_layouts generator layouts for each of n_rooms test rooms (same room, door and
    furniture), the G0 reference for the diversity ratio. info["source"] is the
    room's row in `test`."""
    sources = np.sort(rng.choice(len(test), size=min(n_rooms, len(test)), replace=False))
    layouts, info, log = [], [], []
    for room, source in enumerate(sources):
        cond = Condition.of(test.layout(int(source), catalog), catalog)
        for draw in range(n_layouts):
            result = generate_layout(cond, rng, catalog, rules, config)
            log += [{"room": room, "draw": draw, "attempt": i, "style": a.style, "outcome": a.outcome}
                    for i, a in enumerate(result.attempts, start=1)]
            if result.layout is not None:
                layouts.append(result.layout)
                info.append({"room": room, "source": int(source), "style": result.style,
                             "attempts": len(result.attempts), "quality": result.quality.total})
    return GeneratedSet(stack_layouts(layouts, catalog), pd.DataFrame(info), pd.DataFrame(log))
