"""Set B: labelled layouts for training the CNN evaluator.

The evaluator must see invalid layouts as well as valid ones. Half of Set B are generator
layouts as they come (all valid). The other half are perturbed copies of fresh generator
layouts, a tenth of Set B per type:

  jitter_<sigma>     each item moves by N(0, sigma^2) along x and y with probability
                     jitter_items (at least one moves); sigma 0.1, 0.3 or 0.6 m, a third each
  rotation           one or two items turn to a facing that looks different; centres kept
  random             every item placed uniformly inside the room, facing a random way
  overlap            one item's centre moved to a random point inside another item
  near_miss_overlap  one item slid along x or y into a neighbour until they overlap by an
                     area drawn log-uniformly around the 0.005 m^2 tolerance
  near_miss_door     one item slid along x or y to somewhere between near_miss_gap clear of
                     the door zone and near_miss_gap inside it
                     (the two near-miss types share the near-miss tenth)

The checker labels every sample (valid, and which checks fail) and the rule score gives its
quality. The perturbation type is kept, so the evaluator's results can be reported per type
(experiment E9a: the near-misses are the hardest).

Report the labels per type:  python -m spacegen.perturb --layouts 2000
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spacegen import geometry
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch, stack_layouts
from spacegen.generator import (GeneratorConfig, RoomRange, failed_checks, generate_layout, load_generator_config,
                                sample_condition)
from spacegen.layout import Layout, canonicalize
from spacegen.quality import quality_score
from spacegen.rules import Rules, check_layout, door_geometry, load_rules, reachability

MIN_SHARED = 0.1  # a near-miss slides an item only towards something it faces over at least this width (m)


@dataclass(frozen=True)
class SetBConfig:
    clean_share: float  # generator layouts kept as they come
    jitter_sigmas: tuple[float, ...]  # m
    jitter_items: float  # chance that each item is jittered
    near_miss_area: tuple[float, float]  # overlap area, drawn log-uniformly (m^2)
    near_miss_gap: float  # door near-miss: from this far clear of the zone to this far inside (m)
    near_miss_reach: float  # near-miss moves up to this long are preferred (m)

    def shares(self) -> dict[str, float]:
        """Share of Set B for each perturbation type."""
        each = (1 - self.clean_share) / 5  # jitter, rotation, random, overlap and near-miss
        jitter = {f"jitter_{s:g}": each / len(self.jitter_sigmas) for s in self.jitter_sigmas}
        return {"clean": self.clean_share, **jitter, "rotation": each, "random": each, "overlap": each,
                "near_miss_overlap": each / 2, "near_miss_door": each / 2}


def load_set_b_config(path: Path = DEFAULT_CONFIG) -> SetBConfig:
    raw = dict(load_config(path)["set_b"])
    return SetBConfig(jitter_sigmas=tuple(raw.pop("jitter_sigmas")), near_miss_area=tuple(raw.pop("near_miss_area")),
                      **raw)


# --------------------------------------------------------------------------- perturbations

def perturb(layout: Layout, kind: str, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
            config: SetBConfig) -> Layout | None:
    """The layout changed by one perturbation type, or None when a near-miss finds no move."""
    if kind == "clean":
        return layout
    if kind.startswith("jitter_"):
        return jitter(layout, float(kind.removeprefix("jitter_")), config.jitter_items, rng)
    if kind == "rotation":
        return turn(layout, rng, catalog)
    if kind == "random":
        return random_placement(layout, rng)
    if kind == "overlap":
        return force_overlap(layout, rng)
    if kind == "near_miss_overlap":
        return near_miss_overlap(layout, rng, catalog, rules, config)
    if kind == "near_miss_door":
        return near_miss_door(layout, rng, catalog, rules, config)
    raise ValueError(f"unknown perturbation {kind!r}")


def jitter(layout: Layout, sigma: float, chance: float, rng: np.random.Generator) -> Layout:
    """Each item moves by N(0, sigma^2) along x and y with probability `chance`; at least one moves."""
    present = np.flatnonzero(layout.mask)
    moved = present[rng.uniform(size=len(present)) < chance]
    if len(moved) == 0:
        moved = rng.choice(present, size=1)
    center = layout.center.copy()
    center[moved] += rng.normal(0, sigma, size=(len(moved), 2))
    return dataclasses.replace(layout, center=center)


def turn(layout: Layout, rng: np.random.Generator, catalog: RoomCatalog) -> Layout:
    """One or two items turned to a facing that looks different; their centres stay."""
    symmetry = np.array([s.rot_symmetry for s in catalog.slots])
    turnable = np.flatnonzero(layout.mask & (symmetry < 4))  # a side table looks the same any way round
    rot = layout.rot.copy()
    for k in rng.choice(turnable, size=min(len(turnable), int(rng.integers(1, 3))), replace=False):
        same = geometry.equivalent_rotations(int(rot[k]), int(symmetry[k]))
        rot[k] = rng.choice([r for r in range(4) if r not in same])
    return dataclasses.replace(layout, rot=rot)


def random_placement(layout: Layout, rng: np.random.Generator) -> Layout:
    """Every item placed uniformly where it fits inside the room, facing a random way."""
    present = layout.mask
    rot = np.where(present, rng.integers(4, size=len(present)), 0)
    half = geometry.effective_size(layout.size, rot) / 2
    center = rng.uniform(half, np.maximum(layout.room - half, half))
    return dataclasses.replace(layout, center=np.where(present[:, None], center, 0.0), rot=rot)


def force_overlap(layout: Layout, rng: np.random.Generator) -> Layout:
    """One item's centre moved to a random point inside another item's footprint."""
    mover, target = rng.choice(np.flatnonzero(layout.mask), size=2, replace=False)
    point = layout.center[target] + rng.uniform(-0.5, 0.5, size=2) * layout.eff_size[target]
    return layout.with_item(int(mover), center=point)


def near_miss_overlap(layout: Layout, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                      config: SetBConfig) -> Layout | None:
    """One item slid along x or y into a neighbour until they overlap by a log-uniform area.

    Only moves that touch nothing else (other items, the door zone, the walls) and keep every
    item reachable count, so the overlap is the one thing a checker has to judge.
    """
    area = float(np.exp(rng.uniform(*np.log(config.near_miss_area))))
    present = np.flatnonzero(layout.mask)
    moves = []
    for i in present:
        for j in present[present != i]:
            for axis in (0, 1):
                shared = _shared(layout.center[i], layout.eff_size[i], layout.center[j], layout.eff_size[j], axis)
                if shared >= MIN_SHARED:
                    moves.append((i, _slide(layout, i, layout.center[j], layout.eff_size[j], axis, -area / shared), j))
    return _pick_move(layout, moves, rng, catalog, rules, config, door_clear=True)


def near_miss_door(layout: Layout, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                   config: SetBConfig) -> Layout | None:
    """One item slid along x or y to between near_miss_gap clear of the door zone and near_miss_gap inside it."""
    door = door_geometry(layout.width, layout.depth, layout.door_wall, layout.door_offset, rules.door)
    gap = rng.uniform(-config.near_miss_gap, config.near_miss_gap)
    moves = []
    for i in np.flatnonzero(layout.mask):
        for axis in (0, 1):
            if _shared(layout.center[i], layout.eff_size[i], door.zone_center, door.zone_size, axis) >= MIN_SHARED:
                moves.append((i, _slide(layout, i, door.zone_center, door.zone_size, axis, gap), None))
    return _pick_move(layout, moves, rng, catalog, rules, config, door_clear=False)


def _shared(center, size, other_center, other_size, axis: int) -> float:
    """How far two boxes face each other across `axis`: the overlap of their extents along the other axis."""
    k = 1 - axis
    return float(min(center[k] + size[k] / 2, other_center[k] + other_size[k] / 2)
                 - max(center[k] - size[k] / 2, other_center[k] - other_size[k] / 2))


def _slide(layout: Layout, i: int, target_center, target_size, axis: int, gap: float) -> np.ndarray:
    """Item i's centre after sliding along `axis` until its face is `gap` from the target's
    facing face (a negative gap is a penetration)."""
    center, size = layout.center[i], layout.eff_size[i]
    side = np.sign(center[axis] - target_center[axis]) or 1.0
    moved = center.copy()
    moved[axis] = target_center[axis] + side * ((size[axis] + target_size[axis]) / 2 + gap)
    return moved


def _pick_move(layout: Layout, moves: list, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
               config: SetBConfig, door_clear: bool, reach_checks: int = 8) -> Layout | None:
    """A random move that leaves the item in the room, clear of everything but its target, and
    every item still reachable, so the near-miss is the only thing a checker has to judge.
    Short moves are tried first; at most `reach_checks` candidates get the reachability test."""
    door = door_geometry(layout.width, layout.depth, layout.door_wall, layout.door_offset, rules.door)
    clean, short = [], []
    for i, center, target in moves:
        moved = layout.with_item(int(i), center=center)
        eff = moved.eff_size
        if not geometry.inside_room(center, eff[i], moved.room, tol=0.0).all():
            continue
        others = moved.mask.copy()
        others[[k for k in (i, target) if k is not None]] = False
        if geometry.overlap_area(center, eff[i], moved.center[others], eff[others]).max(initial=0.0) > 0:
            continue
        if door_clear and geometry.overlap_area(center, eff[i], door.zone_center, door.zone_size) > 0:
            continue
        (short if np.linalg.norm(center - layout.center[i]) <= config.near_miss_reach else clean).append(moved)
    candidates = [short[k] for k in rng.permutation(len(short))] + [clean[k] for k in rng.permutation(len(clean))]
    for moved in candidates[:reach_checks]:
        if not reachability(moved, catalog, rules).unreachable:
            return moved
    return None


# --------------------------------------------------------------------------- Set B

@dataclass(frozen=True)
class LabelledSet:
    layouts: LayoutBatch
    info: pd.DataFrame  # one row per layout: perturbation, base style, labels and quality terms


def generate_set_b(n_layouts: int, rng: np.random.Generator, catalog: RoomCatalog, rules: Rules,
                   config: GeneratorConfig, set_b: SetBConfig, rooms: RoomRange | None = None) -> LabelledSet:
    """Set B: exact shares per perturbation type (in random order), each sample a perturbed
    copy of a fresh generator layout, labelled by the checker and the rule score."""
    layouts, rows = [], []
    for kind in _kinds(n_layouts, set_b.shares(), rng):
        for _ in range(100):
            base = generate_layout(sample_condition(rng, catalog, rules, config, rooms), rng, catalog, rules, config)
            changed = base.layout and perturb(base.layout, kind, rng, catalog, rules, set_b)
            if changed:
                break
        else:
            raise RuntimeError(f"no {kind} sample after 100 rooms; check the settings")
        layout = canonicalize(changed, catalog)
        reach = reachability(layout, catalog, rules)
        result = check_layout(layout, catalog, rules, reach=reach)
        q = quality_score(layout, catalog, rules, reach=reach)
        rows.append({"perturbation": kind, "style": base.style, "valid": result.valid,
                     "failed": " ".join(failed_checks(result)), "quality": q.total, "align": q.align,
                     "relations": q.relations, "circulation": q.circulation, "space": q.space})
        layouts.append(layout)
    return LabelledSet(stack_layouts(layouts, catalog), pd.DataFrame(rows))


def label_summary(info: pd.DataFrame) -> pd.DataFrame:
    """Per perturbation type, clean first and the near-misses last: samples, share valid and mean quality."""
    summary = info.groupby("perturbation").agg(samples=("valid", "size"), valid=("valid", "mean"),
                                               quality=("quality", "mean"))
    return summary.loc[sorted(summary.index, key=type_order)]


def type_order(kind: str) -> tuple:
    order = ("clean", "jitter", "rotation", "random", "overlap", "near_miss_overlap", "near_miss_door")
    base, _, sigma = kind.partition("_") if kind.startswith("jitter_") else (kind, "", "")
    return order.index(base) if base in order else len(order), float(sigma or 0)


def _kinds(n: int, shares: dict[str, float], rng: np.random.Generator) -> list[str]:
    """n perturbation types in exact proportion to `shares` (largest remainders), shuffled."""
    names = list(shares)
    exact = np.array([shares[name] for name in names]) * n
    counts = np.floor(exact).astype(int)
    counts[np.argsort(counts - exact, kind="stable")[: n - counts.sum()]] += 1
    kinds = np.repeat(names, counts)
    rng.shuffle(kinds)
    return kinds.tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Set B layouts and report their labels per perturbation type.")
    parser.add_argument("--layouts", type=int, default=2000, help="layouts to generate (default 2000)")
    parser.add_argument("--seed", type=int, default=None, help="default: the seed in configs/default.yaml")
    args = parser.parse_args(argv)
    seed = load_config()["seed"] if args.seed is None else args.seed
    catalog, rules = load_room_catalog("living_room"), load_rules()
    start = time.perf_counter()
    result = generate_set_b(args.layouts, np.random.default_rng(seed), catalog, rules, load_generator_config(),
                            load_set_b_config())
    seconds = time.perf_counter() - start
    info = result.info
    print(f"{len(info)} layouts in {seconds:.1f} s ({1e3 * seconds / len(info):.1f} ms each); "
          f"{info['valid'].mean():.1%} valid")
    print(label_summary(info).round(3).to_string())
    print("failed checks:", info.loc[info["failed"] != "", "failed"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
