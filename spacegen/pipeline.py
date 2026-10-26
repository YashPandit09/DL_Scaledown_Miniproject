"""Inference: from a room request to the top 3 layouts.

generate() runs the whole pipeline for one request:
  1. check the request (known items, a door that fits, a warning for rooms unlike the training data)
  2. choose catalog variants within the budget and the crowding limit f_max
  3. sample 64 candidate layouts: z ~ N(0, I) decoded by the CVAE, then repaired by latent
     optimization (M2; without it, M1)
  4. keep the candidates that pass the rule checker (H1 to H4); the checker is the final authority
  5. rank the valid ones by the CNN evaluator's predicted score (or by the rule score without one)
  6. pick a diverse top 3: each pick must differ from the earlier ones by at least tau_div meters

  python run.py generate --width 5 --depth 4 --door W --offset 0.5 --items sofa,tv_unit,coffee_table

The samplers: CVAESampler is M1 (positions clamped to the room, each rotation the most likely
class), LatentOptSampler is M2 (the same draws, repaired before decoding). Both have the
baselines' sample(cond, n, rng) interface, so spacegen.evaluate runs them like B1, B2 and G0.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spacegen.baselines import Samples
from spacegen.catalog import RoomCatalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import LayoutBatch, decode_targets, encode_conditions, stack_layouts
from spacegen.generator import Condition
from spacegen.latent_opt import LatentOptConfig, Problem, optimize, snap_pins
from spacegen.layout import Layout, make_layout
from spacegen.metrics import layout_distance
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.models.evaluator import Evaluator, evaluator_config_from_record
from spacegen.pins import Pin, displacement, pin_errors, snap
from spacegen.quality import quality_score
from spacegen.raster import RasterConfig, rasterize_layouts
from spacegen.rules import Rules, check_layout, door_geometry, footprint_ratio, reachability


def condition_batch(cond: Condition, n: int, catalog: RoomCatalog) -> LayoutBatch:
    """n copies of the room and its items, positions not yet known (all zero)."""
    blank = make_layout(catalog, cond.width, cond.depth, cond.door_wall, cond.door_offset,
                        {name: (0.0, 0.0, 0) for name in cond.items}, cond.items)
    return stack_layouts([blank] * n, catalog)


def load_cvae_run(run_dir: Path, device: str | torch.device = "cpu") -> CVAE:
    """The trained CVAE of a train_cvae run, in eval() mode."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model = CVAE(CVAEConfig(**record["cvae"]))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval()


def load_evaluator_run(run_dir: Path, device: str | torch.device = "cpu") -> tuple[Evaluator, RasterConfig]:
    """The trained evaluator of a train_evaluator run, in eval() mode, and the raster settings it
    was trained with."""
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = evaluator_config_from_record(record["evaluator"])
    raster = RasterConfig(**record["raster"])
    model = Evaluator(config, pixels=raster.pixels)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), raster


class CVAESampler:
    """M1: layouts decoded from prior samples of the CVAE (BatchNorm in inference mode)."""

    name = "M1"

    def __init__(self, model: CVAE, catalog: RoomCatalog, device: str | torch.device = "cpu",
                 pins: dict[str, Pin] | None = None):
        self.model, self.catalog, self.device = model.to(device).eval(), catalog, device
        self.pins = pins or {}  # pinned items are moved onto their spots after decoding
        self.last_displacement = np.zeros(0)  # per sample: the pinned items' distance from their pins before that

    def sample(self, cond: Condition, n: int, rng: np.random.Generator) -> Samples:
        blank = condition_batch(cond, n, self.catalog)
        c = torch.as_tensor(encode_conditions(blank, self.catalog), device=self.device)
        positions, logits = self.model.generate(c, _torch_generator(rng, self.device))
        decoded = decode_targets(positions, logits, blank, self.catalog)
        layouts = [decoded.layout(i, self.catalog) for i in range(n)]
        if self.pins:
            self.last_displacement = np.array([displacement(layout, self.pins, self.catalog) for layout in layouts])
            layouts = [snap(layout, self.pins, self.catalog) for layout in layouts]
        return Samples(layouts, n)


class LatentOptSampler:
    """M2: the M1 draws, each repaired by latent optimization before it is decoded."""

    name = "M2"

    def __init__(self, model: CVAE, catalog: RoomCatalog, rules: Rules, config: LatentOptConfig,
                 device: str | torch.device = "cpu", pins: dict[str, Pin] | None = None):
        self.model, self.catalog, self.rules, self.config = model.to(device).eval(), catalog, rules, config
        self.device = device
        self.pins = pins or {}  # pulled towards their spots while optimizing, then snapped onto them
        self.last_displacement = np.zeros(0)  # per sample: the pinned items' distance from their pins before the snap

    def sample(self, cond: Condition, n: int, rng: np.random.Generator) -> Samples:
        blank = condition_batch(cond, n, self.catalog)
        problem = problem_for(cond, blank, self.catalog, self.rules, self.device, self.pins)
        z0 = torch.randn(n, self.model.config.latent, generator=_torch_generator(rng, self.device), device=self.device)
        result = optimize(self.model, z0, problem, self.config)
        positions = result.positions
        if self.pins:
            room = torch.tensor([cond.width, cond.depth], dtype=positions.dtype, device=positions.device)
            off = ((positions * room - problem.pin_center) ** 2).sum(dim=-1).sqrt()  # (n, K) meters
            self.last_displacement = ((off * problem.pin_mask).sum(dim=1) / problem.pin_mask.sum(dim=1)).cpu().numpy()
            positions = snap_pins(positions, problem)
        decoded = decode_targets(positions.clamp(0.0, 1.0), F.one_hot(result.rot, 4), blank, self.catalog)
        return Samples([decoded.layout(i, self.catalog) for i in range(n)], n)


def problem_for(cond: Condition, blank: LayoutBatch, catalog: RoomCatalog, rules: Rules,
                device: str | torch.device, pins: dict[str, Pin] | None = None) -> Problem:
    """The latent-optimization problem of a room: its condition vectors, its door zone and its pins."""
    door = door_geometry(cond.width, cond.depth, cond.door_wall, cond.door_offset, rules.door)
    n = len(blank)
    c = torch.as_tensor(encode_conditions(blank, catalog), device=device)

    def repeat(values) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device).expand(n, -1)

    if not pins:
        return Problem(c, repeat(door.zone_center), repeat(door.zone_size))
    k = catalog.num_slots
    mask, center, facing = np.zeros(k, dtype=bool), np.zeros((k, 2), dtype=np.float32), np.full(k, -1, dtype=np.int64)
    for name, pin in pins.items():
        slot = catalog.slot(name).index
        mask[slot], center[slot], facing[slot] = True, (pin.x, pin.y), -1 if pin.facing is None else pin.facing
    return Problem(c, repeat(door.zone_center), repeat(door.zone_size),
                   pin_mask=torch.as_tensor(mask, device=device).expand(n, -1),
                   pin_center=torch.as_tensor(center, device=device).expand(n, -1, -1),
                   pin_rot=torch.as_tensor(facing, device=device).expand(n, -1))


# --------------------------------------------------------------------------- the request

@dataclass(frozen=True)
class PipelineConfig:
    candidates: int = 64  # latent samples per request
    latent_opt: bool = True  # M2; False gives M1
    top_k: int = 3
    tau_div: float = 0.3  # minimum layout distance between picks (m)
    relax: float = 0.25  # share by which tau_div shrinks when too few qualify ...
    relax_times: int = 2  # ... up to this many times


def load_pipeline_config(path: Path = DEFAULT_CONFIG, **overrides) -> PipelineConfig:
    return PipelineConfig(**{**load_config(path)["pipeline"], **overrides})


@dataclass(frozen=True)
class Request:
    """What a user asks for: the room, its door, the furniture and an optional budget (INR)."""
    width: float
    depth: float
    door_wall: str
    door_offset: float
    items: dict[str, str | None]  # slot name -> variant id, or None to let the pipeline choose
    budget: float | None = None
    pins: dict[str, Pin] = field(default_factory=dict)  # slot name -> where the user fixed it


@dataclass
class Candidate:
    layout: Layout
    quality: float  # rule quality score S
    evaluator_score: float | None  # the CNN's predicted score (ranks the candidates)
    evaluator_valid: float | None  # the CNN's P(valid), shown next to the checker's verdict


@dataclass
class PipelineResult:
    request: Request
    condition: Condition | None  # the room with the variants chosen; None if turned away
    top: list[Candidate] = field(default_factory=list)
    message: str | None = None  # why nothing (or fewer than top_k) came back
    warnings: list[str] = field(default_factory=list)
    candidates: int = 0
    valid: int = 0
    tau_used: float | None = None
    seconds: dict[str, float] = field(default_factory=dict)


def check_request(request: Request, catalog: RoomCatalog, rules: Rules) -> tuple[list[str], list[str]]:
    """(errors that stop the request, warnings): unknown or missing items, a door that does not
    fit, a room outside the training ranges (out of distribution: expect lower validity)."""
    errors, warnings = [], []
    names = {slot.name for slot in catalog.slots}
    errors += [f"unknown item {name!r}" for name in request.items if name not in names]
    errors += [f"the {slot.name.replace('_', ' ')} is required" for slot in catalog.slots
               if slot.mandatory and slot.name not in request.items]
    for name, variant in request.items.items():
        if name in names and variant is not None and variant not in {v.id for v in catalog.slot(name).variants}:
            errors.append(f"unknown variant {variant!r} for the {name.replace('_', ' ')}")
    if min(request.width, request.depth) <= 0:
        errors.append("the room needs a positive width and depth")
    else:
        try:
            door_geometry(request.width, request.depth, request.door_wall, request.door_offset, rules.door)
        except ValueError as error:
            errors.append(str(error))
        room = rules.rooms[catalog.room_type]
        if not (room.width[0] <= request.width <= room.width[1] and room.depth[0] <= request.depth <= room.depth[1]):
            warnings.append(f"the room is outside the training ranges (W {room.width[0]:g} to {room.width[1]:g} m, "
                            f"D {room.depth[0]:g} to {room.depth[1]:g} m): expect lower validity")
    return errors, warnings


def choose_variants(request: Request, catalog: RoomCatalog, rules: Rules) -> tuple[Condition | None, str | None]:
    """The feasibility pre-check and the choice of variants.

    Every combination of variants for the items left open is tried; those within the budget and
    the footprint limit f_max qualify, and the one with the largest total footprint wins (a
    comfort proxy), the cheaper one on a tie. If none qualifies: None and a message with a
    suggestion."""
    options = [[catalog.slot(name).variant(v)] if v else list(catalog.slot(name).variants)
               for name, v in request.items.items()]
    names = list(request.items)
    combos = list(itertools.product(*options))
    price = [sum(v.price for v in combo) for combo in combos]
    area = [sum(v.area for v in combo) for combo in combos]
    ratio = [footprint_ratio(a, request.width, request.depth, rules.door) for a in area]
    fits = [i for i in range(len(combos)) if ratio[i] <= rules.f_max
            and (request.budget is None or price[i] <= request.budget)]
    if fits:
        best = max(fits, key=lambda i: (area[i], -price[i]))
        cond = Condition(request.width, request.depth, request.door_wall, request.door_offset,
                         {name: v.id for name, v in zip(names, combos[best])})
        return cond, None
    if request.budget is not None and min(price) > request.budget:
        return None, (f"the cheapest choice costs {min(price):,} INR, over the budget of {request.budget:,.0f} INR: "
                      "remove an optional item or raise the budget")
    optional = [n for n in names if not catalog.slot(n).mandatory]
    hint = f"remove an optional item (for example the {optional[0].replace('_', ' ')})" if optional else "use a bigger room"
    return None, (f"the furniture covers {min(ratio):.0%} of the free floor at best, above the {rules.f_max:.0%} "
                  f"that can usually be furnished: {hint} or choose a smaller variant")


def select_diverse(layouts: list[Layout], k: int, tau: float, relax: float, times: int) -> tuple[list[int], float]:
    """Greedy top k over layouts already sorted best first: a layout joins if it is at least tau
    from every earlier pick; tau shrinks by `relax` up to `times` times if too few join.
    Returns the picked positions and the tau used."""
    picks: list[int] = []
    threshold = tau
    for attempt in range(times + 1):
        threshold = tau * (1 - relax) ** attempt
        picks = []
        for i, layout in enumerate(layouts):
            if all(layout_distance(layout, layouts[j]) >= threshold for j in picks):
                picks.append(i)
                if len(picks) == k:
                    return picks, threshold
    return picks, threshold


def generate(request: Request, catalog: RoomCatalog, rules: Rules, cvae: CVAE, rng: np.random.Generator,
             evaluator: Evaluator | None = None, raster: RasterConfig | None = None,
             config: PipelineConfig = PipelineConfig(), latent: LatentOptConfig = LatentOptConfig(),
             device: str | torch.device = "cpu") -> PipelineResult:
    """The whole pipeline for one request (the six steps in the module docstring)."""
    # 1-2. Check the request and choose the variants.
    start = time.perf_counter()
    errors, warnings = check_request(request, catalog, rules)
    if errors:
        return PipelineResult(request, None, message="; ".join(errors), warnings=warnings)
    cond, message = choose_variants(request, catalog, rules)
    if cond is None:
        return PipelineResult(request, None, message=message, warnings=warnings)
    pin_problems = pin_errors(request.pins, cond, catalog, rules)
    if pin_problems:
        return PipelineResult(request, None, message="; ".join(pin_problems), warnings=warnings)
    timings = {"check": time.perf_counter() - start}

    # 3. Sample the candidates (M2: repaired by latent optimization; M1: as decoded).
    start = time.perf_counter()
    sampler = (LatentOptSampler(cvae, catalog, rules, latent, device, request.pins) if config.latent_opt
               else CVAESampler(cvae, catalog, device, request.pins))
    layouts = sampler.sample(cond, config.candidates, rng).layouts
    timings["sample"] = time.perf_counter() - start

    # 4. Keep the candidates that pass every hard check: the checker is the final authority.
    start = time.perf_counter()
    valid, quality = [], []
    for layout in layouts:
        reach = reachability(layout, catalog, rules)
        if check_layout(layout, catalog, rules, reach=reach).valid:
            valid.append(layout)
            quality.append(quality_score(layout, catalog, rules, reach=reach).total)
    timings["checks"] = time.perf_counter() - start

    # 5. Rank by the CNN's predicted score (or the rule score without an evaluator).
    start = time.perf_counter()
    scores = probabilities = [None] * len(valid)
    if evaluator is not None and valid:
        with torch.no_grad():
            logit, score = evaluator.eval()(rasterize_layouts(valid, catalog, rules, raster, device))
        scores, probabilities = score.cpu().tolist(), torch.sigmoid(logit).cpu().tolist()
    ranking = sorted(range(len(valid)), key=lambda i: -(scores[i] if scores[i] is not None else quality[i]))

    # 6. A diverse top k.
    picks, tau = select_diverse([valid[i] for i in ranking], config.top_k, config.tau_div, config.relax,
                                config.relax_times)
    top = [Candidate(valid[ranking[i]], quality[ranking[i]], scores[ranking[i]], probabilities[ranking[i]])
           for i in picks]
    timings["rank"] = time.perf_counter() - start
    note = None
    if len(top) < config.top_k:
        note = (f"only {len(top)} distinct valid layouts out of {config.candidates} candidates"
                + ("" if valid else ": remove an optional item or try another door position"))
    return PipelineResult(request, cond, top, note, warnings, len(layouts), len(valid), tau if top else None,
                          {name: round(value, 3) for name, value in timings.items()})


def _torch_generator(rng: np.random.Generator, device) -> torch.Generator:
    """A torch generator seeded from the numpy one, so a seeded run draws the same z."""
    return torch.Generator(device=device).manual_seed(int(rng.integers(2 ** 62)))


def main(argv: list[str] | None = None) -> int:
    import argparse

    from spacegen.catalog import load_room_catalog
    from spacegen.latent_opt import load_latent_opt_config
    from spacegen.layout import layout_to_dict
    from spacegen.paths import REPORTS_DIR, saved_run
    from spacegen.rules import load_rules
    from spacegen.viz import SURFACE, plot_layout

    parser = argparse.ArgumentParser(description="Generate the top layouts for one room.")
    parser.add_argument("--width", type=float, required=True, help="W in meters")
    parser.add_argument("--depth", type=float, required=True, help="D in meters")
    parser.add_argument("--door", required=True, choices=["N", "E", "S", "W"], help="wall with the door")
    parser.add_argument("--offset", type=float, default=0.5, help="door position along its wall, 0 to 1")
    parser.add_argument("--items", required=True,
                        help="comma-separated slots, each optionally name:variant, e.g. sofa:sofa_2seater,tv_unit")
    parser.add_argument("--budget", type=float, default=None, help="INR")
    frozen = saved_run("cvae", "frozen", "seed-0")
    parser.add_argument("--cvae", type=Path, default=frozen if (frozen / "model.pt").exists()
                        else saved_run("cvae", "default"), help="default: the frozen seed 0, else runs/cvae/default")
    parser.add_argument("--evaluator", type=Path, default=saved_run("evaluator", "e9a"),
                        help="trained evaluator run; rank by the rule score if it does not exist")
    parser.add_argument("--pin", action="append", default=[], metavar="ITEM:X,Y[,FACING]",
                        help="keep an item at (x, y) meters, optionally facing N, E, S or W, e.g. sofa:2.5,3.4,S")
    parser.add_argument("--no-latent-opt", action="store_true", help="M1 instead of M2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "demo")
    parser.add_argument("--device", default="cpu", help="cpu is faster than a GPU for one request")
    args = parser.parse_args(argv)

    items = {}
    for entry in args.items.split(","):  # "sofa:sofa_2seater" or just "sofa"
        name, _, variant = entry.partition(":")
        items[name] = variant or None
    pins = {}
    for entry in args.pin:  # "sofa:2.5,3.4,S"
        name, _, where = entry.partition(":")
        x, y, *facing = where.split(",")
        pins[name] = Pin(float(x), float(y), "NESW".index(facing[0].upper()) if facing else None)
    request = Request(args.width, args.depth, args.door, args.offset, items, args.budget, pins)
    catalog, rules = load_room_catalog("living_room"), load_rules()
    cvae = load_cvae_run(args.cvae, args.device)
    evaluator, raster = load_evaluator_run(args.evaluator, args.device) if args.evaluator.exists() else (None, None)
    result = generate(request, catalog, rules, cvae, np.random.default_rng(args.seed), evaluator, raster,
                      load_pipeline_config(latent_opt=not args.no_latent_opt), load_latent_opt_config(), args.device)
    for warning in result.warnings:
        print("warning:", warning)
    if result.condition is None:
        print("turned away:", result.message)
        return 1
    print(f"variants: {result.condition.items}")
    print(f"{result.valid} of {result.candidates} candidates valid; seconds {result.seconds} "
          f"(total {sum(result.seconds.values()):.2f})")
    args.out.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(result.top, start=1):
        metrics = {"rank": rank, "valid": True, "quality": round(candidate.quality, 4),
                   "evaluator_score": None if candidate.evaluator_score is None else round(candidate.evaluator_score, 4),
                   "evaluator_valid": None if candidate.evaluator_valid is None else round(candidate.evaluator_valid, 4)}
        meta = {"method": "M1" if args.no_latent_opt else "M2", "seed": args.seed, "cvae": str(args.cvae)}
        data = layout_to_dict(candidate.layout, catalog, rules.door.width, metrics, meta)
        (args.out / f"top{rank}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        title = (f"Top {rank} of {len(result.top)}: rule quality {candidate.quality:.2f}"
                 + ("" if candidate.evaluator_score is None else f", evaluator score {candidate.evaluator_score:.2f}"))
        plot_layout(candidate.layout, catalog, rules, title=title).savefig(args.out / f"top{rank}.png", facecolor=SURFACE)
        print(f"top {rank}: quality {candidate.quality:.3f}, evaluator {metrics['evaluator_score']}, "
              f"P(valid) {metrics['evaluator_valid']}")
    if result.message:
        print("note:", result.message)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
