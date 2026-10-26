"""What the Streamlit app computes, kept apart from the user interface so that it can be tested.

load_models()       the CVAE, the evaluator and B2, with a note for each one that is missing
condition_for()     request checks and variant choice: the room to furnish, or why not
run_request()       the pipeline for one request (spacegen.pipeline.generate)
describe()          what a planner sees for a layout: quality and its terms, cost, floor use and
                    each hard check
compare_methods()   B1, B2, G0, M1 and M2 on the same room: valid samples, quality, time, and
                    the best layout of each
layout_png(), layout_json()    the exports
result_figures()    the saved figures for the "Training and results" tab

The models come from runs/cvae/frozen/seed-0 (or runs/cvae/default before the freeze) and
runs/evaluator/e9a, or from their committed copies in checkpoints/ when runs/ does not have them
(a fresh clone); the environment variables SPACEGEN_CVAE_RUN, SPACEGEN_EVALUATOR_RUN and
SPACEGEN_DATA_DIR point the app elsewhere (the tests use them).
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from spacegen.baselines import StatisticalBaseline, UniformBaseline, load_baseline_config
from spacegen.catalog import RoomCatalog
from spacegen.config import load_config
from spacegen.dataset import load_layouts
from spacegen.evaluate import generator_sampler
from spacegen.generator import Condition
from spacegen.latent_opt import load_latent_opt_config
from spacegen.layout import Layout, layout_to_dict
from spacegen.metrics import score_layouts
from spacegen.models.cvae import CVAE
from spacegen.models.evaluator import Evaluator
from spacegen.paths import DATA_DIR, REPORTS_DIR, saved_run
from spacegen.pins import Pin, PinnedSampler, pin_errors
from spacegen.pipeline import (CVAESampler, LatentOptSampler, PipelineResult, Request, check_request, choose_variants,
                               generate, load_cvae_run, load_evaluator_run, load_pipeline_config)
from spacegen.quality import furniture_share, quality_score
from spacegen.raster import RasterConfig
from spacegen.rules import Rules, check_layout, reachability
from spacegen.viz import SURFACE, plot_layout

CHECKS = {"in_room": "H1 inside the room", "no_overlap": "H2 no overlap", "door_clear": "H3 door zone clear",
          "reachable": "H4 every item reachable"}
METHODS = ("B1 uniform", "B2 statistical", "G0 generator", "M1 CVAE", "M2 CVAE + latent optimization")
E1_COLUMNS = {"method": "method", "seeds": "seeds", "rvr": "raw valid (%)", "rvr_std": "± over seeds",
              "quality": "quality", "quality_top3": "top 3: evaluator", "quality_top3_rule": "top 3: rule score",
              "quality_top3_random": "top 3: random", "ms_per_valid": "ms per valid"}  # what the app shows of E1
FIGURE_TITLES = {
    "e1_final_results": "E1: every method on the same test rooms (frozen configuration, three seeds)",
    "e1_final_ranking": "E1: quality of the top 3 under three rankings (frozen configuration)",
    "e8_final_steps": "E8: latent-optimization steps (frozen configuration)",
    "e10_final_generalization": "E10: rooms beyond the training distribution (frozen configuration)",
    "gate2_candidates": "Gate 2: the candidates for the frozen configuration",
    "e2_losses": "E2: position losses and their gradients", "e2_results": "E2: position loss, clean data and outliers",
    "e2_curves": "E2: position error during training", "e3a_results": "E3a: hidden activation with BatchNorm",
    "e3b_gradients": "E3b: gradient norms per layer without BatchNorm",
    "e3b_units": "E3b: dead units and loss without BatchNorm", "e4_curves": "E4: optimizers, validation loss",
    "e4_epochs": "E4: epochs to the target loss", "e5_tradeoff": "E5: KL weight and latent size",
    "e6_gap": "E6: regularization and the generalization gap", "e7_walls": "E7: position head near walls",
    "e8_steps": "E8: latent-optimization steps (default configuration)",
    "e1_results": "E1, first pass (default configuration)", "e1_ranking": "E1, first pass: ranking the top 3",
    "e10_generalization": "E10, first pass (default configuration)",
    "evaluator_confusion": "E9a: the evaluator's confusion matrix",
    "dataset_v1_rooms": "Dataset: rooms and held-out regions", "dataset_v1_set_b": "Dataset: Set B labels",
    "dataset_v1_rejection": "Dataset: rejected placement attempts", "dataset_v1_f_max": "Dataset: calibration of f_max",
    "generator_rejection": "Generator: rejected attempts by room area and item count",
}


@dataclass
class Models:
    cvae: CVAE | None = None
    cvae_run: Path | None = None
    evaluator: Evaluator | None = None
    raster: RasterConfig | None = None
    b2: StatisticalBaseline | None = None  # fitted on the Set A training split
    notes: list[str] = field(default_factory=list)  # what is missing, and the command that makes it


def default_runs() -> tuple[Path, Path, Path]:
    """(CVAE run, evaluator run, dataset folder), each overridable by its environment variable."""
    frozen = saved_run("cvae", "frozen", "seed-0")
    cvae = frozen if (frozen / "model.pt").exists() else saved_run("cvae", "default")
    data = DATA_DIR / load_config()["dataset"]["version"]
    return (Path(os.environ.get("SPACEGEN_CVAE_RUN", cvae)), Path(os.environ.get("SPACEGEN_EVALUATOR_RUN",
            saved_run("evaluator", "e9a"))), Path(os.environ.get("SPACEGEN_DATA_DIR", data)))


def load_models(catalog: RoomCatalog, cvae_run: Path | None = None, evaluator_run: Path | None = None,
                data_dir: Path | None = None) -> Models:
    """Load what exists; every missing part gets a note instead of an error, so the app still opens."""
    defaults = default_runs()
    cvae_run, evaluator_run, data_dir = cvae_run or defaults[0], evaluator_run or defaults[1], data_dir or defaults[2]
    models = Models()
    if (cvae_run / "model.pt").exists():
        models.cvae, models.cvae_run = load_cvae_run(cvae_run), cvae_run
    else:
        models.notes.append(f"No trained CVAE in {cvae_run}: run `python run.py train-cvae` "
                            "(the layouts and M1, M2 need it).")
    if (evaluator_run / "model.pt").exists():
        models.evaluator, models.raster = load_evaluator_run(evaluator_run)
    else:
        models.notes.append(f"No trained evaluator in {evaluator_run}: run `python run.py train-evaluator --epochs 30 "
                            "--name e9a`. Until then layouts are ranked by the rule score.")
    if (data_dir / "set_a.npz").exists():
        set_a = load_layouts(data_dir / "set_a.npz")
        with np.load(data_dir / "splits.npz") as splits:
            models.b2 = StatisticalBaseline(catalog, load_baseline_config()).fit(set_a.subset(splits["set_a_train"]))
    else:
        models.notes.append(f"No dataset in {data_dir}: run `python run.py data` (the comparison's B2 is fitted on it).")
    return models


def condition_for(request: Request, catalog: RoomCatalog, rules: Rules) -> tuple[Condition | None, str | None, list[str]]:
    """(the room with its variants chosen, or None; the reason when None; warnings)."""
    errors, warnings = check_request(request, catalog, rules)
    if errors:
        return None, "; ".join(errors), warnings
    cond, message = choose_variants(request, catalog, rules)
    if cond is not None and request.pins:
        problems = pin_errors(request.pins, cond, catalog, rules)
        if problems:
            return None, "; ".join(problems), warnings
    return cond, message, warnings


def run_request(models: Models, request: Request, seed: int, candidates: int, latent_opt: bool,
                catalog: RoomCatalog, rules: Rules) -> PipelineResult:
    """The pipeline on the CPU (faster than the GPU for one request)."""
    if models.cvae is None:
        raise RuntimeError("no trained CVAE is loaded")
    config = load_pipeline_config(candidates=candidates, latent_opt=latent_opt)
    return generate(request, catalog, rules, models.cvae, np.random.default_rng(seed), models.evaluator,
                    models.raster, config, load_latent_opt_config(), "cpu")


def describe(layout: Layout, catalog: RoomCatalog, rules: Rules) -> dict:
    """Quality and its four terms, cost, floor use and the result of each hard check."""
    reach = reachability(layout, catalog, rules)
    result = check_layout(layout, catalog, rules, reach=reach)
    quality = quality_score(layout, catalog, rules, reach=reach)
    cost = sum(catalog.slot(int(k)).variant(layout.variant_ids[k]).price for k in np.flatnonzero(layout.mask))
    return {"valid": bool(result.valid),
            "checks": {label: bool(getattr(result, name)) for name, label in CHECKS.items()},
            "quality": float(quality.total),
            "terms": {"alignment": float(quality.align), "relations": float(quality.relations),
                      "circulation": float(quality.circulation), "space": float(quality.space)},
            "cost": int(cost), "floor_use": furniture_share(layout)}


def compare_methods(models: Models, cond: Condition, n: int, seed: int, catalog: RoomCatalog, rules: Rules,
                    pins: dict[str, Pin] | None = None) -> tuple[pd.DataFrame, dict[str, Layout | None]]:
    """Every method samples the same room n times: one table row per method, and the valid layout
    with the best rule score of each. M1 and M2 start from the same draws. G0 runs single
    attempts, so its raw valid rate is its acceptance per attempt. With `pins` every method keeps
    the pinned items on their spots (spacegen.pins): G0 becomes G0-pin."""
    latent = load_latent_opt_config()
    pins = pins or {}
    samplers = [UniformBaseline(catalog), models.b2, generator_sampler(catalog, rules),
                None if models.cvae is None else CVAESampler(models.cvae, catalog, pins=pins),
                None if models.cvae is None else LatentOptSampler(models.cvae, catalog, rules, latent, pins=pins)]
    if pins:
        samplers[:3] = [None if base is None else PinnedSampler(base, pins, catalog, takes_pins=index < 2)
                        for index, base in enumerate(samplers[:3])]
    rows, best = [], {}
    for index, (name, sampler) in enumerate(zip(METHODS, samplers)):
        if sampler is None:
            rows.append({"method": name, "note": "not available (see the notes at the top)"})
            best[name] = None
            continue
        rng = np.random.default_rng([seed, 3 if index >= 3 else index])  # M1 and M2 share their draws
        start = time.perf_counter()
        samples = sampler.sample(cond, n, rng)
        scores = score_layouts(samples.layouts, catalog, rules)
        seconds = time.perf_counter() - start
        valid = scores["valid"].to_numpy(dtype=bool)
        quality = scores["quality"].to_numpy(dtype=float)
        best[name] = samples.layouts[int(np.flatnonzero(valid)[np.argmax(quality[valid])])] if valid.any() else None
        rows.append({"method": name, "raw samples": samples.attempts, "valid": int(valid.sum()),
                     "raw valid (%)": 100 * valid.sum() / samples.attempts,
                     "mean quality (valid)": float(quality[valid].mean()) if valid.any() else np.nan,
                     "best quality": float(quality[valid].max()) if valid.any() else np.nan,
                     "seconds": seconds, "note": ""})
    return pd.DataFrame(rows), best


def layout_png(layout: Layout, catalog: RoomCatalog, rules: Rules, title: str | None = None,
               width: float | None = None) -> bytes:
    """The floor plan as a PNG: the full picture for an export, or with `width` (inches) the small
    plan the app shows in a column (spacegen.viz.plot_layout)."""
    buffer = io.BytesIO()
    plot_layout(layout, catalog, rules, title=title, width=width).savefig(buffer, format="png", facecolor=SURFACE)
    return buffer.getvalue()


def layout_json(layout: Layout, catalog: RoomCatalog, rules: Rules, metrics: dict | None = None,
                meta: dict | None = None) -> str:
    return json.dumps(layout_to_dict(layout, catalog, rules.door.width, metrics, meta), indent=2)


def headline_table(table: pd.DataFrame) -> pd.DataFrame:
    """An E1 table as the app shows it: the columns a viewer needs under readable names, with the
    raw valid rate in percent. The first-pass table has no seeds, so those columns are left out."""
    shown = table[[column for column in E1_COLUMNS if column in table.columns]].copy()
    for column in shown.columns.intersection(["rvr", "rvr_std"]):
        shown[column] = 100 * shown[column]
    return shown.rename(columns=E1_COLUMNS)


def result_figures(figures_dir: Path = REPORTS_DIR / "figures") -> list[tuple[str, Path]]:
    """(title, file) of every saved figure, the headline ones first."""
    files = {path.stem: path for path in sorted(figures_dir.glob("*.png"))}
    listed = [(title, files.pop(stem)) for stem, title in FIGURE_TITLES.items() if stem in files]
    return listed + [(stem.replace("_", " "), path) for stem, path in files.items()]
