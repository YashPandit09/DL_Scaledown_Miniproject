"""Gate 2 tests: the selection rule, the frozen config, the validation rooms and the runner."""
import numpy as np
import pandas as pd
import torch

from experiments.figures import FIGURES
from experiments.gate2 import choose, run_gate2, summarize_candidates, verdicts, write_frozen
from experiments.headline import Headline
from spacegen.dataset import load_layouts
from spacegen.evaluate import EvaluationConfig
from spacegen.generator import Condition
from spacegen.latent_opt import load_latent_opt_config
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.models.evaluator import Evaluator, EvaluatorConfig
from spacegen.pipeline import PipelineConfig
from spacegen.raster import load_raster_config
from spacegen.train_cvae import load_configs

TINY = ["cvae.hidden=16", "cvae_training.max_epochs=3", "cvae_training.anneal_epochs=1",
        "cvae_training.check_rooms=2", "cvae_training.batch=8"]


def _runs(values: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    """Per-seed rows from (M2 raw valid, M2 top-3 quality) pairs."""
    return pd.DataFrame([{"candidate": name, "seed": seed, "run": f"{name}-{seed}", "m2_rvr": rvr, "m2_top3": top}
                         for name, pairs in values.items() for seed, (rvr, top) in enumerate(pairs)])


def test_the_rule_needs_a_gain_beyond_the_spread_and_no_loss():
    summary = summarize_candidates(_runs({
        "default": [(0.60, 0.78), (0.62, 0.79), (0.61, 0.77)],
        "valid only": [(0.70, 0.78), (0.71, 0.77), (0.69, 0.79)],  # clearly more valid, same quality
        "trade-off": [(0.50, 0.85), (0.51, 0.86), (0.49, 0.84)],  # better quality, clearly less valid
        "within noise": [(0.615, 0.785), (0.625, 0.775), (0.605, 0.795)],
    }))
    found = verdicts(summary)
    assert [name for name, (ok, _, _) in found.items() if ok] == ["valid only"]
    assert found["trade-off"][2].startswith("does not qualify")
    assert choose(summary) == "valid only"
    better = summarize_candidates(_runs({"default": [(0.60, 0.78), (0.62, 0.79), (0.61, 0.77)],
                                         "a": [(0.70, 0.78), (0.71, 0.79), (0.69, 0.79)],
                                         "b": [(0.62, 0.84), (0.61, 0.85), (0.62, 0.86)]}))
    assert choose(better) == "b"  # both qualify; b gains more top-3 quality
    assert choose(summarize_candidates(_runs({"default": [(0.6, 0.8), (0.61, 0.8)]}))) == "default"


def test_frozen_config_is_the_default_with_the_overrides(tmp_path):
    overrides = ["cvae.activation=tanh", "cvae.lambda_overlap=1.0"]
    target = write_frozen(overrides, "a test.", target=tmp_path / "frozen.yaml")
    assert load_configs(target) == load_configs(overrides=overrides)
    text = target.read_text(encoding="utf-8")
    assert "was relu" in text and text.startswith("# Frozen configuration")
    from spacegen.config import load_config
    assert load_config(target)["latent_opt"]["lambda_overlap"] == 10.0  # the same field in another section stays
    assert load_configs(write_frozen([], "unchanged.", target=tmp_path / "same.yaml")) == load_configs()


def test_validation_rooms_come_from_the_validation_split(tiny_dataset, tmp_path, catalog, rules):
    headline = Headline(tiny_dataset, tmp_path, 0, EvaluationConfig(rooms=3, samples=2), CVAE(CVAEConfig(hidden=16)),
                        catalog, rules, load_latent_opt_config(steps=2), PipelineConfig())
    rooms, reference = headline.rooms("validation")
    set_a = load_layouts(tiny_dataset / "set_a.npz")
    with np.load(tiny_dataset / "splits.npz") as splits:
        allowed = [Condition.of(set_a.layout(int(i), catalog), catalog) for i in splits["set_a_validation"]]
    assert len(rooms) == min(3, len(allowed)) and all(room in allowed for room in rooms) and reference.empty


def test_runner_trains_each_seed_once_and_scores_m1_and_m2(tiny_dataset, tmp_path):
    torch.manual_seed(0)
    raster = load_raster_config()
    evaluator = Evaluator(EvaluatorConfig(channels=(4, 4, 4, 4), hidden=8), pixels=raster.pixels).eval()
    log = []
    options = dict(candidates={"default": [], "elu": ["cvae.activation=elu"]}, seeds=2, rooms=2, samples=3, base=TINY,
                   screen_dir=tmp_path / "screen", gate2_dir=tmp_path / "gate2", cache_dir=tmp_path / "cache",
                   evaluator=evaluator, raster=raster, log=log.append)
    runs = run_gate2(tiny_dataset, 0, "cpu", **options)
    assert sum("training" in line for line in log) == 4
    assert list(zip(runs["candidate"], runs["seed"])) == [("default", 0), ("default", 1), ("elu", 0), ("elu", 1)]
    assert runs["run"].str.startswith("screen").tolist() == [True, False, True, False]  # the first seed is the screening run
    assert runs[["m1_rvr", "m2_rvr"]].stack().between(0, 1).all() and runs["m2_top3"].notna().any()
    log.clear()
    again = run_gate2(tiny_dataset, 0, "cpu", **options)
    assert not any("training" in line or "rooms in" in line for line in log)  # trained and scored once
    pd.testing.assert_frame_equal(again, runs)
    summary = summarize_candidates(runs)
    assert summary["seeds"].tolist() == [2, 2] and {"m2_rvr_mean", "m2_rvr_std"} <= set(summary.columns)
    figures = FIGURES["gate2"](runs, None)
    assert all(fig.axes for fig in figures.values())
    skipped = run_gate2(tiny_dataset, 0, "cpu", **{**options, "seeds": 3}, train=False)
    assert len(skipped) == 4 and any("not trained" in line for line in log)
