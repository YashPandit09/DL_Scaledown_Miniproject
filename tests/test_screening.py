"""Screening tests: run naming, the experiment grid, the outlier variant, the runner and figures."""
import numpy as np
import pandas as pd
import pytest

from experiments.figures import FIGURES, e2_losses
from experiments.screening import EXPERIMENTS, effective, run_experiment, run_latent_steps, run_name
from spacegen import geometry
from spacegen.dataset import load_layouts
from spacegen.train_cvae import with_outliers

TINY = ["cvae.hidden=16", "cvae_training.max_epochs=3", "cvae_training.anneal_epochs=1",
        "cvae_training.check_rooms=2", "cvae_training.batch=8"]


def test_the_same_settings_share_one_run():
    assert run_name([]) == "default" == run_name(["cvae.activation=relu", "cvae.depth=2"])  # both are the defaults
    assert run_name(["cvae.depth=6", "cvae.activation=elu"]) == run_name(["cvae.activation=elu", "cvae.depth=6"])
    assert effective(["cvae.batch_norm=true", "cvae.batch_norm=false"]) == ["cvae.batch_norm=false"]


def test_the_experiment_grid_matches_the_plan():
    assert {name: len(settings) for name, settings in EXPERIMENTS.items()} == {
        "e2": 11, "e3a": 6, "e3b": 14, "e4": 9, "e5": 6, "e6": 5, "e7": 2}
    assert len({run_name(o) for settings in EXPERIMENTS.values() for _, o in settings}) == 48  # shared runs count once
    assert all("cvae_training.max_epochs=60" in o for _, o in EXPERIMENTS["e3b"])  # gradients exist at epoch 50
    assert sum("cvae_training.outliers=0.04" in o for _, o in EXPERIMENTS["e2"]) == 5


def test_outlier_variant_moves_one_item_in_a_share_of_layouts(tiny_dataset, catalog):
    batch = load_layouts(tiny_dataset / "set_a.npz")
    changed, rows = with_outliers(batch, 0.25, np.random.default_rng(0), catalog)
    assert len(rows) == 10
    moved = (np.abs(changed.center - batch.center).max(axis=2) > 0)
    assert moved[rows].sum(axis=1).tolist() == [1] * 10 and moved.sum() == 10  # one item each, nothing else
    eff = geometry.effective_size(batch.sizes(catalog), batch.rot)
    assert geometry.inside_room(changed.center, eff, changed.room)[batch.mask].all()


def test_runner_trains_each_setting_once_and_collects_a_row_each(tiny_dataset, tmp_path):
    trained = []
    settings = [("relu", []), ("also the default", ["cvae.activation=relu"]), ("elu", ["cvae.activation=elu"])]
    table = run_experiment("test", tiny_dataset, 0, "cpu", tmp_path, settings, TINY, log=trained.append)
    assert len(trained) == 2  # the first two settings are the same configuration
    assert table["run"].nunique() == 2 and table["setting"].tolist() == ["relu", "also the default", "elu"]
    assert {"val_loss", "position_error_mean", "m1_rvr", "dead_share", "active_units"} <= set(table.columns)
    again = run_experiment("test", tiny_dataset, 0, "cpu", tmp_path, settings, TINY, log=trained.append)
    assert len(trained) == 2 and again.equals(table)  # finished runs are reused


def test_every_figure_draws_from_a_table(tiny_dataset, tmp_path):
    settings = [("mse", []), ("mae, outliers", ["cvae.position_loss=mae", "cvae_training.outliers=0.04"]),
                ("relu, depth 6", ["cvae.depth=3", "cvae.batch_norm=false"])]
    table = run_experiment("test", tiny_dataset, 0, "cpu", tmp_path, settings, TINY, log=lambda m: None)
    steps = run_latent_steps(tiny_dataset, 0, "cpu", steps=(0, 2), rooms=2, model_dir=tmp_path / run_name(TINY),
                             log=lambda m: None)
    assert steps["steps"].tolist() == [0, 2] and steps["rvr"].between(0, 1).all()
    for name in [*EXPERIMENTS, "e8"]:  # the headline figures (E1, E10) are tested in test_headline.py
        figures = FIGURES[name](steps if name == "e8" else table, tmp_path)
        assert figures and all(fig.axes for fig in figures.values())
    assert len(e2_losses().axes[0].lines) == 5


def test_old_runs_get_the_fields_added_later(tiny_dataset, tmp_path):
    import json

    settings = [("default", [])]
    table = run_experiment("test", tiny_dataset, 0, "cpu", tmp_path, settings, TINY, log=lambda m: None)
    path = tmp_path / run_name(TINY) / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    del summary["train_eval_loss"], summary["m1_check"]["diversity"]  # as if trained before they existed
    path.write_text(json.dumps(summary), encoding="utf-8")
    again = run_experiment("test", tiny_dataset, 0, "cpu", tmp_path, settings, TINY, log=lambda m: None)
    pd.testing.assert_frame_equal(again, table)  # the same values come back, the raw valid rate unchanged
