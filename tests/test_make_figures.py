"""Figure regeneration tests: figures from saved tables, curves from the runs or their copied logs."""
import pandas as pd
import pytest

import experiments.figures
from experiments.make_figures import copy_logs, rebuild_tables, redraw
from experiments.screening import run_experiment

TINY = ["cvae.hidden=16", "cvae_training.max_epochs=3", "cvae_training.anneal_epochs=1",
        "cvae_training.check_rooms=2", "cvae_training.batch=8"]
SETTINGS = [("mse", []), ("mae, outliers", ["cvae.position_loss=mae", "cvae_training.outliers=0.04"]),
            ("relu, depth 6", ["cvae.depth=3", "cvae.batch_norm=false"])]
CURVES = {"e2_curves.png", "e3b_gradients.png", "e4_curves.png"}


@pytest.fixture
def saved(tiny_dataset, tmp_path, monkeypatch):
    """Three tiny screening runs, their table saved as three experiments, and one evaluator row."""
    runs, tables = tmp_path / "runs", tmp_path / "tables"
    table = run_experiment("test", tiny_dataset, 0, "cpu", runs, SETTINGS, TINY, log=lambda message: None)
    tables.mkdir()
    for name in ("e2", "e3b", "e4"):
        table.to_csv(tables / f"{name}.csv", index=False)
    pd.DataFrame([{"run": "tiny", "accuracy": 0.8, "f1": 0.83, "f1_invalid": 0.75, "roc_auc": 0.9, "true_valid": 5,
                   "false_valid": 1, "true_invalid": 3, "false_invalid": 1}]).to_csv(tables / "evaluator.csv", index=False)
    monkeypatch.setattr(experiments.figures, "LOGS_DIR", tmp_path / "copies")
    return runs, tables, table


def test_every_saved_table_gets_its_figures(saved, tiny_dataset, tmp_path):
    runs, tables, _ = saved
    notes = []
    written = redraw(tables, tmp_path / "figures", runs, tiny_dataset, log=notes.append)
    names = {path.name for path in written}
    assert CURVES | {"e2_results.png", "e2_losses.png", "e3b_units.png", "e4_epochs.png", "evaluator_confusion.png",
                     "dataset_tiny_rooms.png", "dataset_tiny_f_max.png"} <= names
    assert all(path.exists() and path.stat().st_size > 1000 for path in written)
    assert any(note.startswith("e1: no table") for note in notes)  # tables that do not exist are skipped, and said so


def test_curves_come_from_the_copied_logs_when_the_runs_are_gone(saved, tmp_path):
    runs, tables, table = saved
    notes = []
    without = redraw(tables, tmp_path / "none", None, None, log=notes.append)
    needs_logs = {"e2_curves.png", "e3b_gradients.png"}  # these settings include no optimizer run for e4_curves
    assert not needs_logs & {path.name for path in without} and sum("skipped" in note for note in notes) >= 3
    copies = copy_logs(tables, runs, experiments.figures.LOGS_DIR)
    assert {path.stem for path in copies} == set(table["run"])
    again = redraw(tables, tmp_path / "copied", None, None, log=lambda message: None)
    assert CURVES <= {path.name for path in again}


def test_tables_are_rebuilt_only_from_saved_runs(saved, tiny_dataset, tmp_path):
    runs, _, table = saved
    collected = run_experiment("test", tiny_dataset, 0, "cpu", runs, SETTINGS, TINY, log=lambda message: None,
                               train=False)
    pd.testing.assert_frame_equal(collected, table)
    with pytest.raises(FileNotFoundError):  # the real E7 runs are not in this folder, and nothing may be trained
        rebuild_tables(tiny_dataset, 0, tmp_path / "rebuilt", runs, experiments=("e7",), log=lambda message: None)
