"""Evaluator report tests (E9a): metric definitions, per-type F1 and the written outputs."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from spacegen.evaluator_report import overall_metrics, per_type, report
from spacegen.models.evaluator import EvaluatorConfig
from spacegen.raster import RasterConfig
from spacegen.train_evaluator import train_evaluator
from spacegen.viz import plot_confusion


def test_overall_metrics_take_valid_as_the_positive_class():
    valid = np.array([True, True, True, False, False, False])
    probability = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1])  # one valid missed, one invalid let through
    score, quality = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), np.array([0.8, 0.9, 0.6, 0.4, 0.1, 0.2])
    metrics = overall_metrics(valid, probability, score, quality)
    assert (metrics["true_valid"], metrics["false_invalid"], metrics["false_valid"], metrics["true_invalid"]) == (2, 1, 1, 2)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3) and metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(f1_score(valid, probability > 0.5))
    assert metrics["f1_invalid"] == pytest.approx(f1_score(~valid, probability <= 0.5))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(valid, probability))
    assert metrics["spearman_score"] == pytest.approx(0.8857, abs=1e-4)


def test_per_type_f1_spots_invalid_layouts_and_clean_reports_false_alarms():
    types = np.array(["clean"] * 4 + ["overlap"] * 3 + ["near_miss_door"] * 4)
    valid = np.array([True] * 4 + [False] * 3 + [True, True, False, False])
    predicted = np.array([True, True, True, False] + [False, False, True] + [True, False, False, False])
    table = per_type(types, valid, predicted).set_index("perturbation")
    assert table.index.tolist() == ["clean", "overlap", "near_miss_door"]  # the Set B type order
    assert np.isnan(table.loc["clean", "f1_invalid"]) and table.loc["clean", "false_alarm_rate"] == 0.25
    assert table.loc["overlap", "f1_invalid"] == pytest.approx(f1_score([1, 1, 1], [1, 1, 0]))
    assert np.isnan(table.loc["overlap", "false_alarm_rate"])
    assert table.loc["near_miss_door", "accuracy"] == 0.75


def test_report_writes_the_tables_and_the_figure(tiny_dataset, catalog, rules, tmp_path):
    run = tmp_path / "runs" / "tiny"
    small = RasterConfig(canvas=8.0, pixels=32, front_strip=0.10)
    train_evaluator(tiny_dataset, run, epochs=1, seed=0, device="cpu", config=dataclasses.replace(EvaluatorConfig(), batch=8),
                    raster=small, catalog=catalog, rules=rules, log=lambda message: None)
    result = report(run, tiny_dataset, tmp_path / "tables", tmp_path / "figures", "cpu", log=lambda message: None)
    overall = pd.read_csv(tmp_path / "tables" / "evaluator.csv")
    assert overall.loc[0, "run"] == "tiny" and overall.loc[0, "samples"] == 6  # 15% of the 40 Set B layouts
    assert len(pd.read_csv(tmp_path / "tables" / "evaluator_per_type.csv")) == result["per_type"].shape[0]
    assert (tmp_path / "figures" / "evaluator_confusion.png").read_bytes()[:4] == b"\x89PNG"


def test_confusion_figure_shows_counts_and_shares():
    metrics = {"true_valid": 50, "false_invalid": 10, "false_valid": 5, "true_invalid": 35, "accuracy": 0.85,
               "f1": 0.87, "f1_invalid": 0.82, "roc_auc": 0.9}
    texts = [t.get_text() for t in plot_confusion(metrics, "x").axes[0].texts]
    assert texts == ["50\n83.3%", "10\n16.7%", "5\n12.5%", "35\n87.5%"]
