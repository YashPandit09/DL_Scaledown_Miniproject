"""How good is the CNN evaluator? Its metrics on the Set B test split (experiment E9a).

python run.py evaluator-report               the model in runs/evaluator/e9a/
python run.py evaluator-report --run NAME

Writes reports/tables/evaluator.csv (one row of overall metrics), evaluator_per_type.csv and
reports/figures/evaluator_confusion.png.

Overall, "valid" is the positive class: accuracy, precision, recall, F1 and ROC-AUC on P(valid),
plus the F1 of the invalid class. Per perturbation type, F1 takes "invalid" as the positive
class, because spotting violations is the evaluator's job and every perturbed type contains
invalid layouts; the clean type (all valid) reports its false-alarm rate, the share of valid
layouts called invalid. Spearman compares the predicted score with the rule score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from spacegen.catalog import load_room_catalog
from spacegen.config import load_config
from spacegen.models.evaluator import Evaluator
from spacegen.paths import DATA_DIR, REPORTS_DIR, RUNS_DIR
from spacegen.perturb import type_order
from spacegen.raster import RasterConfig
from spacegen.rules import load_rules
from spacegen.train_evaluator import EvaluatorData, load_evaluator_data


@torch.no_grad()
def predict(model: Evaluator, data: EvaluatorData, rows: np.ndarray, raster: RasterConfig,
            batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """P(valid) and the predicted quality score for `rows`, in order."""
    model.eval()
    probabilities, scores = [], []
    for index in torch.as_tensor(rows, device=data.valid.device).split(batch):
        logit, score = model(data.inputs.rasterize(index, raster))
        probabilities.append(torch.sigmoid(logit).cpu())
        scores.append(score.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(scores).numpy()


def overall_metrics(valid: np.ndarray, probability: np.ndarray, score: np.ndarray, quality: np.ndarray) -> dict:
    """Valid is the positive class; F1 of the invalid class and Spearman with the rule score too."""
    predicted = probability > 0.5
    (tn, fp), (fn, tp) = confusion_matrix(valid, predicted, labels=[False, True])
    return {"samples": len(valid), "valid_share": valid.mean(), "accuracy": (predicted == valid).mean(),
            "precision": precision_score(valid, predicted, zero_division=0),
            "recall": recall_score(valid, predicted, zero_division=0), "f1": f1_score(valid, predicted, zero_division=0),
            "f1_invalid": f1_score(~valid, ~predicted, zero_division=0), "roc_auc": roc_auc_score(valid, probability),
            "spearman_score": spearmanr(score, quality).statistic,
            "true_valid": int(tp), "false_valid": int(fp), "true_invalid": int(tn), "false_invalid": int(fn)}


def per_type(types: np.ndarray, valid: np.ndarray, predicted_valid: np.ndarray) -> pd.DataFrame:
    """Per perturbation type: samples, valid share, accuracy, and F1 for spotting invalid layouts
    (or, where every layout is valid, the false-alarm rate)."""
    rows = []
    for kind in sorted(set(types), key=type_order):
        here = types == kind
        truth, guess = valid[here], predicted_valid[here]
        invalid_present = (~truth).any()
        rows.append({"perturbation": kind, "samples": int(here.sum()), "valid_share": truth.mean(),
                     "accuracy": (truth == guess).mean(),
                     "f1_invalid": f1_score(~truth, ~guess, zero_division=0) if invalid_present else np.nan,
                     "false_alarm_rate": (~guess[truth]).mean() if truth.all() else np.nan})
    return pd.DataFrame(rows)


def report(run_dir: Path, data_dir: Path, tables_dir: Path, figures_dir: Path, device: str, log=print) -> dict:
    """Score the run's model on the Set B test split and write the tables and the figure."""
    from spacegen.pipeline import load_evaluator_run
    from spacegen.viz import SURFACE, plot_confusion

    model, raster = load_evaluator_run(run_dir, device)
    catalog, rules = load_room_catalog("living_room"), load_rules()
    data, rows = load_evaluator_data(data_dir, catalog, rules, device)
    test = rows["test"]
    probability, score = predict(model, data, test, raster)
    info = pd.read_csv(data_dir / "set_b.csv").iloc[test]
    valid = info["valid"].to_numpy(dtype=bool)
    overall = {"run": run_dir.name, **overall_metrics(valid, probability, score, info["quality"].to_numpy())}
    types = per_type(info["perturbation"].to_numpy(), valid, probability > 0.5)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([overall]).round(4).to_csv(tables_dir / "evaluator.csv", index=False, lineterminator="\n")
    types.round(4).to_csv(tables_dir / "evaluator_per_type.csv", index=False, lineterminator="\n")
    plot_confusion(overall, run_dir.name).savefig(figures_dir / "evaluator_confusion.png", facecolor=SURFACE)
    log(pd.DataFrame([overall]).round(3).T.to_string(header=False))
    log(types.round(3).to_string(index=False))
    return {"overall": overall, "per_type": types}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluator metrics on the Set B test split (E9a).")
    parser.add_argument("--run", default="e9a", help="run folder under runs/evaluator/")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    data_dir = args.data or DATA_DIR / load_config()["dataset"]["version"]
    report(RUNS_DIR / "evaluator" / args.run, data_dir, REPORTS_DIR / "tables", REPORTS_DIR / "figures", args.device)
    print("wrote reports/tables/evaluator.csv, evaluator_per_type.csv and reports/figures/evaluator_confusion.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
