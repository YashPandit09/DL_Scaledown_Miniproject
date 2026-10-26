"""Training the CNN evaluator on Set B.

python run.py train-evaluator                       one epoch on dataset v1 (a quick check)
python run.py train-evaluator --epochs 30 --name e9a    the evaluator used everywhere (experiment E9a)

Set B (60,000 labelled layouts) stays on the device as layout tensors, and each mini-batch is
turned into images on the fly (spacegen/raster.py): the 60,000 images at 4 x 128 x 128 would
need about 16 GB. Adam minimizes BCE (valid or not) + MSE (quality score). The validation split
is scored before the first epoch and after every epoch, and the epoch with the lowest validation
loss is kept.

A run writes runs/evaluator/<name>/: run.json (seed, configs, git commit, dataset hash), log.csv
(one row per epoch), model.pt (the best epoch) and summary.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import load_config
from spacegen.dataset import load_layouts, minibatches
from spacegen.models.evaluator import Evaluator, EvaluatorConfig, evaluator_loss, load_evaluator_config
from spacegen.paths import DATA_DIR, RUNS_DIR
from spacegen.provenance import write_run_record
from spacegen.raster import RasterConfig, RasterInputs, load_raster_config, raster_inputs
from spacegen.rules import Rules, load_rules
from spacegen.seed import set_seed
from spacegen.splits import SPLITS


@dataclass(frozen=True)
class EvaluatorData:
    inputs: RasterInputs  # Set B layouts, as tensors on the training device
    valid: torch.Tensor  # (N,) 1.0 where the layout passes the hard checks H1 to H4
    quality: torch.Tensor  # (N,) the rule quality score

    def __len__(self) -> int:
        return len(self.valid)


def load_evaluator_data(data_dir: Path, catalog: RoomCatalog, rules: Rules,
                        device: str | torch.device) -> tuple[EvaluatorData, dict[str, np.ndarray]]:
    """Set B on `device`, with its labels, and the rows of its train, validation and test splits."""
    info = pd.read_csv(data_dir / "set_b.csv")
    with np.load(data_dir / "splits.npz") as splits:
        rows = {part: splits[f"set_b_{part}"] for part in SPLITS}
    labels = {name: torch.as_tensor(info[name].to_numpy(dtype=np.float32), device=device)
              for name in ("valid", "quality")}
    return EvaluatorData(raster_inputs(load_layouts(data_dir / "set_b.npz"), catalog, rules, device), **labels), rows


def run_epoch(model: Evaluator, data: EvaluatorData, rows: np.ndarray, raster: RasterConfig,
              config: EvaluatorConfig, optimizer: torch.optim.Optimizer | None = None,
              generator: torch.Generator | None = None) -> dict[str, float]:
    """One pass over `rows`. With an optimizer it trains on shuffled mini-batches (dropping the
    last, short one, for BatchNorm); without, it only scores them, in order. Returns the mean
    loss terms and the accuracy of the validity head (logit > 0 means "valid")."""
    training = optimizer is not None
    model.train(training)
    device = data.valid.device
    rows = torch.as_tensor(rows, device=device)
    if training:
        batches = (rows[i] for i in minibatches(len(rows), config.batch, generator, device, drop_last=True))
    else:
        batches = rows.split(config.batch)
    sums = {"loss": 0.0, "bce": 0.0, "score": 0.0, "accuracy": 0.0}
    seen = 0
    with torch.set_grad_enabled(training):
        for index in batches:
            logit, score = model(data.inputs.rasterize(index, raster))  # images made on the fly
            valid = data.valid[index]
            loss = evaluator_loss(logit, score, valid, data.quality[index], config)
            if training:
                optimizer.zero_grad()
                loss.total.backward()
                optimizer.step()
            n = len(index)
            seen += n
            sums["loss"] += loss.total.item() * n
            sums["bce"] += loss.bce.item() * n
            sums["score"] += loss.score.item() * n
            sums["accuracy"] += ((logit > 0) == (valid > 0.5)).sum().item()
    return {name: value / seen for name, value in sums.items()}


def train_evaluator(data_dir: Path, out_dir: Path, epochs: int, seed: int, device: str | torch.device,
                    config: EvaluatorConfig | None = None, raster: RasterConfig | None = None,
                    catalog: RoomCatalog | None = None, rules: Rules | None = None, log=print) -> pd.DataFrame:
    """Train for `epochs` epochs; returns the log (row 0: the untrained model on validation)."""
    set_seed(seed)
    config, raster = config or load_evaluator_config(), raster or load_raster_config()
    catalog, rules = catalog or load_room_catalog("living_room"), rules or load_rules()
    write_run_record(out_dir, seed, data_dir, task="evaluator training", device=str(device), epochs=epochs,
                     evaluator=dataclasses.asdict(config), raster=dataclasses.asdict(raster))
    data, rows = load_evaluator_data(data_dir, catalog, rules, device)
    model = Evaluator(config, pixels=raster.pixels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    shuffle = torch.Generator().manual_seed(seed)  # the order of the mini-batches

    def scored(prefix: str, values: dict) -> dict:
        return {f"{prefix}_{name}": value for name, value in values.items()}

    untrained = run_epoch(model, data, rows["validation"], raster, config)
    history = [{"epoch": 0, "seconds": 0.0, **scored("validation", untrained)}]
    log(f"epoch 0 (untrained): validation loss {history[0]['validation_loss']:.4f}")
    best_loss, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train = run_epoch(model, data, rows["train"], raster, config, optimizer, shuffle)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()  # the GPU works asynchronously: wait for it before reading the clock
        seconds = time.perf_counter() - start
        validation = run_epoch(model, data, rows["validation"], raster, config)
        improved = validation["loss"] < best_loss
        if improved:  # keep the best epoch: single epochs can spike (BatchNorm running statistics)
            best_loss, best_epoch = validation["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "seconds": round(seconds, 1), **scored("train", train),
                        **scored("validation", validation), "best": improved})
        log(f"epoch {epoch}: {seconds:.0f} s; train loss {train['loss']:.4f}; validation loss "
            f"{validation['loss']:.4f}, accuracy {validation['accuracy']:.3f}{'; best' if improved else ''}")
    table = pd.DataFrame(history)
    table.to_csv(out_dir / "log.csv", index=False, lineterminator="\n")
    torch.save(best_state if best_state is not None else model.state_dict(), out_dir / "model.pt")
    summary = {"epochs": epochs, "selected_epoch": best_epoch, "selected_validation_loss": best_loss,
               "parameters": sum(p.numel() for p in model.parameters())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CNN evaluator on Set B.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--name", default="t20-smoke", help="run folder under runs/evaluator/")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    raw = load_config()
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    out_dir = RUNS_DIR / "evaluator" / args.name
    table = train_evaluator(data_dir, out_dir, args.epochs, raw["seed"], args.device)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
