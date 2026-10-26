"""Training the CVAE on Set A.

python run.py train-cvae                                    the default configuration
python run.py train-cvae --name e5-beta1 --set cvae_training.beta_target=1.0 --set cvae.latent=4
python run.py train-cvae --config configs/frozen.yaml --name frozen/seed-0 --seed 0

One epoch: shuffle the training layouts, and for each mini-batch of 256 run the CVAE, compute the
loss, backpropagate and take an optimizer step (Adam by default). Then score the validation split.

- KL annealing: the KL weight beta rises linearly from 0 in epoch 1 to beta_target in epoch
  anneal_epochs + 1. Starting at 0 lets the decoder learn to use z before the KL term pushes the
  encoder towards the prior; starting at full weight risks posterior collapse.
- The validation loss is always computed at beta_target and with the same draw of z in every
  epoch, so the epochs compare fairly.
- Early stopping: training stops when the validation loss has not improved for `patience` epochs,
  and the best epoch's weights are kept. It only starts once annealing is over: while beta still
  rises, the training objective changes and the loss can climb, and a stopper running then could
  keep a model trained almost without the KL term.
- log.csv records, per epoch: the loss terms, the validation position error in meters, the
  gradient norm of every linear layer (to see vanishing gradients), the share of dead units in
  every hidden layer (to see dying ReLU) and the number of active latent units (to see
  posterior collapse).

A run writes runs/cvae/<name>/: run.json (seed, configs, git commit, dataset hash), log.csv,
model.pt (the selected epoch's weights) and summary.json, which ends with a first raw valid rate
of M1 on `check_rooms` test rooms.
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
import yaml
from torch import nn

from spacegen import geometry
from spacegen.catalog import RoomCatalog, load_room_catalog
from spacegen.config import DEFAULT_CONFIG, load_config
from spacegen.dataset import (LayoutBatch, TrainingTensors, load_layouts, minibatches, split_targets, training_tensors,
                              unpack_conditions)
from spacegen.evaluate import evaluate
from spacegen.generator import Condition
from spacegen.models.cvae import CVAE, CVAEConfig, CVAEOutput, active_units, cvae_loss
from spacegen.paths import DATA_DIR, RUNS_DIR
from spacegen.pipeline import CVAESampler
from spacegen.provenance import write_run_record
from spacegen.rules import load_rules
from spacegen.seed import set_seed
from spacegen.splits import SPLITS

OPTIMIZERS = ("adam", "sgd", "sgd_momentum", "rmsprop")  # compared in experiment E4
TRAIN_TERMS = ("loss", "position", "rotation", "kl", "overlap")  # loss terms logged per epoch
_ACTIVATION_TYPES = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU, nn.Tanh, nn.Sigmoid)


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adam"  # a name in OPTIMIZERS
    lr: float = 1e-3
    momentum: float = 0.9  # for sgd_momentum
    batch: int = 256
    max_epochs: int = 200
    beta_target: float = 0.1  # the final KL weight
    anneal_epochs: int = 20  # epochs over which beta rises from 0 to beta_target
    patience: int = 15  # epochs without improvement before early stopping
    early_stopping: bool = True  # False trains for max_epochs (E6)
    check_rooms: int = 100  # test rooms of the M1 check after training (0: no check)
    outliers: float = 0.0  # E2: share of training layouts with one item moved to a random spot

    def __post_init__(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")


def load_configs(path: Path = DEFAULT_CONFIG, overrides: list[str] = ()) -> tuple[CVAEConfig, TrainingConfig]:
    """The cvae and cvae_training sections of a config file, with overrides such as
    "cvae.activation=elu" (how the experiments change one setting at a time)."""
    raw = load_config(path)
    sections = {"cvae": dict(raw["cvae"]), "cvae_training": dict(raw["cvae_training"])}
    for override in overrides:
        key, _, value = override.partition("=")
        section, _, field = key.partition(".")
        if section not in sections or field not in sections[section] or not value:
            raise ValueError(f"override {override!r} must look like cvae.<field>=<value> or cvae_training.<field>=<value>")
        sections[section][field] = yaml.safe_load(value)
    return CVAEConfig(**sections["cvae"]), TrainingConfig(**sections["cvae_training"])


# --------------------------------------------------------------------------- the building blocks

def beta_at(epoch: int, training: TrainingConfig) -> float:
    """KL annealing: the KL weight in a (1-based) epoch, 0 at epoch 1, rising linearly to
    beta_target at epoch anneal_epochs + 1 and staying there."""
    if training.anneal_epochs <= 0:
        return training.beta_target
    return training.beta_target * min(1.0, (epoch - 1) / training.anneal_epochs)


@dataclass
class EarlyStopping:
    """Patience on the validation loss, counted only from epoch `start` (the first one after annealing)."""
    patience: int
    start: int
    best: float = float("inf")
    best_epoch: int | None = None

    def update(self, epoch: int, loss: float) -> tuple[bool, bool]:
        """(this epoch is the new best, stop now). Epochs before `start` are neither."""
        if epoch < self.start:
            return False, False
        if loss < self.best:
            self.best, self.best_epoch = loss, epoch
            return True, False
        return False, epoch - self.best_epoch >= self.patience


def make_optimizer(parameters, training: TrainingConfig) -> torch.optim.Optimizer:
    """Adam, RMSProp, or SGD with or without momentum (experiment E4)."""
    if training.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=training.lr)
    if training.optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=training.lr)
    if training.optimizer == "sgd_momentum":
        return torch.optim.SGD(parameters, lr=training.lr, momentum=training.momentum)
    return torch.optim.SGD(parameters, lr=training.lr)


def train_one_epoch(model: CVAE, optimizer: torch.optim.Optimizer, train: TrainingTensors, batch_size: int,
                    beta: float, shuffle: torch.Generator,
                    linear: list[tuple[str, nn.Linear]]) -> tuple[dict[str, float], dict[str, float]]:
    """One pass over the training split in shuffled mini-batches. Returns the mean of each loss
    term and the mean gradient norm of each linear layer over the epoch's steps."""
    model.train()  # dropout on; BatchNorm normalizes with each batch's statistics
    device = train.x.device
    term_sums = torch.zeros(len(TRAIN_TERMS), device=device)
    grad_sums = torch.zeros(len(linear), device=device)
    steps = 0
    # drop_last: a final batch of a single layout would break BatchNorm
    for index in minibatches(len(train), batch_size, shuffle, device, drop_last=True):
        batch = train[index]
        loss = cvae_loss(model(batch.x, batch.c), batch, beta, model.config)
        optimizer.zero_grad()
        loss.total.backward()
        grad_sums += torch.stack([layer.weight.grad.norm() for _, layer in linear])
        optimizer.step()
        term_sums += torch.stack([loss.total, loss.position, loss.rotation, loss.kl, loss.overlap]).detach()
        steps += 1
    terms = dict(zip(TRAIN_TERMS, (term_sums / steps).tolist()))
    grad_norms = {name: value / steps for (name, _), value in zip(linear, grad_sums.tolist())}
    return terms, grad_norms


def layer_names(model: CVAE) -> tuple[list[tuple[str, nn.Linear]], list[tuple[str, nn.Module]]]:
    """Readable names for the linear layers (for the gradient norms) and the activation layers
    (for the dead units): encoder_1, encoder_2, to_mu, to_logvar, decoder_1, ..., head_rot."""
    linear, activations = [], []
    for stack in ("encoder", "decoder"):
        layers = getattr(model, stack)
        linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        activation_layers = [layer for layer in layers if isinstance(layer, _ACTIVATION_TYPES)]
        linear += [(f"{stack}_{i}", layer) for i, layer in enumerate(linear_layers, start=1)]
        activations += [(f"{stack}_{i}", layer) for i, layer in enumerate(activation_layers, start=1)]
        if stack == "encoder":
            linear += [("to_mu", model.to_mu), ("to_logvar", model.to_logvar)]
    linear += [("head_pos", model.head_pos), ("head_rot", model.head_rot)]
    return linear, activations


@torch.no_grad()
def validate(model: CVAE, data: TrainingTensors, eps: torch.Tensor, beta: float) -> dict[str, float]:
    """The loss terms at `beta` with the fixed noise `eps`, the position error in meters, the
    active latent units, and the share of dead units in each hidden layer."""
    model.eval()  # dropout off; BatchNorm uses its running statistics
    _, activations = layer_names(model)
    outputs: dict[str, torch.Tensor] = {}  # each activation layer's output, recorded by a forward hook

    def recorder(name: str):
        def hook(module, inputs, output):
            outputs[name] = output
        return hook

    hooks = [module.register_forward_hook(recorder(name)) for name, module in activations]
    try:
        mu, logvar = model.encode(data.x, data.c)
        z = mu + torch.exp(0.5 * logvar) * eps  # the reparameterization, with the fixed eps
        positions, logits = model.decode(z, data.c)
    finally:
        for hook in hooks:
            hook.remove()
    loss = cvae_loss(CVAEOutput(positions, logits, mu, logvar, z), data, beta, model.config)
    error = position_errors(positions, data, model.config.num_slots)
    # A unit is dead if it outputs 0 for every validation sample (a "dying ReLU" never recovers).
    dead = {f"dead_{name}": float((out.abs() <= 1e-8).all(dim=0).float().mean()) for name, out in outputs.items()}
    return {"loss": loss.total.item(), "recon": loss.recon.item(), "position": loss.position.item(),
            "rotation": loss.rotation.item(), "kl": loss.kl.item(), "position_m": float(error.mean()),
            "active_units": active_units(mu, logvar), **dead}


def _copy_state(model: CVAE) -> dict[str, torch.Tensor]:
    """A copy of the model's weights on the CPU (the "best so far" checkpoint)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- training

def train_cvae(data_dir: Path, out_dir: Path, seed: int, device: str | torch.device,
               model_config: CVAEConfig | None = None, training: TrainingConfig | None = None,
               catalog: RoomCatalog | None = None, log=print, config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Train on the Set A training split and write the run folder; returns the per-epoch log.
    `config_path` is only recorded in run.json."""
    set_seed(seed)
    if model_config is None or training is None:
        defaults = load_configs()
        model_config, training = model_config or defaults[0], training or defaults[1]
    catalog = catalog or load_room_catalog("living_room")
    write_run_record(out_dir, seed, data_dir, task="CVAE training", device=str(device), config=str(config_path),
                     cvae=dataclasses.asdict(model_config), cvae_training=dataclasses.asdict(training))

    # The data: Set A's training and validation splits, encoded once and kept on the device.
    set_a = load_layouts(data_dir / "set_a.npz")
    with np.load(data_dir / "splits.npz") as splits:
        rows = {part: splits[f"set_a_{part}"] for part in SPLITS}
    train_layouts, outlier_rows = set_a.subset(rows["train"]), np.array([], dtype=int)
    if training.outliers > 0:  # E2: only the training split changes; validation and test stay clean
        train_layouts, outlier_rows = with_outliers(train_layouts, training.outliers, np.random.default_rng(seed),
                                                    catalog)
    train = training_tensors(train_layouts, catalog, device)
    val = training_tensors(set_a.subset(rows["validation"]), catalog, device)

    model = CVAE(model_config).to(device)
    optimizer = make_optimizer(model.parameters(), training)
    val_eps = torch.randn(len(val), model_config.latent, generator=torch.Generator().manual_seed(seed)).to(device)
    shuffle = torch.Generator().manual_seed(seed)  # the order of the mini-batches
    stopper = EarlyStopping(training.patience, start=training.anneal_epochs + 1)
    linear, _ = layer_names(model)
    history, best_state = [], None
    for epoch in range(1, training.max_epochs + 1):
        beta = beta_at(epoch, training)
        start = time.perf_counter()
        train_terms, grad_norms = train_one_epoch(model, optimizer, train, training.batch, beta, shuffle, linear)
        seconds = time.perf_counter() - start
        validation = validate(model, val, val_eps, training.beta_target)
        improved, stop = stopper.update(epoch, validation["loss"])
        if improved:
            best_state = _copy_state(model)

        row = {"epoch": epoch, "beta": beta, "seconds": round(seconds, 2)}
        row.update({f"train_{name}": value for name, value in train_terms.items()})
        row.update({f"val_{name}": value for name, value in validation.items()})
        row["best"] = improved
        row.update({f"grad_{name}": value for name, value in grad_norms.items()})
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or improved or stop:
            log(f"epoch {epoch}: beta {beta:.3f}; train {train_terms['loss']:.4f}; validation {validation['loss']:.4f} "
                f"(position {validation['position']:.4f}, rotation {validation['rotation']:.3f}, KL {validation['kl']:.2f}); "
                f"active units {validation['active_units']}{'; best' if improved else ''}")
        if stop and training.early_stopping:
            break

    selected = stopper.best_epoch if training.early_stopping and stopper.best_epoch else len(history)
    if selected == len(history):  # fixed epochs (E6), or no epoch after annealing: keep the final model
        best_state = _copy_state(model)
    table = pd.DataFrame(history)
    table.to_csv(out_dir / "log.csv", index=False, lineterminator="\n")
    torch.save(best_state, out_dir / "model.pt")
    model.load_state_dict(best_state)

    chosen = table.loc[table["epoch"] == selected].iloc[0]
    summary = {"epochs_run": len(history), "stopped_early": len(history) < training.max_epochs,
               "selected_epoch": int(selected), "selected_validation_loss": float(chosen["val_loss"]),
               "selected_position_loss": float(chosen["val_position"]),
               "selected_rotation_loss": float(chosen["val_rotation"]), "selected_kl": float(chosen["val_kl"]),
               "active_units": int(chosen["val_active_units"]),
               "parameters": sum(p.numel() for p in model.parameters()), "outlier_layouts": len(outlier_rows),
               "train_eval_loss": train_eval_loss(model, set_a.subset(rows["train"]), seed, training.beta_target,
                                                  catalog, device),
               **reconstruction_errors(model, val)}
    if training.check_rooms > 0:
        check = m1_check(model, set_a, rows["test"], training.check_rooms, seed, catalog, device)
        summary["m1_check"] = check
        quality = "none valid" if check["quality"] is None else f"{check['quality']:.3f}"
        log(f"M1 on {check['rooms']} test rooms x 64 samples: raw valid {check['rvr']:.1%}, "
            f"quality of valid layouts {quality}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    return table


# --------------------------------------------------------------------------- measurements after training

def position_errors(positions: torch.Tensor, data: TrainingTensors, num_slots: int) -> np.ndarray:
    """Distance in meters between predicted and true centres, one value per present item."""
    target, _ = split_targets(data.x)
    room, _, _ = unpack_conditions(data.c, num_slots)
    distance = ((positions - target) * room[:, None, :]).norm(dim=-1)
    return distance[data.mask > 0.5].detach().cpu().numpy()


@torch.no_grad()
def reconstruction_errors(model: CVAE, data: TrainingTensors) -> dict[str, float]:
    """Position error in meters with z = mu (E2), overall and split at 0.05 from the walls (E7)."""
    model.eval()
    out = model(data.x, data.c, sample=False)
    target, _ = split_targets(data.x)
    present = data.mask > 0.5
    near = ((target < 0.05) | (target > 0.95)).any(dim=-1)[present].cpu().numpy()
    error = position_errors(out.positions.clamp(0.0, 1.0), data, model.config.num_slots)
    return {"position_error_median": float(np.median(error)), "position_error_mean": float(error.mean()),
            "position_error_near_walls": float(error[near].mean()) if near.any() else None,
            "position_error_rest": float(error[~near].mean())}


def train_eval_loss(model: CVAE, train: LayoutBatch, seed: int, beta: float, catalog: RoomCatalog,
                    device) -> float:
    """The loss on the (clean) training split, scored like the validation split: eval mode,
    beta_target and a fixed draw of z. Compared with the validation loss it gives the
    generalization gap (E6). (The training loss in the log is measured in training mode, with
    dropout and fresh noise, so it is higher.)"""
    data = training_tensors(train, catalog, device)
    eps = torch.randn(len(data), model.config.latent, generator=torch.Generator().manual_seed(seed)).to(device)
    return validate(model, data, eps, beta)["loss"]


def m1_check(model: CVAE, set_a, test_rows: np.ndarray, n_rooms: int, seed: int, catalog: RoomCatalog,
             device) -> dict:
    """A first raw valid rate of M1: 64 prior samples for each of n_rooms Set A test rooms, with the
    diversity of the valid samples (mean over the rooms with at least two)."""
    rng = np.random.default_rng(seed)
    rooms = [Condition.of(set_a.layout(int(row), catalog), catalog)
             for row in rng.choice(test_rows, size=min(n_rooms, len(test_rows)), replace=False)]
    table = evaluate(CVAESampler(model, catalog, device), rooms, 64, rng, catalog, load_rules())
    valid = int(table["valid"].sum())
    return {"rooms": len(rooms), "samples": int(table["attempts"].sum()), "rvr": valid / int(table["attempts"].sum()),
            "quality": float(table["quality"].sum() / valid) if valid else None,
            "mean_overlap": float(table["overlap"].sum() / table["returned"].sum()),
            "diversity": float(table["diversity"].dropna().astype(float).mean()) if table["diversity"].notna().any()
            else None}


def with_outliers(batch: LayoutBatch, share: float, rng: np.random.Generator,
                  catalog: RoomCatalog) -> tuple[LayoutBatch, np.ndarray]:
    """E2's outlier variant: in a share of the layouts, one present item moves to a uniformly
    random spot where it still fits in the room. Returns the batch and the changed rows."""
    rows = np.sort(rng.choice(len(batch), size=round(share * len(batch)), replace=False))
    center = batch.center.copy()
    half = geometry.effective_size(batch.sizes(catalog), batch.rot) / 2
    for row in rows:
        k = rng.choice(np.flatnonzero(batch.mask[row]))
        center[row, k] = rng.uniform(half[row, k], batch.room[row] - half[row, k])
    return dataclasses.replace(batch, center=center), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CVAE on Set A.")
    parser.add_argument("--name", default="default", help="run folder under runs/cvae/ (may contain a /)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="configuration file, e.g. configs/frozen.yaml for the final runs")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config value, e.g. cvae.activation=elu (repeatable)")
    parser.add_argument("--seed", type=int, default=None, help="default: the seed in configs/default.yaml")
    parser.add_argument("--data", type=Path, default=None, help="default: data/<version from configs/default.yaml>")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    raw = load_config(args.config)
    model_config, training = load_configs(args.config, overrides=args.overrides)
    data_dir = args.data or DATA_DIR / raw["dataset"]["version"]
    out_dir = RUNS_DIR / "cvae" / args.name
    seed = raw["seed"] if args.seed is None else args.seed
    train_cvae(data_dir, out_dir, seed, args.device, model_config, training, config_path=args.config)
    print(json.dumps(json.loads((out_dir / "summary.json").read_text(encoding="utf-8")), indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
