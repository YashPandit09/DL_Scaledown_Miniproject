"""CVAE training tests: annealing, early stopping, logging, provenance."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch

from spacegen.generator import Condition, sample_condition, load_generator_config
from spacegen.layout import is_canonical
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.pipeline import CVAESampler
from spacegen.train_cvae import (EarlyStopping, TrainingConfig, beta_at, layer_names, load_configs, make_optimizer,
                                 train_cvae, validate)

SMALL = CVAEConfig(hidden=32, latent=4)


def test_beta_rises_linearly_over_the_annealing_epochs():
    training = TrainingConfig(beta_target=0.1, anneal_epochs=20)
    assert [beta_at(e, training) for e in (1, 11, 21, 50)] == pytest.approx([0.0, 0.05, 0.1, 0.1])
    assert beta_at(1, dataclasses.replace(training, anneal_epochs=0)) == 0.1


def test_early_stopping_cannot_fire_during_annealing():
    stopper = EarlyStopping(patience=2, start=21)
    rising = [stopper.update(epoch, 1.0 + 0.1 * epoch) for epoch in range(1, 21)]  # the loss climbs while beta rises
    assert rising == [(False, False)] * 20 and stopper.best_epoch is None
    assert stopper.update(21, 9.0) == (True, False)  # the first epoch after annealing is the first best
    assert stopper.update(22, 9.5) == (False, False)
    assert stopper.update(23, 8.0) == (True, False)
    assert stopper.update(24, 8.5) == (False, False)
    assert stopper.update(25, 8.1) == (False, True)  # two epochs without a better loss
    assert stopper.best_epoch == 23


def test_config_overrides():
    model, training = load_configs(overrides=["cvae.activation=elu", "cvae_training.beta_target=0.5",
                                              "cvae.batch_norm=false"])
    assert (model.activation, model.batch_norm, training.beta_target) == ("elu", False, 0.5)
    assert load_configs() == (CVAEConfig(), TrainingConfig())
    for bad in ("cvae.colour=red", "cvae_training.beta_target", "trainer.lr=1"):
        with pytest.raises(ValueError):
            load_configs(overrides=[bad])


@pytest.mark.parametrize("name, kind, momentum", [("adam", torch.optim.Adam, None), ("sgd", torch.optim.SGD, 0.0),
                                                  ("sgd_momentum", torch.optim.SGD, 0.9),
                                                  ("rmsprop", torch.optim.RMSprop, None)])
def test_optimizers_for_e4(name, kind, momentum):
    optimizer = make_optimizer(CVAE(SMALL).parameters(), TrainingConfig(optimizer=name))
    assert isinstance(optimizer, kind)
    if momentum is not None:
        assert optimizer.param_groups[0]["momentum"] == momentum


def test_layer_names():
    linear, activations = layer_names(CVAE(SMALL))
    assert [name for name, _ in linear] == ["encoder_1", "encoder_2", "to_mu", "to_logvar", "decoder_1", "decoder_2",
                                            "head_pos", "head_rot"]
    assert [name for name, _ in activations] == ["encoder_1", "encoder_2", "decoder_1", "decoder_2"]
    deep = CVAE(dataclasses.replace(SMALL, depth=6, batch_norm=False))
    assert len(layer_names(deep)[1]) == 12


def test_validation_counts_dead_units(tiny_dataset, catalog):
    from spacegen.dataset import load_layouts, training_tensors
    data = training_tensors(load_layouts(tiny_dataset / "set_a.npz"), catalog)
    model = CVAE(SMALL)
    with torch.no_grad():
        model.encoder[0].bias.fill_(-1e3)  # every unit of the first hidden layer silent
    result = validate(model, data, torch.zeros(len(data), SMALL.latent), beta=0.1)
    assert result["dead_encoder_1"] == 1.0 and result["dead_decoder_1"] < 1.0
    assert set(result) >= {"loss", "position", "rotation", "kl", "active_units"}


@pytest.fixture(scope="module")
def run(tiny_dataset, catalog, tmp_path_factory):
    out = tmp_path_factory.mktemp("cvae_run")
    training = TrainingConfig(batch=8, max_epochs=30, anneal_epochs=5, patience=1, check_rooms=2)
    log = train_cvae(tiny_dataset, out, 0, "cpu", SMALL, training, catalog, log=lambda message: None)
    return out, log, training


def test_training_runs_past_annealing_and_logs_every_epoch(run):
    out, log, training = run
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(log) >= training.anneal_epochs + 1  # patience 1 still cannot stop it during annealing
    assert summary["selected_epoch"] >= training.anneal_epochs + 1
    assert not log.loc[log["epoch"] <= training.anneal_epochs, "best"].any()
    assert {"beta", "train_loss", "val_loss", "val_active_units", "grad_encoder_1", "grad_head_rot",
            "val_dead_decoder_2"} <= set(log.columns)
    assert np.isfinite(log.filter(like="grad_").to_numpy()).all()
    assert 0 <= summary["m1_check"]["rvr"] <= 1 and summary["m1_check"]["samples"] == 2 * 64
    assert summary["train_eval_loss"] > 0 and "diversity" in summary["m1_check"]


def test_a_run_records_where_it_came_from(run, tiny_dataset):
    out, log, _ = run
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    metadata = json.loads((tiny_dataset / "metadata.json").read_text(encoding="utf-8"))
    assert record["dataset_hash"] == metadata["hash"] and record["cvae"]["hidden"] == 32
    assert record["config"].endswith("default.yaml")  # the configuration file is recorded
    assert pd.read_csv(out / "log.csv").shape[0] == len(log)
    state = torch.load(out / "model.pt", weights_only=True)
    model = CVAE(SMALL)
    model.load_state_dict(state)  # the saved checkpoint fits the model


def test_training_is_reproducible(tiny_dataset, catalog, tmp_path):
    training = TrainingConfig(batch=8, max_epochs=3, anneal_epochs=1, patience=5, check_rooms=0)
    first, second = (train_cvae(tiny_dataset, tmp_path / name, 0, "cpu", SMALL, training, catalog,
                                log=lambda message: None) for name in ("a", "b"))
    pd.testing.assert_frame_equal(first.drop(columns="seconds"), second.drop(columns="seconds"))


def test_m1_sampler_returns_layouts_of_the_requested_room(catalog, rules):
    cond = sample_condition(np.random.default_rng(0), catalog, rules, load_generator_config())
    sampler = CVAESampler(CVAE(SMALL), catalog)
    samples = sampler.sample(cond, 5, np.random.default_rng(1))
    assert samples.attempts == 5 and len(samples.layouts) == 5
    for layout in samples.layouts:
        assert Condition.of(layout, catalog) == cond and is_canonical(layout, catalog)
        present = layout.center[layout.mask]
        assert (present >= 0).all() and (present <= layout.room).all()  # centres clamped into the room
    again = sampler.sample(cond, 5, np.random.default_rng(1))
    assert all(np.array_equal(a.center, b.center) for a, b in zip(samples.layouts, again.layouts))
