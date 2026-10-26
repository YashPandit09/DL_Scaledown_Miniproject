"""Evaluator tests: architecture, loss, Set B on the device, a falling loss."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from spacegen.dataset import load_layouts
from spacegen.models.evaluator import (Evaluator, EvaluatorConfig, evaluator_config_from_record, evaluator_loss,
                                       load_evaluator_config)
from spacegen.paths import CHECKPOINTS_DIR
from spacegen.raster import RasterConfig, rasterize_layouts
from spacegen.seed import set_seed
from spacegen.train_evaluator import load_evaluator_data, run_epoch, train_evaluator

SMALL = RasterConfig(canvas=8.0, pixels=32, front_strip=0.10)  # 0.25 m pixels keep the CPU tests fast


@pytest.fixture(scope="module")
def set_b(tiny_dataset, catalog, rules):
    return load_evaluator_data(tiny_dataset, catalog, rules, "cpu")


def test_config_file_holds_the_spec_values():
    assert load_evaluator_config() == EvaluatorConfig()


def test_architecture_matches_the_spec():
    set_seed(0)
    model = Evaluator()
    valid, score = model(torch.rand(3, 4, 128, 128))
    assert valid.shape == score.shape == (3,)
    assert sum(p.numel() for p in model.parameters()) == 585_682  # about 0.59 M parameters
    assert model.dense[1].in_features == 64 * 8 * 8  # four halvings: 128 px -> 8 px
    assert Evaluator(pixels=32).dense[1].in_features == 64 * 2 * 2
    with pytest.raises(ValueError, match="halved"):
        Evaluator(pixels=100)


def test_loss_is_bce_on_the_logit_plus_the_score_regression():
    logit, score = torch.tensor([2.0, -1.0, 0.5]), torch.tensor([0.8, 0.2, 0.6])
    valid, quality = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.9, 0.3, 0.4])
    loss = evaluator_loss(logit, score, valid, quality, EvaluatorConfig(lambda_score=2.0))
    bce = F.binary_cross_entropy(torch.sigmoid(logit), valid)
    torch.testing.assert_close(loss.bce, bce)
    torch.testing.assert_close(loss.total, bce + 2.0 * ((score - quality) ** 2).mean())


def test_the_saved_evaluator_record_gives_the_current_config():
    """The committed evaluator was trained when the config also had three options no experiment
    changed (score_loss, huber_delta, pos_weight); reading its record skips them."""
    record = json.loads((CHECKPOINTS_DIR / "evaluator" / "e9a" / "run.json").read_text(encoding="utf-8"))
    assert {"score_loss", "huber_delta", "pos_weight"} <= set(record["evaluator"])
    assert evaluator_config_from_record(record["evaluator"]) == EvaluatorConfig()


def test_set_b_on_the_device_matches_its_files(set_b, tiny_dataset, catalog, rules):
    data, rows = set_b
    info = pd.read_csv(tiny_dataset / "set_b.csv")
    assert len(data) == 40 and sorted(np.concatenate(list(rows.values())).tolist()) == list(range(40))
    np.testing.assert_array_equal(data.valid.numpy(), info["valid"].to_numpy(dtype=np.float32))
    batch = load_layouts(tiny_dataset / "set_b.npz")
    expected = rasterize_layouts([batch.layout(i, catalog) for i in (0, 7)], catalog, rules, SMALL)
    torch.testing.assert_close(data.inputs.rasterize(torch.tensor([0, 7]), SMALL), expected)


def test_training_lowers_the_loss(set_b):
    """Training runs and the loss falls. On 32 layouts the CNN gets past its
    first plateau (predicting the share of valid layouts) and then learns them."""
    data, _ = set_b
    rows = np.arange(32)
    config = EvaluatorConfig(batch=16)
    set_seed(0)
    model = Evaluator(config, pixels=SMALL.pixels)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    before = run_epoch(model, data, rows, SMALL, config)
    shuffle = torch.Generator().manual_seed(0)
    for _ in range(40):
        run_epoch(model, data, rows, SMALL, config, optimizer, shuffle)
    after = run_epoch(model, data, rows, SMALL, config)
    assert after["loss"] < 0.25 * before["loss"] and after["bce"] < 0.5 * before["bce"]
    assert after["accuracy"] >= 0.9


def test_training_is_reproducible(set_b):
    data, _ = set_b
    config = EvaluatorConfig(batch=16)

    def trained():
        set_seed(0)
        model = Evaluator(config, pixels=SMALL.pixels)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        run_epoch(model, data, np.arange(32), SMALL, config, optimizer, torch.Generator().manual_seed(0))
        return model

    for a, b in zip(trained().parameters(), trained().parameters()):
        assert torch.equal(a, b)


def test_a_training_run_records_where_it_came_from(tiny_dataset, catalog, rules, tmp_path):
    config = dataclasses.replace(EvaluatorConfig(), batch=8)
    log = train_evaluator(tiny_dataset, tmp_path / "run", epochs=1, seed=0, device="cpu", config=config, raster=SMALL,
                          catalog=catalog, rules=rules, log=lambda message: None)
    assert log["epoch"].tolist() == [0, 1] and np.isfinite(log["train_loss"][1])
    record = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    metadata = json.loads((tiny_dataset / "metadata.json").read_text(encoding="utf-8"))
    assert record["dataset_hash"] == metadata["hash"] and record["seed"] == 0
    assert set(record["configs"]) == {"default.yaml", "rules.yaml", "catalog.yaml"}
    assert (tmp_path / "run" / "model.pt").exists() and (tmp_path / "run" / "log.csv").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_a_training_step_on_the_gpu(tiny_dataset, catalog, rules):
    data, rows = load_evaluator_data(tiny_dataset, catalog, rules, "cuda")
    config = EvaluatorConfig(batch=8)
    model = Evaluator(config, pixels=SMALL.pixels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    result = run_epoch(model, data, rows["train"], SMALL, config, optimizer)
    assert np.isfinite(result["loss"]) and next(model.parameters()).is_cuda
