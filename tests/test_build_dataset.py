"""Dataset build tests: a tiny build is complete, reproducible, leak-free, and f_max is fitted."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from spacegen.build_dataset import (build_dataset, dataset_hash, file_hashes, fit_f_max, load_dataset_config,
                                    make_figures)
from spacegen.dataset import load_layouts, save_layouts, stack_layouts
from spacegen.generator import Condition, load_generator_config
from spacegen.layout import is_canonical
from spacegen.perturb import load_set_b_config
from spacegen.splits import leaks, load_split_config

SIZES = dataclasses.replace(load_dataset_config(), version="tiny", set_a=30, set_b=30, calibration_rooms=60)
SPLIT = dataclasses.replace(load_split_config(), held_out_rooms=4, diversity_rooms=3, diversity_layouts=2)


def _build(directory, catalog, rules, figures=None):
    return build_dataset(directory, 5, SIZES, catalog, rules, load_generator_config(), load_set_b_config(), SPLIT,
                         figures_dir=figures, log=lambda message: None)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, catalog, rules):
    directory = tmp_path_factory.mktemp("tiny")
    return directory, _build(directory, catalog, rules)


def test_every_file_is_written_and_counted(tiny):
    directory, metadata = tiny
    names = {path.name for path in directory.iterdir()}
    for stem in ("set_a", "set_b", "held_out_interpolation", "held_out_unseen_combination", "held_out_out_of_range",
                 "diversity_reference"):
        assert {f"{stem}.npz", f"{stem}.csv"} <= names
    assert {"set_a_attempts.csv", "splits.npz", "calibration.csv", "metadata.json"} <= names
    assert metadata["counts"]["set_a"] == 30 and metadata["counts"]["set_b"] == 30
    assert metadata["counts"]["set_a_train"] == 21 and metadata["counts"]["set_b_test"] == 4
    saved = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert saved["hash"] == metadata["hash"] == dataset_hash(file_hashes(directory))
    assert "default.yaml" in saved["configs"] and saved["seed"] == 5


def test_a_rebuild_gives_the_same_hash(tiny, tmp_path, catalog, rules):
    directory, metadata = tiny
    again = _build(tmp_path, catalog, rules)
    assert again["files"] == metadata["files"] and again["hash"] == metadata["hash"]


def test_saved_sets_are_canonical_and_leak_free(tiny, catalog):
    directory, _ = tiny
    for name in ("set_a", "set_b"):
        batch = load_layouts(directory / f"{name}.npz")
        assert leaks(batch, SPLIT) == {region: 0 for region in SPLIT.held_out}
        assert all(is_canonical(batch.layout(i, catalog), catalog) for i in range(len(batch)))
    info = pd.read_csv(directory / "set_b.csv")
    assert len(info) == 30 and set(info.columns) >= {"perturbation", "valid", "quality"}


def test_splits_partition_both_sets(tiny):
    directory, _ = tiny
    with np.load(directory / "splits.npz") as splits:
        for name in ("set_a", "set_b"):
            rows = np.concatenate([splits[f"{name}_{part}"] for part in ("train", "validation", "test")])
            assert np.array_equal(np.sort(rows), np.arange(30))


def test_diversity_reference_points_at_set_a_test_rooms(tiny, catalog):
    directory, _ = tiny
    set_a = load_layouts(directory / "set_a.npz")
    reference = load_layouts(directory / "diversity_reference.npz")
    info = pd.read_csv(directory / "diversity_reference.csv")
    with np.load(directory / "splits.npz") as splits:
        assert set(info["source"]) <= set(splits["set_a_test"].tolist())
    for i, source in enumerate(info["source"]):
        assert Condition.of(reference.layout(i, catalog), catalog) == Condition.of(set_a.layout(source, catalog), catalog)


def test_figures_are_drawn_from_the_files(tiny, tmp_path):
    directory, _ = tiny
    paths = make_figures(directory, tmp_path, SPLIT)
    assert sorted(p.name for p in paths) == sorted(f"dataset_tiny_{n}.png" for n in ("rooms", "set_b", "rejection", "f_max"))
    assert all(p.read_bytes()[:4] == b"\x89PNG" for p in paths)


def test_f_max_is_where_half_the_rooms_can_be_furnished():
    rng = np.random.default_rng(0)
    ratio = rng.uniform(0.05, 0.5, size=4000)
    furnished = rng.uniform(size=4000) < 1 / (1 + np.exp(40 * (ratio - 0.3)))
    f_max, coefficients = fit_f_max(pd.DataFrame({"ratio": ratio, "furnished": furnished}))
    assert f_max == pytest.approx(0.3, abs=0.01) and coefficients[1] == pytest.approx(-40, rel=0.15)
    assert fit_f_max(pd.DataFrame({"ratio": ratio, "furnished": True}))[0] is None  # nothing ever fails


def test_npz_hash_ignores_when_the_file_was_written(tmp_path, catalog, good_layout):
    batch = stack_layouts([good_layout], catalog)
    first, second = tmp_path / "a", tmp_path / "b"
    for directory in (first, second):
        directory.mkdir()
        save_layouts(directory / "set.npz", batch)
    assert file_hashes(first) == file_hashes(second)
    loaded = load_layouts(first / "set.npz")
    np.testing.assert_array_equal(loaded.center, batch.center)
    assert loaded.room_type == "living_room"
