"""Split tests: held-out regions never reach training data, splits partition it."""
import numpy as np
import pytest
import yaml

from spacegen.config import DEFAULT_CONFIG
from spacegen.generator import Condition, Region, RoomRange, generate_set_a, load_generator_config
from spacegen.perturb import generate_set_b, load_set_b_config
from spacegen.rules import check_layout
from spacegen.splits import (diversity_reference, generate_held_out, held_out_rooms, in_region, leaks,
                             load_split_config, split_indices, training_rooms)

CONFIG = load_generator_config()
SPLIT = load_split_config()


@pytest.fixture(scope="module")
def room(rules):
    return rules.rooms["living_room"]


def test_held_out_regions_match_the_tech_spec():
    regions = SPLIT.held_out
    assert regions["interpolation"].contains(5.0, 4.8) and not regions["interpolation"].contains(5.0, 4.0)  # 24, 20 m^2
    assert regions["unseen_combination"].contains(6.5, 5.5) and not regions["unseen_combination"].contains(6.0, 5.0)
    assert regions["out_of_range"].contains(7.5, 6.5) and not regions["out_of_range"].contains(7.0 - 1e-9, 6.5)
    assert SPLIT.fractions == (0.70, 0.15, 0.15)


def test_training_rooms_skip_every_held_out_region(room):
    rng = np.random.default_rng(0)
    rooms = np.array([training_rooms(room, SPLIT).sample(rng) for _ in range(3000)])
    area = rooms.prod(axis=1)
    assert not ((area >= 22) & (area <= 26)).any() and area.max() < 32  # the two area gaps
    assert rooms[:, 0].min() >= 3.5 and rooms[:, 0].max() <= 7.0
    assert rooms[:, 1].min() >= 3.0 and rooms[:, 1].max() <= 6.0
    assert (area < 22).any() and ((area > 26) & (area < 32)).any()  # both sides of the gap remain


@pytest.mark.parametrize("name", ["interpolation", "unseen_combination", "out_of_range"])
def test_held_out_rooms_fall_in_their_region(room, name):
    rng = np.random.default_rng(0)
    rooms = held_out_rooms(name, room, SPLIT)
    drawn = [rooms.sample(rng) for _ in range(500)]
    assert all(SPLIT.held_out[name].contains(w, d) for w, d in drawn)
    widths, depths = np.array(drawn).T
    if name == "out_of_range":
        assert widths.min() > 7.0 and depths.min() > 6.0
    else:  # inside the training ranges: only the combination is new
        assert widths.min() >= 3.5 and widths.max() <= 7.0 and depths.min() >= 3.0 and depths.max() <= 6.0


def test_a_room_range_that_cannot_be_met_fails_loudly():
    with pytest.raises(ValueError, match="no room"):
        RoomRange((3.0, 4.0), (3.0, 4.0), within=Region(area=(50.0, 60.0))).sample(np.random.default_rng(0))


def test_splits_partition_the_layouts():
    parts = split_indices(1000, SPLIT.fractions, np.random.default_rng(0))
    assert {name: len(rows) for name, rows in parts.items()} == {"train": 700, "validation": 150, "test": 150}
    assert np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(1000))  # disjoint and complete
    again = split_indices(1000, SPLIT.fractions, np.random.default_rng(0))
    assert all(np.array_equal(parts[name], again[name]) for name in parts)


def test_split_fractions_are_checked(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["splits"]["fractions"] = [0.7, 0.2, 0.2]
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="summing to 1"):
        load_split_config(path)


# --------------------------------------------------------------------------- no leakage

def test_training_data_never_contains_a_held_out_room(catalog, rules, room):
    rng = np.random.default_rng(0)
    rooms = training_rooms(room, SPLIT)
    set_a = generate_set_a(80, rng, catalog, rules, CONFIG, rooms)
    set_b = generate_set_b(80, rng, catalog, rules, CONFIG, load_set_b_config(), rooms)
    assert leaks(set_a.layouts, SPLIT) == leaks(set_b.layouts, SPLIT) == {name: 0 for name in SPLIT.held_out}


@pytest.mark.parametrize("name", ["interpolation", "unseen_combination", "out_of_range"])
def test_held_out_sets_hold_valid_layouts_of_their_region(catalog, rules, name):
    held_out = generate_held_out(name, 15, np.random.default_rng(0), catalog, rules, CONFIG, SPLIT)
    assert len(held_out.layouts) == 15 and in_region(held_out.layouts, SPLIT.held_out[name]).all()
    assert all(check_layout(held_out.layouts.layout(i, catalog), catalog, rules).valid for i in range(15))


def test_diversity_reference_reuses_each_test_room(catalog, rules):
    test = generate_set_a(10, np.random.default_rng(1), catalog, rules, CONFIG).layouts
    reference = diversity_reference(test, 4, 3, np.random.default_rng(2), catalog, rules, CONFIG)
    info = reference.info
    assert info["source"].nunique() == 4 and (info.groupby("room").size() <= 3).all()
    for i, source in enumerate(info["source"]):
        layout = reference.layouts.layout(i, catalog)
        assert Condition.of(layout, catalog) == Condition.of(test.layout(source, catalog), catalog)
        assert check_layout(layout, catalog, rules).valid


def test_condition_of_a_layout(good_layout, catalog):
    cond = Condition.of(good_layout.without(5), catalog)
    assert (cond.width, cond.depth, cond.door_wall, cond.door_offset) == (5.0, 4.0, "W", 0.5)
    assert cond.items == {"sofa": "sofa_3seater", "tv_unit": "tv_unit_standard", "coffee_table": "coffee_table_standard",
                          "bookshelf": "bookshelf_standard", "armchair": "armchair_standard"}
