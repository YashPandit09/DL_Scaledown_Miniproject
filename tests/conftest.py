"""Shared fixtures: the real catalog and rules, and a hand-built valid living room."""
import pytest

from spacegen.catalog import load_room_catalog
from spacegen.layout import make_layout
from spacegen.rules import load_rules
from tests.layouts import GOOD_ITEMS


@pytest.fixture(scope="session")
def catalog():
    return load_room_catalog("living_room")


@pytest.fixture(scope="session")
def rules():
    return load_rules()


@pytest.fixture
def good_layout(catalog):
    return make_layout(catalog, 5.0, 4.0, "W", 0.5, GOOD_ITEMS)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory, catalog, rules):
    """A dataset build small enough for tests: 40 Set A and 40 Set B layouts, tiny held-out sets."""
    import dataclasses

    from spacegen.build_dataset import build_dataset, load_dataset_config
    from spacegen.generator import load_generator_config
    from spacegen.perturb import load_set_b_config
    from spacegen.splits import load_split_config

    sizes = dataclasses.replace(load_dataset_config(), version="tiny", set_a=40, set_b=40, calibration_rooms=20)
    split = dataclasses.replace(load_split_config(), held_out_rooms=2, diversity_rooms=3, diversity_layouts=3)
    directory = tmp_path_factory.mktemp("tiny_dataset")
    build_dataset(directory, 1, sizes, catalog, rules, load_generator_config(), load_set_b_config(), split,
                  log=lambda message: None)
    return directory
