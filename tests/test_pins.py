"""Pinned-furniture tests: the pin holds exactly in every method, impossible pins are refused, and E12 runs."""
import numpy as np
import pytest
import torch

from experiments.figures import FIGURES
from experiments.pinned import METHODS, pinned_requests, run_requests, summarize
from spacegen.baselines import StatisticalBaseline, UniformBaseline, load_baseline_config
from spacegen.dataset import load_layouts
from spacegen.evaluate import generator_sampler
from spacegen.generator import Condition
from spacegen.latent_opt import load_latent_opt_config
from spacegen.layout import is_canonical
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.pins import Pin, PinnedSampler, displacement, pin_errors, snap
from spacegen.pipeline import CVAESampler, LatentOptSampler, Request, generate

SOFA, TABLE = 0, 2


@pytest.fixture(scope="module")
def cvae():
    torch.manual_seed(0)
    return CVAE(CVAEConfig(hidden=16)).eval()


@pytest.fixture
def room(good_layout, catalog):
    return Condition.of(good_layout, catalog)


@pytest.fixture(scope="module")
def b2(tiny_dataset, catalog):
    set_a = load_layouts(tiny_dataset / "set_a.npz")
    with np.load(tiny_dataset / "splits.npz") as splits:
        return StatisticalBaseline(catalog, load_baseline_config()).fit(set_a.subset(splits["set_a_train"]))


def on_pin(layout, slot: int, pin: Pin) -> bool:
    return bool(np.allclose(layout.center[slot], (pin.x, pin.y), atol=1e-5)) and (
        pin.facing is None or int(layout.rot[slot]) == pin.facing)


def test_snap_moves_only_the_pinned_item_and_keeps_the_canonical_form(good_layout, catalog):
    pins = {"sofa": Pin(2.0, 1.5, facing=1)}
    assert displacement(good_layout, pins, catalog) == pytest.approx(np.hypot(*(good_layout.center[SOFA] - (2.0, 1.5))))
    snapped = snap(good_layout, pins, catalog)
    assert on_pin(snapped, SOFA, pins["sofa"]) and is_canonical(snapped, catalog)
    others = [k for k in np.flatnonzero(good_layout.mask) if k != SOFA]
    assert np.array_equal(snapped.center[others], good_layout.center[others])
    turned = snap(good_layout, {"coffee_table": Pin(2.5, 2.0, facing=2)}, catalog)  # a half turn changes nothing
    assert int(turned.rot[TABLE]) == 0 and np.allclose(turned.center[TABLE], (2.5, 2.0))
    kept = snap(good_layout, {"sofa": Pin(2.0, 1.5)}, catalog)  # no facing asked for: the item keeps its own
    assert int(kept.rot[SOFA]) == int(good_layout.rot[SOFA])


def test_impossible_pins_are_refused_with_the_reason(room, catalog, rules):
    assert pin_errors({"sofa": Pin(2.5, 2.0)}, room, catalog, rules) == []
    assert "not among the furniture" in pin_errors({"bookshelf": Pin(1, 1)}, Condition(5.0, 4.0, "W", 0.5, {
        "sofa": "sofa_3seater", "tv_unit": "tv_unit_standard"}), catalog, rules)[0]
    assert "stick out of the room" in pin_errors({"sofa": Pin(0.2, 0.2)}, room, catalog, rules)[0]
    assert "facing must be" in pin_errors({"sofa": Pin(2.5, 2.0, facing=5)}, room, catalog, rules)[0]
    # the 2.1 m sofa fits beside a wall only when it runs along it: facing East or West, not North
    assert pin_errors({"sofa": Pin(room.width - 0.5, 2.0, facing=1)}, room, catalog, rules) == []
    assert "stick out" in pin_errors({"sofa": Pin(room.width - 0.5, 2.0, facing=0)}, room, catalog, rules)[0]
    assert pin_errors({"sofa": Pin(room.width - 0.5, 2.0)}, room, catalog, rules) == []  # some facing fits
    door = Condition(5.0, 4.0, "W", 0.5, room.items)  # the door's zone reaches 0.9 m into the room at mid-height
    assert "door" in pin_errors({"side_table": Pin(0.4, 2.0)}, Condition(5.0, 4.0, "W", 0.5, {
        **door.items, "side_table": "side_table_standard"}), catalog, rules)[0]


def test_baselines_place_the_pinned_item_first(room, catalog, b2):
    pins = {"sofa": Pin(2.5, 2.0, facing=2)}
    for base in (UniformBaseline(catalog), b2):
        sampler = PinnedSampler(base, pins, catalog, takes_pins=True)
        layouts = sampler.sample(room, 8, np.random.default_rng(0)).layouts
        assert len(layouts) == 8 and all(on_pin(layout, SOFA, pins["sofa"]) for layout in layouts)
        same = [base.sample(room, 4, np.random.default_rng(1), pins=p).layouts for p in (None, {})]
        plain = base.sample(room, 4, np.random.default_rng(1)).layouts
        assert all(np.array_equal(a.center, b.center) for group in same for a, b in zip(group, plain))  # no pin, no change


def test_generator_layouts_get_the_item_moved_onto_its_pin(room, catalog, rules):
    pins = {"tv_unit": Pin(2.5, 0.25, facing=0)}
    sampler = PinnedSampler(generator_sampler(catalog, rules), pins, catalog)
    samples = sampler.sample(room, 12, np.random.default_rng(0))
    assert samples.attempts == 12 and sampler.name == "G0-pin"
    assert samples.layouts and all(on_pin(layout, 1, pins["tv_unit"]) for layout in samples.layouts)
    assert sampler.last_displacement.shape == (len(samples.layouts),) and (sampler.last_displacement >= 0).all()


def test_cvae_samplers_snap_the_pinned_item(room, catalog, rules, cvae):
    pins = {"sofa": Pin(2.5, 2.0, facing=3), "coffee_table": Pin(3.6, 2.0)}
    for sampler in (CVAESampler(cvae, catalog, pins=pins),
                    LatentOptSampler(cvae, catalog, rules, load_latent_opt_config(steps=5), pins=pins)):
        layouts = sampler.sample(room, 6, np.random.default_rng(0)).layouts
        assert all(on_pin(layout, SOFA, pins["sofa"]) and on_pin(layout, TABLE, pins["coffee_table"]) for layout in layouts)
        assert all(is_canonical(layout, catalog) for layout in layouts)
        assert sampler.last_displacement.shape == (6,) and (sampler.last_displacement >= 0).all()
    plain = CVAESampler(cvae, catalog).sample(room, 6, np.random.default_rng(0)).layouts
    pinned = CVAESampler(cvae, catalog, pins={"sofa": Pin(2.5, 2.0)}).sample(room, 6, np.random.default_rng(0)).layouts
    assert all(np.allclose(a.center[1:], b.center[1:]) for a, b in zip(plain, pinned))  # the same draws, one item moved


def test_pipeline_honours_a_pin_or_says_why_not(catalog, rules, cvae):
    items = {"sofa": "sofa_3seater", "tv_unit": "tv_unit_standard"}
    pin = Pin(2.5, 3.5, facing=2)
    result = generate(Request(5.0, 4.0, "W", 0.5, items, pins={"sofa": pin}), catalog, rules, cvae,
                      np.random.default_rng(0))
    assert result.condition is not None and result.candidates == 64
    assert all(on_pin(candidate.layout, SOFA, pin) for candidate in result.top)
    refused = generate(Request(5.0, 4.0, "W", 0.5, items, pins={"sofa": Pin(0.1, 0.1)}), catalog, rules, cvae,
                       np.random.default_rng(0))
    assert refused.condition is None and "stick out of the room" in refused.message and not refused.top


def test_e12_runs_on_the_reference_rooms(tiny_dataset, catalog, rules, cvae, b2, tmp_path):
    requests = pinned_requests(tiny_dataset, catalog, 2, 0)
    per_item = {}
    for item, cond, pins in requests:
        per_item[item] = per_item.get(item, 0) + 1
        assert list(pins) == [item] and item in cond.items and pin_errors(pins, cond, catalog, rules) == []
    assert per_item["sofa"] == 2 and all(count <= 2 for count in per_item.values())
    assert [item for item, _, _ in requests] == sorted((item for item, _, _ in requests),
                                                       key=lambda name: catalog.slot(name).index)
    log = []
    rows = run_requests(requests[:3], {"a": cvae, "b": cvae}, b2, 4, 0, catalog, rules, load_latent_opt_config(steps=3),
                        log=log.append)
    assert len(rows) == 3 * 7 and rows["sampler"].nunique() == 7 and log == ["e12: 3 of 3 requests"]
    assert rows.loc[rows["sampler"].isin(["B1", "B2"]), "displacement"].isna().all()
    same = rows[rows["sampler"].isin(["M2|a", "M2|b"])].groupby("sampler")["valid"].apply(list)
    assert same["M2|a"] == same["M2|b"]  # the same model with the same draws
    table = summarize(rows)
    assert set(table["method"]) == set(METHODS) and set(table["item"]) == {"sofa", "tv_unit", "all"}
    both = table.set_index(["item", "method"])
    assert both.loc[("all", "M2"), "seeds"] == 2 and both.loc[("all", "B1"), "seeds"] == 1
    assert both.loc[("all", "M2"), "rvr_std"] == pytest.approx(0) and np.isnan(both.loc[("all", "B1"), "rvr_std"])
    assert all(figure.axes for figure in FIGURES["e12"](table, tmp_path).values())
