"""App tests: the logic behind the Streamlit app, and the app itself run headless."""
import dataclasses
import json

import numpy as np
import pytest
import torch

from spacegen import app_logic as logic
from spacegen.layout import layout_from_dict
from spacegen.models.cvae import CVAEConfig
from spacegen.models.evaluator import Evaluator, EvaluatorConfig
from spacegen.paths import REPO_ROOT
from spacegen.pipeline import Request
from spacegen.raster import load_raster_config
from spacegen.train_cvae import TrainingConfig, train_cvae

ITEMS = {"sofa": None, "tv_unit": None, "coffee_table": "coffee_table_standard"}


@pytest.fixture(scope="module")
def runs(tiny_dataset, catalog, tmp_path_factory):
    """A tiny trained CVAE run and an untrained evaluator run, saved the way the real runs are."""
    root = tmp_path_factory.mktemp("app_runs")
    training = TrainingConfig(batch=8, max_epochs=3, anneal_epochs=1, patience=5, check_rooms=0)
    train_cvae(tiny_dataset, root / "cvae", 0, "cpu", CVAEConfig(hidden=32), training, catalog,
               log=lambda message: None)
    torch.manual_seed(0)
    raster, config = load_raster_config(), EvaluatorConfig(channels=(4, 4, 4, 4), hidden=8)
    (root / "evaluator").mkdir()
    torch.save(Evaluator(config, pixels=raster.pixels).state_dict(), root / "evaluator" / "model.pt")
    (root / "evaluator" / "run.json").write_text(json.dumps({
        "evaluator": dataclasses.asdict(config), "raster": dataclasses.asdict(raster)}), encoding="utf-8")
    return root / "cvae", root / "evaluator", tiny_dataset


@pytest.fixture(scope="module")
def models(runs, catalog):
    return logic.load_models(catalog, *runs)


def test_missing_models_become_notes_not_errors(catalog, tmp_path):
    empty = logic.load_models(catalog, tmp_path / "a", tmp_path / "b", tmp_path / "c")
    assert empty.cvae is None and empty.evaluator is None and empty.b2 is None
    assert len(empty.notes) == 3 and all("python run.py" in note for note in empty.notes)


def test_everything_loads_from_saved_runs(models, runs):
    assert models.cvae is not None and models.evaluator is not None and models.b2 is not None
    assert models.cvae_run == runs[0] and not models.notes


def test_describe_reports_checks_cost_and_floor_use(good_layout, catalog, rules):
    facts = logic.describe(good_layout, catalog, rules)
    assert facts["valid"] and all(facts["checks"].values()) and list(facts["checks"]) == list(logic.CHECKS.values())
    prices = sum(catalog.slot(int(k)).variant(good_layout.variant_ids[k]).price for k in np.flatnonzero(good_layout.mask))
    assert facts["cost"] == prices and 0 < facts["floor_use"] < rules.f_max and 0 <= facts["quality"] <= 1
    assert set(facts["terms"]) == {"alignment", "relations", "circulation", "space"}
    crowded = good_layout.with_item(0, center=good_layout.center[1])  # the sofa on top of the TV unit
    broken = logic.describe(crowded, catalog, rules)
    assert not broken["valid"] and not broken["checks"]["H2 no overlap"]


def test_condition_or_the_reason_it_cannot_be_furnished(catalog, rules):
    cond, reason, warnings = logic.condition_for(Request(5.0, 4.0, "W", 0.5, ITEMS), catalog, rules)
    assert cond is not None and reason is None and not warnings and cond.items["sofa"] == "sofa_3seater"
    cond, reason, _ = logic.condition_for(Request(5.0, 4.0, "W", 0.5, ITEMS, budget=1000.0), catalog, rules)
    assert cond is None and "budget" in reason
    cond, reason, _ = logic.condition_for(Request(5.0, 4.0, "W", 0.5, {"tv_unit": None}), catalog, rules)
    assert cond is None and "sofa" in reason
    _, _, warnings = logic.condition_for(Request(7.8, 6.5, "W", 0.5, ITEMS), catalog, rules)
    assert warnings and "training ranges" in warnings[0]


def test_request_runs_through_the_pipeline_and_exports(models, catalog, rules):
    result = logic.run_request(models, Request(5.0, 4.0, "W", 0.5, ITEMS), 0, 16, True, catalog, rules)
    assert result.candidates == 16 and result.condition is not None and len(result.top) <= 3
    assert result.valid >= len(result.top)
    layout = result.top[0].layout if result.top else None
    if layout is None:  # a three-epoch model may give no valid layout; the exports still need one
        layout = logic.compare_methods(models, result.condition, 16, 0, catalog, rules)[1]["G0 generator"]
    assert logic.layout_png(layout, catalog, rules)[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(logic.layout_json(layout, catalog, rules, {"quality": 0.5}, {"rank": 1}))
    assert data["metrics"] == {"quality": 0.5} and data["meta"] == {"rank": 1}
    assert np.allclose(layout_from_dict(data, catalog).center, layout.center)


def test_five_methods_sample_the_same_room(models, catalog, rules):
    cond, _, _ = logic.condition_for(Request(5.0, 4.0, "W", 0.5, ITEMS), catalog, rules)
    table, best = logic.compare_methods(models, cond, 12, 0, catalog, rules)
    assert table["method"].tolist() == list(logic.METHODS) == list(best)
    assert (table["raw samples"] == 12).all() and table["valid"].between(0, 12).all()
    for name, layout in best.items():
        row = table.set_index("method").loc[name]
        assert (layout is None) == (row["valid"] == 0)
        if layout is not None:
            assert logic.describe(layout, catalog, rules)["quality"] == pytest.approx(row["best quality"])
    again, _ = logic.compare_methods(models, cond, 12, 0, catalog, rules)
    assert again.drop(columns="seconds").equals(table.drop(columns="seconds"))  # the same seed, the same samples
    without, best = logic.compare_methods(logic.Models(), cond, 4, 0, catalog, rules)
    assert without["note"].str.startswith("not available").tolist() == [False, True, False, True, True]
    assert best["M1 CVAE"] is None


def test_saved_figures_are_listed_headline_first(tmp_path):
    for name in ("e2_results", "e1_final_results", "something_else"):
        (tmp_path / f"{name}.png").write_bytes(b"png")
    listed = logic.result_figures(tmp_path)
    assert [path.stem for _, path in listed] == ["e1_final_results", "e2_results", "something_else"]
    assert listed[0][0].startswith("E1") and listed[2][0] == "something else"


def test_results_table_shows_the_headline_columns_by_name():
    import pandas as pd

    from spacegen.paths import REPO_ROOT

    final = logic.headline_table(pd.read_csv(REPO_ROOT / "reports" / "tables" / "e1_final.csv"))
    assert list(final.columns) == list(logic.E1_COLUMNS.values())
    m2 = final.set_index("method").loc["M2"]
    assert m2["raw valid (%)"] == pytest.approx(69.8, abs=0.05) and m2["seeds"] == 3  # percent, not a share
    first = logic.headline_table(pd.read_csv(REPO_ROOT / "reports" / "tables" / "e1.csv"))  # one run: no seed columns
    assert "seeds" not in first.columns and list(first.columns)[:2] == ["method", "raw valid (%)"]


@pytest.fixture
def app(runs, monkeypatch):
    """The Streamlit script run headless on the tiny models."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    for name, path in zip(("SPACEGEN_CVAE_RUN", "SPACEGEN_EVALUATOR_RUN", "SPACEGEN_DATA_DIR"), runs):
        monkeypatch.setenv(name, str(path))
    st.cache_resource.clear()
    yield AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=300)
    st.cache_resource.clear()


def test_app_generates_and_compares_from_a_clean_start(app):
    app.run()
    assert not app.exception and not app.warning  # every model was found
    assert [button.label for button in app.button] == ["Generate layouts", "Compare the five methods on this room"]
    assert "result" not in app.session_state
    app.button[0].click().run()
    assert not app.exception
    result, _ = app.session_state["result"]
    assert result.candidates == 64 and result.condition is not None
    # the large numbers are the two that differ between layouts; long values (the cost) are in a caption
    assert [metric.label for metric in app.metric] == ["Rule quality", "Evaluator score"] * len(result.top)
    assert all(len(metric.value) == 4 for metric in app.metric)
    assert sum("INR" in caption.value for caption in app.caption) == len(result.top)
    app.button[1].click().run()
    assert not app.exception
    table, best, _ = app.session_state["comparison"]
    assert len(table) == 5 and set(best) == set(logic.METHODS)


def test_app_explains_an_infeasible_request(app):
    app.run()
    budget = next(widget for widget in app.number_input if widget.label.startswith("Budget"))
    budget.set_value(1000).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("cannot be furnished" in error.value for error in app.error)


def test_app_opens_without_trained_models(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    for name in ("SPACEGEN_CVAE_RUN", "SPACEGEN_EVALUATOR_RUN", "SPACEGEN_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name))
    st.cache_resource.clear()
    app = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    st.cache_resource.clear()
    assert not app.exception and len(app.warning) == 3
    assert app.button[0].disabled  # nothing to generate with


def test_comparison_keeps_a_pinned_item_on_its_spot(models, catalog, rules):
    from spacegen.pins import Pin

    request = Request(5.0, 4.0, "W", 0.5, ITEMS, pins={"sofa": Pin(2.5, 3.5, 2)})
    cond, reason, _ = logic.condition_for(request, catalog, rules)
    assert cond is not None and reason is None
    _, best = logic.compare_methods(models, cond, 12, 0, catalog, rules, request.pins)
    for layout in best.values():
        if layout is not None:
            assert np.allclose(layout.center[0], (2.5, 3.5), atol=1e-5) and int(layout.rot[0]) == 2
    cond, reason, _ = logic.condition_for(Request(5.0, 4.0, "W", 0.5, ITEMS, pins={"sofa": Pin(0.1, 0.1)}), catalog, rules)
    assert cond is None and "stick out" in reason


def test_app_keeps_a_pinned_item_on_its_spot(app):
    app.run()
    pin = next(widget for widget in app.selectbox if widget.label.startswith("Keep this item"))
    pin.select("sofa").run()
    app.button[0].click().run()
    assert not app.exception
    result, used = app.session_state["result"]
    ((name, wanted),) = used[-1]
    assert name == "sofa" and (wanted.x, wanted.y) == (2.5, 2.0) and result.request.pins == {"sofa": wanted}
    assert all(np.allclose(candidate.layout.center[0], (2.5, 2.0), atol=1e-5) for candidate in result.top)


def test_demo_assets_come_from_the_apps_logic(models, catalog, rules, tmp_path):
    from experiments.demo_assets import make_assets

    written = make_assets(models, tmp_path, catalog, rules, candidates=16, log=lambda message: None)
    names = {path.name for path in written}
    assert {"layouts.png", "pinned.png", "compare.csv", "compare.png"} <= names
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_generate_command_takes_a_pin(runs, tmp_path, capsys):
    from spacegen.pipeline import main

    code = main(["--width", "5", "--depth", "4", "--door", "W", "--items", "sofa,tv_unit", "--pin", "sofa:2.5,3.4,S",
                 "--cvae", str(runs[0]), "--evaluator", str(runs[1]), "--out", str(tmp_path)])
    assert code == 0 and "candidates valid" in capsys.readouterr().out
    for path in tmp_path.glob("top*.json"):
        sofa = next(item for item in json.loads(path.read_text(encoding="utf-8"))["items"] if item["slot"] == 0)
        assert (sofa["x"], sofa["y"]) == pytest.approx((2.5, 3.4), abs=1e-5) and sofa["rotation"] == 2
