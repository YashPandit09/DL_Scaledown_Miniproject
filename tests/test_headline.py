"""E1 and E10 tests: the harness's rooms and generators, M2 paired with M1, the cache,
the top 3 in three orders, and the figures."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.figures import FIGURES
from experiments.headline import (E10_METHODS, METHODS, MODEL_FREE, SETS, Headline, TopThree, combine, speed_check,
                                 timing_pass, with_timing)
from spacegen.dataset import load_layouts
from spacegen.evaluate import EvaluationConfig, evaluate_baselines
from spacegen.latent_opt import load_latent_opt_config
from spacegen.models.cvae import CVAE, CVAEConfig
from spacegen.models.evaluator import Evaluator, EvaluatorConfig
from spacegen.pipeline import PipelineConfig
from spacegen.raster import load_raster_config

SMALL = EvaluationConfig(rooms=4, samples=3)
TIMING = ["ms_per_valid", "seconds_per_room"]


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    raster = load_raster_config()
    evaluator = Evaluator(EvaluatorConfig(channels=(4, 4, 4, 4), hidden=8), pixels=raster.pixels).eval()
    return CVAE(CVAEConfig(hidden=16)).eval(), evaluator, raster


@pytest.fixture
def make(tiny_dataset, tmp_path, catalog, rules, models):
    cvae, evaluator, raster = models
    log = []

    def headline(steps: int = 3, inputs: dict | None = None, fresh: bool = False) -> Headline:
        return Headline(tiny_dataset, tmp_path / "cache", 0, SMALL, cvae, catalog, rules,
                        load_latent_opt_config(steps=steps), PipelineConfig(), evaluator, raster, "cpu", inputs, fresh,
                        log=log.append)

    headline.log = log
    return headline


def test_e1_baselines_reproduce_the_harness(make, tiny_dataset, catalog, rules):
    table = make().e1()
    assert table["method"].tolist() == list(METHODS)
    assert table["rvr"].between(0, 1).all() and (table["samples"] == SMALL.rooms * SMALL.samples).all()
    expected = evaluate_baselines(tiny_dataset, SMALL, 0, catalog, rules, log=lambda message: None)
    shared = [c for c in expected.columns if c not in TIMING]
    pd.testing.assert_frame_equal(table[shared].iloc[:3].reset_index(drop=True), expected[shared])


def test_m2_starts_from_m1s_draws(make):
    """With no optimization steps M2 is M1: both draw the same z for every room."""
    headline = make(steps=0)
    m1, m2 = (headline.per_room("in_distribution", method) for method in ("M1", "M2"))
    columns = ["valid", "overlap", "reachability", "quality", "diversity", "top_rule", "top_evaluator"]
    pd.testing.assert_frame_equal(m1[columns], m2[columns])


def test_e10_takes_the_in_distribution_rows_from_e1(make):
    headline = make()
    e1 = headline.e1().set_index("method")
    computed = len(make.log)
    e10 = headline.e10()
    assert [line.split(",")[0] for line in make.log[computed:]] == [s for s in SETS[1:] for _ in E10_METHODS]
    assert list(zip(e10["set"], e10["method"])) == [(s, m) for s in SETS for m in E10_METHODS]
    assert not any(c.startswith("diversity") for c in e10.columns)
    inside = e10[e10["set"] == "in_distribution"].set_index("method")
    for column in ("rvr", "quality", "quality_top3", "attempts_per_valid"):
        assert inside[column].equals(e1.loc[list(E10_METHODS), column])
    assert (e10.groupby("set", sort=False)["area_m2"].first().loc[["unseen_combination", "out_of_range"]] > 32).all()


def test_cached_rows_are_reused_until_the_inputs_change(make):
    first = make().e1()
    computed = len(make.log)
    assert computed == len(METHODS)
    again = make().e1()
    assert len(make.log) == computed  # nothing recomputed
    pd.testing.assert_frame_equal(again.drop(columns=TIMING), first.drop(columns=TIMING))
    pd.testing.assert_frame_equal(again[TIMING], first[TIMING])  # the timings come from the cache too
    make(inputs={"cvae": "another model"}).e1()
    assert len(make.log) == 2 * computed
    make(inputs={"cvae": "another model"}, fresh=True).e1()
    assert len(make.log) == 3 * computed


def test_top_three_in_three_orders(tiny_dataset, catalog, rules, models):
    _, evaluator, raster = models
    batch = load_layouts(tiny_dataset / "set_a.npz")
    a, b = batch.layout(0, catalog), batch.layout(1, catalog)
    quality = np.array([0.9, 0.8, 0.1])
    config = PipelineConfig(top_k=2)
    top = TopThree(config, np.random.default_rng(0), catalog, rules)
    row = top([a, a, b], quality)  # the copy of a is never diverse enough, so b joins
    assert row["top_rule"] == pytest.approx(0.5) and np.isnan(row["top_evaluator"])
    assert row["top_random"] in (pytest.approx(0.5), pytest.approx(0.45))
    best = TopThree(dataclasses.replace(config, top_k=1), np.random.default_rng(0), catalog, rules, evaluator, raster)
    row = best([a, a, b], quality)
    assert row["top_rule"] == 0.9 and row["top_evaluator"] in (0.9, 0.8, 0.1)
    assert all(np.isnan(v) for v in top([], np.array([])).values())


def test_figures_draw_from_the_tables(make, tmp_path):
    headline = make()
    for name, table in (("e1", headline.e1()), ("e10", headline.e10())):
        figures = FIGURES[name](table, tmp_path)
        assert figures and all(fig.axes for fig in figures.values())


@pytest.fixture
def seeds(tiny_dataset, tmp_path, catalog, rules, models):
    """Two CVAE 'seeds' sharing one cache for B1, B2 and G0."""
    _, evaluator, raster = models
    log = []
    runs = []
    for k in range(2):
        torch.manual_seed(k)
        runs.append(Headline(tiny_dataset, tmp_path / f"seed-{k}", 0, SMALL, CVAE(CVAEConfig(hidden=16)), catalog,
                             rules, load_latent_opt_config(steps=3), PipelineConfig(), evaluator, raster, "cpu",
                             {"cvae": f"model {k}"}, log=log.append, baseline_cache=tmp_path / "baselines"))
    return runs, log


def test_several_runs_share_the_baselines_and_combine_into_mean_and_spread(seeds, tmp_path):
    runs, log = seeds
    tables = [run.e1() for run in runs]
    computed = [line.split(":")[0] for line in log]
    assert computed.count("in_distribution, B1") == 1 and computed.count("in_distribution, M2") == 2
    pd.testing.assert_frame_equal(tables[0].iloc[:3], tables[1].iloc[:3])  # the same cached B1, B2, G0 rows
    table = combine(tables, ["method"])
    assert table["method"].tolist() == list(METHODS) and table["seeds"].tolist() == [1, 1, 1, 2, 2]
    m2 = table.set_index("method").loc["M2"]
    assert m2["rvr"] == pytest.approx(np.mean([t.set_index("method").loc["M2", "rvr"] for t in tables]))
    assert m2["rvr_std"] == pytest.approx(np.std([t.set_index("method").loc["M2", "rvr"] for t in tables], ddof=1))
    assert table.set_index("method").loc[list(MODEL_FREE), "rvr_std"].isna().all()
    for figure in FIGURES["e1"](table, tmp_path).values():
        assert figure.axes and "2 CVAE seeds" in figure.texts[0].get_text()


def test_e10_can_be_computed_one_set_at_a_time(seeds, tmp_path):
    runs, log = seeds
    part = runs[0].e10(["interpolation"])
    assert part["set"].unique().tolist() == ["interpolation"]
    log.clear()
    whole = runs[0].e10()
    assert not any(line.startswith("interpolation") for line in log)  # computed in the first call
    pd.testing.assert_frame_equal(whole[whole["set"] == "interpolation"].reset_index(drop=True), part)
    table = combine([whole, runs[1].e10()], ["set", "method"])
    assert list(zip(table["set"], table["method"])) == [(s, m) for s in SETS for m in E10_METHODS]
    assert (table.loc[table["method"] == "G0", "seeds"] == 1).all()
    assert all(fig.axes for fig in FIGURES["e10"](table, tmp_path).values())


def test_e8_figure_takes_a_combined_table(tmp_path):
    rows = [{"steps": s, "method": f"M2, {s} steps", "rooms": 4, "rvr": 0.1 + s / 400 + k / 100,
             "diversity_ratio": 0.8, "seconds_per_room": 0.1 + s / 200} for k in range(2) for s in (0, 25, 50)]
    tables = [pd.DataFrame(rows[:3]), pd.DataFrame(rows[3:])]
    table = combine(tables, ["steps", "method"])
    assert table["rvr_std"].round(6).tolist() == [round(np.std([0.0, 0.01], ddof=1), 6)] * 3
    figure = FIGURES["e8"](table, tmp_path)["steps"]
    assert "2 CVAE seeds" in figure.texts[0].get_text()


def test_timing_pass_takes_the_samplers_in_turn(make, catalog, rules):
    from spacegen.baselines import UniformBaseline
    from spacegen.evaluate import generator_sampler

    rooms = make().rooms("in_distribution")[0][:2]
    samplers = {"B1": (UniformBaseline(catalog), False), "G0": (generator_sampler(catalog, rules), True)}
    rows = timing_pass(samplers, rooms, 3, 0, catalog, rules)
    assert rows["sampler"].tolist() == ["B1", "G0", "B1", "G0"] and rows["room"].tolist() == [0, 0, 1, 1]
    assert (rows["seconds"] > 0).all() and (rows["attempts"] == 3).all() and rows["valid"].between(0, 3).all()


def test_timing_columns_come_from_the_timing_pass():
    rows = pd.DataFrame([{"sampler": name, "room": room, "seconds": seconds, "valid": valid}
                         for name, seconds, valid in (("B1", 0.1, 2), ("M2|a", 1.0, 10), ("M2|b", 2.0, 20))
                         for room in (0, 1)])
    rows.loc[1, "seconds"] = 0.3  # B1: 0.1 s and 0.3 s
    table = pd.DataFrame({"method": ["B1", "M2"], "seeds": [1, 2], "seconds_per_room": [9.0, 9.0],
                          "seconds_per_room_std": [np.nan, 1.0], "ms_per_valid": [9.0, 9.0],
                          "ms_per_valid_std": [np.nan, 1.0]})
    timed = with_timing(table, rows, "method").set_index("method")
    assert timed.loc["B1", "seconds_per_room"] == pytest.approx(0.2) and timed.loc["B1", "ms_per_valid"] == pytest.approx(100)
    assert np.isnan(timed.loc["B1", "seconds_per_room_std"])  # one sampler: no spread
    assert timed.loc["M2", "seconds_per_room"] == pytest.approx(1.5)
    assert timed.loc["M2", "seconds_per_room_std"] == pytest.approx(np.std([1.0, 2.0], ddof=1))
    assert timed.loc["M2", "ms_per_valid"] == pytest.approx(100) and timed.loc["M2", "ms_per_valid_std"] == pytest.approx(0)
    assert (timed["seconds_per_room_run"] == 9.0).all() and (timed["timing_rooms"] == 2).all()
    steps = with_timing(pd.DataFrame({"steps": [0, 25], "seconds_per_room": [1.0, 1.0], "ms_per_valid": [1.0, 1.0]}),
                        rows.assign(sampler=["0|a", "0|a", "25|a", "25|a", "25|b", "25|b"]), "steps")
    assert steps["seconds_per_room"].tolist() == pytest.approx([0.2, 1.5]) and "seconds_per_room_std" not in steps


def test_speed_check_times_a_fixed_workload():
    assert 0 < speed_check(repeats=1) < 60
