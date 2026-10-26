"""Regeneration tests: the step list, the output folder, and the comparison with saved results."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd

from experiments.regenerate import check, compare_table, is_timing, model_hashes, record, run_all, steps
from spacegen.paths import REPO_ROOT


def test_steps_are_named_once_and_cover_the_pipeline_in_order():
    names = [name for name, _ in steps()]
    assert len(set(names)) == len(names)
    order = ["data", "evaluator", "screening", "gate2", "frozen-seeds", "e1", "e8", "e10", "e12", "figures"]
    assert [name for name in names if name in order] == order
    commands = {name: commands for name, commands in steps()}
    assert all("--tag" in command and "final" in command for name in ("e1", "e8", "e10") for command in commands[name])
    assert [command[command.index("--sets") + 1] for name, group in steps() if name.startswith("e10-")
            for command in group] == ["interpolation", "unseen_combination", "out_of_range"]


def test_steps_run_in_order_and_stop_at_a_failure():
    ran, log = [], []

    def run(command, cwd):
        ran.append(command[2:])
        return SimpleNamespace(returncode=1 if command[2] == "e8" else 0)

    assert run_all("frozen-seeds", "e1", run=run, log=log.append) == 0
    assert [command[0] for command in ran] == ["train-cvae"] * 3 + ["e1"]
    assert [command[command.index("--seed") + 1] for command in ran[:3]] == ["0", "1", "2"]
    ran.clear()
    assert run_all("e1", "e12", run=run, log=log.append) == 1  # e8 fails: nothing after it runs
    assert [command[0] for command in ran] == ["e1", "e8"] and "continue with: python run.py all --from e8" in log[-1]


def test_outputs_follow_the_environment_variable(tmp_path):
    code = "import spacegen.paths as p; print(p.DATA_DIR); print(p.RUNS_DIR); print(p.REPORTS_DIR); print(p.CONFIG_DIR)"
    env = {**os.environ, "SPACEGEN_OUTPUT": str(tmp_path)}
    lines = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    assert lines[:3] == [str(tmp_path.resolve() / name) for name in ("data", "runs", "reports")]
    assert lines[3] == str(REPO_ROOT / "configs")  # the configuration always comes from the repository
    env.pop("SPACEGEN_OUTPUT")
    plain = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    assert plain[1] == str(REPO_ROOT / "runs")


def test_tables_are_compared_outside_their_timing_columns():
    assert is_timing("seconds_per_room") and is_timing("ms_per_valid_std") and is_timing("m2_seconds_per_room")
    assert not is_timing("rvr") and not is_timing("timing_rooms")
    saved = pd.DataFrame({"method": ["B1", "M2"], "rvr": [0.1, 0.7], "ms_per_valid": [20.0, 22.0], "quality": [0.3, None]})
    assert compare_table(saved, saved.assign(ms_per_valid=[1.0, 2.0])) is None  # timings may differ, NaN equals NaN
    assert compare_table(saved, saved.assign(rvr=[0.1, 0.71])) == "columns differ: rvr"
    assert compare_table(saved, saved.drop(columns="quality")) == "different columns"
    assert compare_table(saved, saved.iloc[:1]) == "2 rows saved, 1 now"


def test_check_counts_differences_in_tables_models_and_the_dataset(tmp_path):
    tables, runs, data = tmp_path / "tables", tmp_path / "runs", tmp_path / "data" / "v1"
    tables.mkdir()
    table = pd.DataFrame({"method": ["M2"], "rvr": [0.7], "seconds_per_room": [0.9]})
    for name in ("same.csv", "changed.csv", "fresh.csv"):
        table.to_csv(tables / name, index=False)
    saved = {"same.csv": table.assign(seconds_per_room=[5.0]).to_csv(index=False),
             "changed.csv": table.assign(rvr=[0.6]).to_csv(index=False)}
    (runs / "cvae" / "frozen" / "seed-0").mkdir(parents=True)
    (runs / "cvae" / "frozen" / "seed-0" / "model.pt").write_bytes(b"weights")
    data.mkdir(parents=True)
    (data / "metadata.json").write_text(json.dumps({"hash": "abc"}), encoding="utf-8")
    hashes = tmp_path / "model_hashes.json"
    recorded = record(hashes, runs, data)
    assert recorded["dataset"] == {"v1": "abc"} and list(recorded["models"]) == ["cvae/frozen/seed-0"]
    assert model_hashes(runs) == recorded["models"]
    log = []
    options = dict(tables_dir=tables, hashes=hashes, runs_dir=runs, data_dir=data, frozen=tmp_path / "none.yaml",
                   show=saved.get, log=log.append, copies=tmp_path / "no_copies", listed=lambda: sorted(saved))
    assert check(**options) == 1  # only changed.csv
    assert log[:3] == ["DIFF   changed.csv: columns differ: rvr", "new    fresh.csv (not in the last commit)",
                       "equal  same.csv"]
    assert "models: 1 identical, 0 different, 0 not regenerated" in log[3] and "identical" in log[4]
    (runs / "cvae" / "frozen" / "seed-0" / "model.pt").write_bytes(b"other weights")
    (data / "metadata.json").write_text(json.dumps({"hash": "xyz"}), encoding="utf-8")
    log.clear()
    assert check(**options) == 3
    assert "1 different" in log[3] and "DIFFERENT" in log[4] and log[-1] == "3 difference(s)"
    (tables / "changed.csv").unlink()
    (runs / "cvae" / "frozen" / "seed-0" / "model.pt").unlink()
    (data / "metadata.json").unlink()
    log.clear()
    assert check(**options) == 0  # what is not there is reported, not counted
    assert log[2] == "tables: 1 saved but not regenerated: changed.csv"
    assert "1 not regenerated" in log[3] and "not built" in log[4]
    assert log[-1] == "everything compared is reproduced exactly"


def test_figures_are_compared_byte_by_byte(tmp_path):
    from experiments.regenerate import compare_figures

    saved, new = tmp_path / "saved", tmp_path / "new"
    saved.mkdir()
    new.mkdir()
    for name in ("e2_results.png", "e1_results.png", "e7_walls.png", "e6_gap.png"):
        (saved / name).write_bytes(b"saved " + name.encode())
    (new / "e2_results.png").write_bytes(b"saved e2_results.png")
    (new / "e1_results.png").write_bytes(b"the same bars with other timings")
    (new / "e7_walls.png").write_bytes(b"changed")
    assert compare_figures(new, saved) == (["e2_results.png"], ["e1_results.png", "e7_walls.png"], ["e6_gap.png"])
    log = []
    problems = check(tables_dir=tmp_path / "no_tables", hashes=tmp_path / "none.json", frozen=tmp_path / "none.yaml",
                     show=lambda name: None, log=log.append, listed=lambda: [], figures_dir=new, saved_figures=saved)
    assert problems == 1  # e7_walls.png plots no measured time, so it must not change
    assert ("figures: 1 identical, 1 redrawn with new timings, 1 different, 1 not regenerated; "
            "different: e7_walls.png") in log


def test_saved_tables_are_the_committed_ones():
    from experiments.regenerate import committed, saved_tables

    names = saved_tables()
    assert {"baselines.csv", "e1_final.csv", "e8_final.csv", "e10_final.csv", "e12.csv", "gate2.csv"} <= set(names)
    assert names == sorted(names) and all(committed(name) is not None for name in names)


def test_saved_run_prefers_the_run_folder_then_the_committed_copy(tmp_path, monkeypatch):
    import spacegen.paths as paths

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    run, copy = tmp_path / "runs" / "cvae" / "x", tmp_path / "checkpoints" / "cvae" / "x"
    assert paths.saved_run("cvae", "x") == run  # neither exists: the caller reports the missing run
    copy.mkdir(parents=True)
    (copy / "model.pt").write_bytes(b"copy")
    assert paths.saved_run("cvae", "x") == copy  # a fresh clone has only the copy
    run.mkdir(parents=True)
    (run / "model.pt").write_bytes(b"run")
    assert paths.saved_run("cvae", "x") == run


def test_committed_checkpoints_match_the_recorded_hashes():
    import hashlib

    from experiments.regenerate import EXPORTED, HASHES
    from spacegen.paths import CHECKPOINTS_DIR

    saved = json.loads(HASHES.read_text(encoding="utf-8"))["models"]
    for run in EXPORTED:
        model = CHECKPOINTS_DIR / run / "model.pt"
        assert model.exists(), f"{run} is not in checkpoints/"
        assert hashlib.sha256(model.read_bytes()).hexdigest() == saved[run]
        record = json.loads((CHECKPOINTS_DIR / run / "run.json").read_text(encoding="utf-8"))
        assert record["dataset"].startswith("data/")  # no machine path in the committed record


def test_committed_models_give_three_valid_layouts(catalog, rules):
    """The smoke test of a fresh clone: the real frozen CVAE and evaluator, straight from checkpoints/."""
    import numpy as np

    from spacegen.paths import CHECKPOINTS_DIR
    from spacegen.pipeline import Request, generate, load_cvae_run, load_evaluator_run
    from spacegen.rules import check_layout

    cvae = load_cvae_run(CHECKPOINTS_DIR / "cvae" / "frozen" / "seed-0")
    evaluator, raster = load_evaluator_run(CHECKPOINTS_DIR / "evaluator" / "e9a")
    request = Request(5.0, 4.0, "W", 0.5, {"sofa": None, "tv_unit": None, "coffee_table": None})
    result = generate(request, catalog, rules, cvae, np.random.default_rng(0), evaluator, raster)
    assert result.candidates == 64 and result.valid >= 20  # 70% of raw samples are valid on average
    assert len(result.top) == 3 and all(check_layout(c.layout, catalog, rules).valid for c in result.top)
