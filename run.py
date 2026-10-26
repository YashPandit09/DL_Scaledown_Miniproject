"""Project task runner: python run.py <task> [extra arguments for the task's command].

Windows has no `make`, so every reproducible step is a task here and the Makefile only
forwards to this script. Tasks are added as the development plan progresses (data, train,
experiments, figures, all).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

TASKS: dict[str, tuple[str, list[list[str]]]] = {
    "test": ("run the unit tests", [[PY, "-m", "pytest", "-q"]]),
    "check-env": (
        "versions, GPU and VRAM, determinism and evaluator-sized training speed",
        [[PY, "-m", "spacegen.env_check"]],
    ),
    "generator-report": (
        "generate Set A layouts and plot how often attempts are rejected",
        [[PY, "-m", "spacegen.generator"]],
    ),
    "set-b-report": (
        "generate Set B layouts and print their labels per perturbation type",
        [[PY, "-m", "spacegen.perturb"]],
    ),
    "data": (
        "build the dataset in data/<version>/ (Set A, Set B, splits, held-out sets, f_max) and its figures",
        [[PY, "-m", "spacegen.build_dataset"]],
    ),
    "baselines": (
        "evaluate B1, B2 and G0 on the dataset; writes reports/tables/baselines.csv",
        [[PY, "-m", "spacegen.evaluate"]],
    ),
    "train-evaluator": (
        "train the CNN evaluator on Set B (default one epoch; --epochs N --name RUN)",
        [[PY, "-m", "spacegen.train_evaluator"]],
    ),
    "train-cvae": (
        "train the CVAE on Set A (--name RUN, --set cvae.<field>=<value> to override the config)",
        [[PY, "-m", "spacegen.train_cvae"]],
    ),
    "evaluator-report": (
        "metrics of a trained evaluator on the Set B test split (E9a; --run NAME, default e9a)",
        [[PY, "-m", "spacegen.evaluator_report"]],
    ),
    "generate": (
        "the top layouts for one room, e.g. --width 5 --depth 4 --door W --items sofa,tv_unit,coffee_table",
        [[PY, "-m", "spacegen.pipeline"]],
    ),
    "screen": (
        "CVAE screening experiments: e2 to e8, or all (trains missing runs, writes tables and figures)",
        [[PY, "-m", "experiments.screening"]],
    ),
    "e1": (
        "E1: B1, B2, G0, M1 and M2 on the same 500 test rooms (--cvae RUN [RUN ...] --tag NAME, --fresh)",
        [[PY, "-m", "experiments.headline", "e1"]],
    ),
    "e10": (
        "E10: M1, M2 and G0 on the in-distribution and the three held-out test sets (--sets to split the run)",
        [[PY, "-m", "experiments.headline", "e10"]],
    ),
    "e8": (
        "E8 on one or several CVAE runs (the T33b headline; the screening's pass is `screen e8`)",
        [[PY, "-m", "experiments.headline", "e8"]],
    ),
    "gate2": (
        "Gate 2 freeze: 3 seeds of the shortlisted CVAE settings on validation rooms; writes configs/frozen.yaml",
        [[PY, "-m", "experiments.gate2"]],
    ),
    "e12": (
        "E12: completion around a pinned item, per item: M2, M1, G0-pin, B1 and B2 (--cvae RUN ..., --rooms N)",
        [[PY, "-m", "experiments.pinned"]],
    ),
    "failures": (
        "failure-case analysis: what breaks, how often, and a gallery (reports/failure_cases.md discusses it)",
        [[PY, "-m", "experiments.failure_cases"]],
    ),
    "figures": (
        "redraw every figure from the saved tables and logs, in seconds (--tables rebuilds the screening tables)",
        [[PY, "-m", "experiments.make_figures"]],
    ),
    "all": (
        "regenerate every result in order, about eight hours (--list, --from STEP, --to STEP; see experiments/regenerate.py)",
        [[PY, "-m", "experiments.regenerate", "all"]],
    ),
    "check-regeneration": (
        "compare regenerated tables, models and the dataset with the saved ones (--record saves the hashes)",
        [[PY, "-m", "experiments.regenerate", "check"]],
    ),
    "export-checkpoints": (
        "copy the final models (frozen CVAE seeds, evaluator) from runs/ to checkpoints/, which is committed",
        [[PY, "-m", "experiments.regenerate", "export"]],
    ),
    "demo-assets": (
        "the demo's backup pictures and the report's demo figures, in reports/demo/",
        [[PY, "-m", "experiments.demo_assets"]],
    ),
    "pdf": (
        "the report, appendix, failure cases, viva notes and demo script as PDF files in reports/pdf/ (needs Edge or Chrome)",
        [[PY, "-m", "experiments.report_pdf"]],
    ),
    "app": (
        "the Streamlit demo: top 3 layouts, the five methods compared, the saved figures",
        [[PY, "-m", "streamlit", "run", "app/streamlit_app.py"]],
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="tasks:\n" + "\n".join(f"  {name:17s} {help}" for name, (help, _) in TASKS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="passed on to the task's last command")
    args = parser.parse_args(argv)

    commands = TASKS[args.task][1]
    for i, command in enumerate(commands):
        if i == len(commands) - 1:
            command = command + args.extra
        returncode = subprocess.run(command, cwd=ROOT).returncode
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
