"""Notebook test: the manual-backprop notebook's checks pass when its code runs, without Jupyter."""
import json

import torch

from spacegen.paths import REPO_ROOT

NOTEBOOK = REPO_ROOT / "notebooks" / "01_manual_backprop.ipynb"


def test_manual_backprop_matches_autograd_and_finite_differences():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = "\n\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    namespace = {}
    dtype = torch.get_default_dtype()
    try:
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)  # the notebook asserts its own checks too
    finally:
        torch.set_default_dtype(dtype)  # the notebook switches to float64
    assert namespace["max_difference"] < 1e-6 and namespace["fd_error"] < 1e-6


def test_the_saved_notebook_shows_the_passing_check():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    printed = "".join("".join(output.get("text", "")) for cell in cells if cell["cell_type"] == "code"
                      for output in cell.get("outputs", []))
    assert "PASS: largest difference" in printed
