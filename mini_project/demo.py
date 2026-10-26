"""
SpaceGen AI - Inference Demo (Simplified for Mini Project)

Shows the complete 3-step pipeline:
  Step 1:  Sample z ~ N(0, I)  ->  CVAE Decoder  ->  64 candidate layouts
  Step 2:  Latent Optimization  ->  Gradient descent on z to fix overlaps
  Step 3:  CNN Evaluator ranks candidates  ->  Pick Top 3 diverse layouts

Uses the REAL pre-trained checkpoints from checkpoints/cvae and checkpoints/evaluator.
Run from the project root: python mini_project/demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from spacegen.catalog import load_room_catalog
from spacegen.rules import load_rules
from spacegen.pipeline import (
    Request, generate, load_cvae_run, load_evaluator_run, load_pipeline_config
)
from spacegen.latent_opt import load_latent_opt_config
from spacegen.viz import plot_layout, SURFACE


def run_demo():
    print("=" * 58)
    print("  SpaceGen AI — Deep Learning Inference Demo")
    print("=" * 58)

    # ─── Room specification ───────────────────────────────────────
    room_width    = 5.0    # metres
    room_depth    = 4.0    # metres
    door_wall     = "W"    # door on the West wall
    door_offset   = 0.5   # door at 50% along the wall

    # Furniture items (name -> variant or None to auto-pick)
    items = {
        "sofa":         "sofa_3seater",
        "tv_unit":      "tv_unit_large",
        "coffee_table": "coffee_table_standard",
    }

    print(f"\n[1] Room specification")
    print(f"    Width={room_width}m  Depth={room_depth}m  Door={door_wall} wall")
    print(f"    Furniture: {', '.join(items)}")

    # ─── Load pre-trained models ──────────────────────────────────
    root         = Path(__file__).resolve().parent.parent
    cvae_path    = root / "checkpoints" / "cvae" / "frozen" / "seed-0"
    eval_path    = root / "checkpoints" / "evaluator" / "e9a"

    print("\n[2] Loading pre-trained Deep Learning models")
    catalog  = load_room_catalog("living_room")
    rules    = load_rules()
    cvae     = load_cvae_run(cvae_path, device="cpu")
    evaluator, raster = load_evaluator_run(eval_path, device="cpu")
    print("    [OK] CVAE loaded  (checkpoints/cvae/frozen/seed-0/model.pt)")
    print("    [OK] CNN Evaluator loaded  (checkpoints/evaluator/e9a/model.pt)")

    # ─── Step 1 + 2 + 3: Run the full pipeline ───────────────────
    print("\n[3] Running the 3-step pipeline  (64 candidates, CPU)...")
    t0 = time.perf_counter()

    request = Request(
        width=room_width, depth=room_depth,
        door_wall=door_wall, door_offset=door_offset,
        items=items, budget=None, pins={}
    )

    result = generate(
        request, catalog, rules, cvae,
        rng=np.random.default_rng(42),
        evaluator=evaluator, raster=raster,
        config=load_pipeline_config(latent_opt=True),
        latent=load_latent_opt_config(),
        device="cpu",
    )

    elapsed = time.perf_counter() - t0

    # ─── Results ──────────────────────────────────────────────────
    print(f"\n[4] Results  ({elapsed:.2f}s total)")
    print(f"    Candidates generated : {result.candidates}")
    print(f"    Valid after checks   : {result.valid}  "
          f"({result.valid / result.candidates * 100:.1f}%)")
    print(f"    Top layouts shown    : {len(result.top)}")
    print()
    for rank, c in enumerate(result.top, 1):
        cnn_str = f"  CNN score={c.evaluator_score:.3f}" if c.evaluator_score else ""
        print(f"    Rank #{rank}  quality={c.quality:.3f}{cnn_str}")

    # ─── Save floorplan images ────────────────────────────────────
    out_dir = root / "mini_project" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[5] Saving floorplan images to mini_project/output/")
    for rank, c in enumerate(result.top, 1):
        title = (f"Rank #{rank}  quality={c.quality:.2f}"
                 + (f"  CNN={c.evaluator_score:.2f}" if c.evaluator_score else ""))
        fig = plot_layout(c.layout, catalog, rules, title=title)
        path = out_dir / f"layout_rank{rank}.png"
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"    Saved: {path.name}")

    print("\nDone! Open mini_project/output/layout_rank1.png to see the best layout.")


if __name__ == "__main__":
    run_demo()
