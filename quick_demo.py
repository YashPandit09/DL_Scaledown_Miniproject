"""SpaceGen AI - Simplified Quick Demo for Mini-Project Presentation.

Run this script:
    python quick_demo.py
    
It generates the best 3 furniture layouts for a living room using:
1. Conditional VAE (CVAE) generator
2. Latent Space Optimization (gradient descent repair)
3. CNN Evaluator ranking
And immediately displays the resulting floorplan!
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from spacegen.catalog import load_room_catalog
from spacegen.rules import load_rules
from spacegen.pipeline import (
    Request, generate, load_cvae_run, load_evaluator_run,
    load_pipeline_config
)
from spacegen.latent_opt import load_latent_opt_config
from spacegen.viz import plot_layout, SURFACE

def main():
    print("=" * 60)
    print("  SPACEGEN AI - DEEP LEARNING LAYOUT GENERATOR")
    print("=" * 60)
    
    # 1. Define Room Specifications
    width = 5.0    # 5 meters wide
    depth = 4.0    # 4 meters deep
    door_wall = "W"  # Door on West wall
    door_offset = 0.5  # Door at the middle of the wall
    
    # Selected furniture items
    items = {
        "sofa": "sofa_3seater",
        "tv_unit": "tv_unit_large",
        "coffee_table": "coffee_table_standard",
        "armchair": None  # Let the model pick variant
    }
    
    print(f"\n[1] Room Inputs:")
    print(f"    - Dimensions: {width}m x {depth}m (Area: {width*depth:.1f} m²)")
    print(f"    - Door: {door_wall} wall at {door_offset*100:.0f}%")
    print(f"    - Furniture: {', '.join(items.keys())}")
    
    # 2. Load Deep Learning Models
    print("\n[2] Loading Deep Learning Models...")
    root = Path(__file__).resolve().parent
    cvae_path = root / "checkpoints" / "cvae" / "frozen" / "seed-0"
    evaluator_path = root / "checkpoints" / "evaluator" / "e9a"
    
    catalog = load_room_catalog("living_room")
    rules = load_rules()
    cvae = load_cvae_run(cvae_path, device="cpu")
    evaluator, raster = load_evaluator_run(evaluator_path, device="cpu")
    print("    - CVAE Decoder loaded from checkpoints/cvae/frozen/seed-0")
    print("    - CNN Evaluator loaded from checkpoints/evaluator/e9a")
    
    # 3. Generate Layouts using CVAE + Latent Optimization
    print("\n[3] Generating 64 candidates via CVAE...")
    print("    - Sampling latent vectors z ~ N(0, I)")
    print("    - Applying Latent Space Optimization (gradient descent on z)...")
    
    request = Request(
        width=width,
        depth=depth,
        door_wall=door_wall,
        door_offset=door_offset,
        items=items,
        budget=None,
        pins={}
    )
    
    rng = np.random.default_rng(42)
    pipeline_cfg = load_pipeline_config(latent_opt=True)
    latent_cfg = load_latent_opt_config()
    
    result = generate(
        request, catalog, rules, cvae, rng,
        evaluator=evaluator, raster=raster,
        config=pipeline_cfg, latent=latent_cfg,
        device="cpu"
    )
    
    print(f"\n[4] Generation Results:")
    print(f"    - Valid layouts generated: {result.valid} / {result.candidates} ({result.valid/result.candidates*100:.1f}%)")
    print(f"    - CNN Evaluator ranked and selected Top {len(result.top)} diverse layouts.")
    
    output_dir = root / "reports" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for rank, candidate in enumerate(result.top, start=1):
        score_text = f"Quality: {candidate.quality:.2f}"
        if candidate.evaluator_score is not None:
            score_text += f" | CNN Score: {candidate.evaluator_score:.2f}"
        print(f"    * Layout #{rank}: {score_text}")
        
        fig = plot_layout(
            candidate.layout, catalog, rules,
            title=f"SpaceGen AI Top #{rank} ({score_text})"
        )
        fig.savefig(output_dir / f"quick_top{rank}.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        
    print(f"\n[5] Saved floorplan visuals to: {output_dir / 'quick_top1.png'}")
    print("\nDemonstration complete! Deep Learning pipeline executed successfully.")

if __name__ == "__main__":
    main()
