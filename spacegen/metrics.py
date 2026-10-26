"""Evaluation metrics, the same for the baselines and the neural pipeline.

Per sample (score_layouts): valid (H1 to H4), summed pairwise overlap area, reachability
ratio and rule quality. Across the valid layouts of one room (diversity): the mean over
pairs of the mean over shared slots of (centre distance + 0.5 m if the canonical rotations
differ). Layouts must be in canonical form, so a coffee table turned 180 degrees or a side
table turned any way is not a difference; the living room has no interchangeable slots.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from spacegen import geometry
from spacegen.catalog import RoomCatalog
from spacegen.layout import Layout
from spacegen.quality import quality_score
from spacegen.rules import Rules, check_layout, reachability

ROTATION_PENALTY = 0.5  # meters added for a slot whose rotation differs


def score_layouts(layouts: Sequence[Layout], catalog: RoomCatalog, rules: Rules) -> pd.DataFrame:
    """One row per layout: valid, overlap (m^2, summed over pairs), reachability ratio, quality."""
    rows = []
    for layout in layouts:
        reach = reachability(layout, catalog, rules)
        result = check_layout(layout, catalog, rules, reach=reach)
        rows.append({"valid": result.valid,
                     "overlap": float(geometry.total_overlap_area(layout.center, layout.eff_size, layout.mask)),
                     "reachability": result.reachability_ratio,
                     "quality": quality_score(layout, catalog, rules, reach=reach).total})
    return pd.DataFrame(rows, columns=["valid", "overlap", "reachability", "quality"])


def layout_distance(a: Layout, b: Layout) -> float:
    """Mean over the slots present in both of centre distance plus the rotation penalty (canonical layouts)."""
    return float(_pair_distances(np.stack([a.center, b.center]), np.stack([a.rot, b.rot]),
                                 np.stack([a.mask, b.mask]))[0])


def diversity(layouts: Sequence[Layout]) -> float | None:
    """Mean layout_distance over all pairs; None for fewer than two layouts."""
    if len(layouts) < 2:
        return None
    return float(np.mean(_pair_distances(np.stack([lay.center for lay in layouts]),
                                         np.stack([lay.rot for lay in layouts]),
                                         np.stack([lay.mask for lay in layouts]))))


def _pair_distances(center: np.ndarray, rot: np.ndarray, mask: np.ndarray) -> np.ndarray:
    first, second = np.triu_indices(len(center), k=1)
    shared = mask[first] & mask[second]
    step = np.linalg.norm(center[first] - center[second], axis=-1) + ROTATION_PENALTY * (rot[first] != rot[second])
    return (step * shared).sum(axis=1) / np.maximum(shared.sum(axis=1), 1)
