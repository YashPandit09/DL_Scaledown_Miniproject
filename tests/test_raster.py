"""Raster tests: exact coverage, overlaps, front bands, GPU and gradients."""
import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from spacegen import geometry
from spacegen.raster import RasterConfig, load_raster_config, rasterize, rasterize_layouts
from tests.layouts import SIDE_TABLE

CONFIG = load_raster_config()
ROOM = (5.0, 4.0)


def _raster(items, sym=None, room=ROOM, zone=((0.45, 2.0), (0.9, 0.9)), config=CONFIG,
            dtype=torch.float64, device="cpu"):
    """Rasterize one layout given as [(x, y, w, d, rot), ...]; returns (4, P, P)."""
    def t(values, kind=dtype):
        return torch.tensor(values, dtype=kind, device=device)

    k = len(items)
    return rasterize(
        t([[[x, y] for x, y, *_ in items]]), t([[[w, d] for _, _, w, d, _ in items]]),
        t([[r for *_, r in items]], torch.long), t([[True] * k], torch.bool),
        t([room]), t([zone[0]]), t([zone[1]]), t(sym or [1] * k, torch.long), config)[0]


def _exact(center, size, config=CONFIG):
    """Reference: each pixel's overlap with the box, computed pixel by pixel with geometry."""
    centres = (np.arange(config.pixels) + 0.5) * config.pixel
    px, py = np.meshgrid(centres, centres, indexing="ij")
    pixels = np.stack([px, py], axis=-1)
    return geometry.overlap_area(pixels, np.full(2, config.pixel), np.asarray(center, float),
                                 np.asarray(size, float)) / config.pixel ** 2


def _supersampled(center, size, n=32, config=CONFIG):
    """Reference: the share of n x n sample points per pixel that fall inside the box."""
    samples = (np.arange(config.pixels * n) + 0.5) * config.pixel / n
    inside_x = np.abs(samples - center[0]) <= size[0] / 2
    inside_y = np.abs(samples - center[1]) <= size[1] / 2
    fine = inside_x[:, None] & inside_y[None, :]
    return fine.reshape(config.pixels, n, config.pixels, n).mean(axis=(1, 3))


@pytest.mark.parametrize("x, y, w, d, rot", [
    (2.8, 3.55, 2.10, 0.90, 2),  # sofa, edges between pixels
    (1.234, 0.987, 0.45, 0.45, 0),  # side table, arbitrary sub-pixel position
    (4.85, 1.5, 0.80, 0.30, 3),  # bookshelf turned east-west
    (7.9, 7.9, 0.8, 0.8, 0),  # partly off the canvas: only the part on it counts
])
def test_furniture_matches_the_pixel_by_pixel_reference(x, y, w, d, rot):
    eff = geometry.effective_size(np.array([w, d]), rot)
    furniture = _raster([(x, y, w, d, rot)])[1].numpy()
    np.testing.assert_allclose(furniture, _exact((x, y), eff), atol=1e-9)
    np.testing.assert_allclose(furniture, _supersampled((x, y), eff), atol=0.05)


def test_summed_coverage_is_the_footprint_area():
    raster = _raster([(2.8, 3.55, 2.10, 0.90, 2), (1.234, 0.987, 0.45, 0.45, 0)])
    areas = raster.sum(dim=(1, 2)).numpy() * CONFIG.pixel ** 2
    np.testing.assert_allclose(areas[[0, 1, 3]], [5.0 * 4.0, 2.10 * 0.90 + 0.45 ** 2, 0.9 * 0.9])


def test_overlap_shows_as_values_above_one():
    alone = _raster([(2.0, 2.0, 1.0, 1.0, 0)])[1]
    both = _raster([(2.0, 2.0, 1.0, 1.0, 0), (2.25, 2.0, 1.0, 1.0, 0)])[1]
    assert alone.max() == pytest.approx(1.0)
    assert both.max() == pytest.approx(2.0)  # pixels covered by both items


def test_a_one_centimetre_shift_changes_the_pixels_but_not_the_area():
    before = _raster([(2.0, 2.0, 1.0, 1.0, 0)])[1]
    after = _raster([(2.01, 2.0, 1.0, 1.0, 0)])[1]
    assert (before - after).abs().max() > 0.1
    assert after.sum() == pytest.approx(before.sum())


def test_front_band_runs_along_the_front_face():
    fronts = _raster([(2.8, 2.0, 2.10, 0.90, 0)])[2].numpy()  # sofa facing North: front at y = 2.45
    centres = (np.arange(CONFIG.pixels) + 0.5) * CONFIG.pixel
    assert fronts.sum() * CONFIG.pixel ** 2 == pytest.approx(2.10 * CONFIG.front_strip)
    mean_y = (fronts.sum(axis=0) * centres).sum() / fronts.sum()  # weighted by pixel centres
    assert mean_y == pytest.approx(2.45 - CONFIG.front_strip / 2, abs=CONFIG.pixel / 2)


@pytest.mark.parametrize("name, w, d, sym, bands", [
    ("armchair", 0.80, 0.80, 1, 0.80 * 0.1),  # one real front
    ("coffee table", 1.00, 0.55, 2, 2 * 1.00 * 0.1),  # front and back look alike
    ("side table", 0.45, 0.45, 4, 4 * 0.45 * 0.1),  # all four sides look alike
])
def test_symmetric_items_get_a_band_on_every_equivalent_face(name, w, d, sym, bands):
    fronts = _raster([(2.0, 2.0, w, d, 0)], sym=[sym])[2]
    assert fronts.sum().item() * CONFIG.pixel ** 2 == pytest.approx(bands)


def test_absent_items_leave_no_trace(good_layout, catalog, rules):
    with_table = rasterize_layouts([good_layout], catalog, rules, CONFIG, dtype=torch.float64)[0]
    without = rasterize_layouts([good_layout.without(SIDE_TABLE)], catalog, rules, CONFIG, dtype=torch.float64)[0]
    removed = (with_table - without)[1].sum().item() * CONFIG.pixel ** 2
    assert removed == pytest.approx(0.45 * 0.45)


def test_layouts_rasterize_with_the_room_and_door_in_place(good_layout, catalog, rules):
    raster = rasterize_layouts([good_layout, good_layout], catalog, rules, CONFIG)
    assert raster.shape == (2, 4, CONFIG.pixels, CONFIG.pixels) and raster.dtype == torch.float32
    room, door = raster[0, 0], raster[0, 3]
    end_x, end_y = round(5.0 / CONFIG.pixel), round(4.0 / CONFIG.pixel)  # pixels 80 and 64
    assert room[:end_x, :end_y].min() == 1.0
    assert room[end_x:, :].max() == 0.0 and room[:, end_y:].max() == 0.0
    assert door.sum().item() * CONFIG.pixel ** 2 == pytest.approx(0.81, rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_gpu_gives_the_same_raster():
    items = [(2.8, 3.55, 2.10, 0.90, 2), (1.234, 0.987, 0.45, 0.45, 0)]
    on_gpu = _raster(items, sym=[1, 4], device="cuda")
    torch.testing.assert_close(on_gpu.cpu(), _raster(items, sym=[1, 4]))


def test_raster_is_differentiable_in_the_item_positions():
    small = RasterConfig(canvas=8.0, pixels=16, front_strip=0.1)
    size = torch.tensor([[[1.0, 0.55], [0.45, 0.45]]], dtype=torch.float64)
    weights = torch.rand(4, 16, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    def weighted_raster(center):
        raster = rasterize(center, size, torch.tensor([[0, 1]]), torch.tensor([[True, True]]),
                           torch.tensor([[5.0, 4.0]], dtype=torch.float64),
                           torch.tensor([[0.45, 2.0]], dtype=torch.float64),
                           torch.tensor([[0.9, 0.9]], dtype=torch.float64), torch.tensor([2, 4]), small)
        return (raster[0] * weights).sum()

    # positions chosen so that no box edge sits exactly on a pixel edge (where the slope jumps)
    center = torch.tensor([[[2.13, 1.71], [3.37, 2.94]]], dtype=torch.float64, requires_grad=True)
    assert gradcheck(weighted_raster, (center,))
