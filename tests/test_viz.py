"""Plot tests: any layout JSON renders to a PNG, and broken checks are named on the item."""
import json

from matplotlib.colors import to_hex

import pandas as pd

from tests.layouts import ARMCHAIR, SOFA
from spacegen.layout import layout_to_dict
from spacegen.viz import CRITICAL, main, plot_layout, plot_raster, plot_rejection


def _labels(fig):
    return [t.get_text() for t in fig.axes[0].texts]


def _red_patches(fig):
    return [p for p in fig.axes[0].patches if to_hex(p.get_edgecolor()) == CRITICAL]


def test_layout_json_renders_to_png(tmp_path, good_layout, catalog, rules):
    source = tmp_path / "layout.json"
    source.write_text(json.dumps(layout_to_dict(good_layout, catalog, rules.door.width)), encoding="utf-8")
    target = tmp_path / "layout.png"
    assert main([str(source), str(target)]) == 0
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_valid_layout_shows_every_item_and_no_red(good_layout, catalog, rules):
    fig = plot_layout(good_layout, catalog, rules)
    assert {"sofa", "tv unit", "coffee table", "bookshelf", "armchair", "side table"} <= set(_labels(fig))
    assert "H4 reachable: pass" in fig.get_suptitle()
    assert _red_patches(fig) == []


def test_broken_check_is_named_on_the_item_and_in_the_title(good_layout, catalog, rules):
    walled = good_layout.without(ARMCHAIR).with_item(SOFA, center=(4.15, 1.5), rot=3)
    fig = plot_layout(walled, catalog, rules)
    assert "bookshelf\n(H4)" in _labels(fig)
    assert "H4 reachable: fail (bookshelf)" in fig.get_suptitle()
    assert len(_red_patches(fig)) == 1


def test_small_plan_keeps_its_labels_readable(good_layout, catalog, rules):
    full = plot_layout(good_layout, catalog, rules, title="Layout 1")
    small = plot_layout(good_layout, catalog, rules, title="Layout 1", width=3.2)
    assert small.get_figwidth() == 3.2 < full.get_figwidth() / 2  # the same font sizes on a much smaller drawing
    assert full.legends and not small.legends  # the app explains the marks once, in words
    assert small.get_suptitle() == "Layout 1" and plot_layout(good_layout, catalog, rules, width=3.2).get_suptitle() == ""
    assert {"sofa", "tv unit", "bookshelf", "armchair", "door"} <= set(_labels(small))
    assert "coffee\ntable" in _labels(small) and "coffee table" in _labels(full)  # two lines where one is too long
    door = next(t for t in small.axes[0].texts if t.get_text() == "door")
    assert 0 < door.get_position()[0] < good_layout.width  # inside the room, clear of the axis labels
    outside = next(t for t in full.axes[0].texts if t.get_text() == "door")
    assert outside.get_position()[0] < 0  # the full picture is unchanged: the label stays outside the west wall


def test_raster_plot_has_one_panel_per_channel(good_layout, catalog, rules):
    from spacegen.raster import CHANNELS, load_raster_config, rasterize_layouts
    config = load_raster_config()
    fig = plot_raster(rasterize_layouts([good_layout], catalog, rules, config)[0], config.canvas)
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert titles == [f"{c}: {name}" for c, name in enumerate(CHANNELS)]


def test_rejection_plot_shows_the_rejected_share_per_group():
    attempts = pd.DataFrame({"room": [0, 0, 1, 2], "area": [12.0, 12.0, 30.0, 30.0], "items": [6, 6, 3, 4],
                             "outcome": ["invalid", "valid", "valid", "no position"]})
    fig = plot_rejection(attempts, cap=20)
    by_area, by_items = fig.axes[:2]
    assert [round(bar.get_height()) for bar in by_area.patches] == [50, 50]  # 12-16 and 28-32 m^2
    assert [round(bar.get_height()) for bar in by_items.patches] == [0, 100, 50]  # 3, 4 and 6 items
    assert "1 of 3 rooms dropped after 20 failed attempts" in fig.get_suptitle()
