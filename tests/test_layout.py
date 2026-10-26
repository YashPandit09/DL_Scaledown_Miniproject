"""Layout tests: building layouts from the catalog, changing one item, and the layout JSON."""
import copy
import dataclasses
import json

import numpy as np
import pytest

from tests.layouts import ARMCHAIR, BOOKSHELF, COFFEE_TABLE, SIDE_TABLE, SOFA
from spacegen.layout import canonicalize, is_canonical, layout_from_dict, layout_to_dict, make_layout


def test_make_layout_fills_one_row_per_slot(good_layout):
    assert good_layout.mask.all()
    np.testing.assert_allclose(good_layout.center[SOFA], [2.8, 3.55])
    np.testing.assert_allclose(good_layout.size[SOFA], [2.10, 0.90])  # default 3-seater
    assert good_layout.variant_ids[SOFA] == "sofa_3seater"
    np.testing.assert_allclose(good_layout.room, [5.0, 4.0])


def test_effective_size_follows_rotation(good_layout):
    np.testing.assert_allclose(good_layout.eff_size[SOFA], [2.10, 0.90])  # facing south
    np.testing.assert_allclose(good_layout.eff_size[BOOKSHELF], [0.30, 0.80])  # facing west


def test_absent_slots_and_variants(catalog):
    layout = make_layout(catalog, 4.0, 3.5, "S", 0.2, {"sofa": (2.0, 3.0, 2), "tv_unit": (2.0, 0.2, 0)},
                         variants={"sofa": "sofa_2seater"})
    assert layout.mask.tolist() == [True, True, False, False, False, False]
    np.testing.assert_allclose(layout.size[SOFA], [1.60, 0.90])
    np.testing.assert_array_equal(layout.size[2:], 0)
    assert layout.variant_ids[2:] == (None,) * 4


@pytest.mark.parametrize("items, variants, error", [
    ({"piano": (1, 1, 0)}, None, KeyError),
    ({"sofa": (1, 1, 4)}, None, ValueError),
    ({"sofa": (1, 1, 0)}, {"armchair": "armchair_standard"}, KeyError),
])
def test_make_layout_rejects_bad_input(catalog, items, variants, error):
    with pytest.raises(error):
        make_layout(catalog, 5.0, 4.0, "S", 0.5, items, variants)


def test_with_item_and_without_return_changed_copies(good_layout):
    moved = good_layout.with_item(ARMCHAIR, center=(1.0, 1.0), rot=0)
    np.testing.assert_allclose(moved.center[ARMCHAIR], [1.0, 1.0])
    assert moved.rot[ARMCHAIR] == 0
    np.testing.assert_allclose(good_layout.center[ARMCHAIR], [4.2, 2.6])  # original unchanged
    removed = good_layout.without(SIDE_TABLE)
    assert not removed.mask[SIDE_TABLE] and good_layout.mask[SIDE_TABLE]
    np.testing.assert_array_equal(removed.center[SIDE_TABLE], 0)  # zero-filled like any absent slot
    np.testing.assert_array_equal(removed.size[SIDE_TABLE], 0)
    assert removed.variant_ids[SIDE_TABLE] is None and good_layout.variant_ids[SIDE_TABLE] == "side_table_standard"
    with pytest.raises(ValueError):
        removed.with_item(SIDE_TABLE, center=(1.0, 1.0))


# --------------------------------------------------------------------------- canonical form

def test_canonical_form_reduces_only_symmetric_rotations(good_layout, catalog):
    layout = good_layout.with_item(COFFEE_TABLE, rot=3).with_item(SIDE_TABLE, rot=2)
    canonical = canonicalize(layout, catalog)
    assert canonical.rot.tolist() == [2, 0, 1, 3, 3, 0]  # coffee table 3 mod 2, side table 0
    assert is_canonical(canonical, catalog) and not is_canonical(layout, catalog)
    assert is_canonical(good_layout, catalog)


@pytest.mark.parametrize("rot", range(4))
def test_canonical_form_keeps_every_footprint(good_layout, catalog, rot):
    for slot in range(catalog.num_slots):
        layout = good_layout.with_item(slot, rot=rot)
        canonical = canonicalize(layout, catalog)
        np.testing.assert_array_equal(canonical.eff_size, layout.eff_size)
        np.testing.assert_array_equal(canonical.center, layout.center)


def test_canonical_form_zero_fills_absent_slots(good_layout, catalog):
    stale = dataclasses.replace(good_layout, mask=np.arange(6) != ARMCHAIR)  # row still holds the armchair
    assert not is_canonical(stale, catalog)
    canonical = canonicalize(stale, catalog)
    assert canonical.variant_ids[ARMCHAIR] is None and canonical.rot[ARMCHAIR] == 0
    np.testing.assert_array_equal(canonical.center[ARMCHAIR], 0)
    np.testing.assert_array_equal(canonical.size[ARMCHAIR], 0)


# --------------------------------------------------------------------------- layout JSON

def test_layout_json_round_trip(good_layout, catalog, rules):
    data = layout_to_dict(good_layout, catalog, rules.door.width, metrics={"valid": True})
    data = json.loads(json.dumps(data))  # through real JSON text
    assert data["room"] == {"type": "living_room", "width": 5.0, "depth": 4.0}
    assert data["door"] == {"wall": "W", "offset": 0.5, "width": 0.9}
    assert data["items"][SOFA] == {"slot": 0, "id": "sofa_3seater", "w": 2.1, "d": 0.9, "h": 0.85,
                                   "x": 2.8, "y": 3.55, "rotation": 2, "price": 28000}
    assert data["metrics"] == {"valid": True}

    back = layout_from_dict(data, catalog)
    for field in ("center", "rot", "size", "mask"):
        np.testing.assert_array_equal(getattr(back, field), getattr(good_layout, field))
    assert back.variant_ids == good_layout.variant_ids
    assert (back.width, back.depth, back.door_wall, back.door_offset) == (5.0, 4.0, "W", 0.5)


def _wrong_room(data):
    data["room"]["type"] = "bedroom"


def _wrong_size(data):
    data["items"][0]["w"] = 2.5


def _repeated_slot(data):
    data["items"].append(copy.deepcopy(data["items"][0]))


@pytest.mark.parametrize("break_json", [_wrong_room, _wrong_size, _repeated_slot])
def test_layout_json_with_mistakes_is_rejected(good_layout, catalog, rules, break_json):
    data = layout_to_dict(good_layout, catalog, rules.door.width)
    break_json(data)
    with pytest.raises(ValueError):
        layout_from_dict(data, catalog)
