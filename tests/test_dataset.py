"""Dataset encoding tests: vector contents, canonical targets, the round trip."""
import dataclasses

import numpy as np
import pytest
import torch

from spacegen.dataset import (condition_dim, decode_targets, encode_conditions, encode_targets, minibatches,
                              split_targets, stack_layouts, target_dim, training_tensors)
from spacegen.layout import canonicalize, make_layout
from spacegen.rules import check_layout
from tests.layouts import ARMCHAIR, BOOKSHELF, COFFEE_TABLE, SIDE_TABLE, SOFA, assert_same_layout

# Catalog sizes (configs/catalog.yaml), default variants, in slot order.
WIDTHS = [2.10, 1.50, 1.00, 0.80, 0.80, 0.45]
DEPTHS = [0.90, 0.40, 0.55, 0.30, 0.80, 0.45]


@pytest.fixture
def layouts(good_layout, catalog):
    """The hand-built room; the same with its symmetric items stored the long way round; a sparse room."""
    turned = good_layout.with_item(COFFEE_TABLE, rot=2).with_item(SIDE_TABLE, rot=3)
    sparse = make_layout(catalog, 4.0, 3.5, "S", 0.2, {"sofa": (2.0, 3.0, 2), "tv_unit": (2.0, 0.2, 0)},
                         variants={"sofa": "sofa_2seater"})
    return [good_layout, turned, sparse]


def _encode(layouts, catalog, dtype=np.float64):
    batch = stack_layouts(layouts, catalog)
    return encode_conditions(batch, catalog, dtype), encode_targets(batch, catalog, dtype)


def test_vector_sizes_match_the_tech_spec(catalog, good_layout):
    c, x = _encode([good_layout], catalog, np.float32)
    assert (condition_dim(catalog.num_slots), target_dim(catalog.num_slots)) == (25, 36)
    assert c.shape == (1, 25) and x.shape == (1, 36) and x.dtype == np.float32


def test_condition_vector(catalog, layouts):
    c, _ = _encode(layouts, catalog)
    np.testing.assert_allclose(c[0, :7], [5.0 / 8, 4.0 / 8, 0, 0, 0, 1, 0.5])  # door on the west wall
    np.testing.assert_array_equal(c[0, 7:13], 1)
    np.testing.assert_allclose(c[0, 13:19], np.array(WIDTHS) / 3)
    np.testing.assert_allclose(c[0, 19:25], np.array(DEPTHS) / 3)
    # sparse room: door on the south wall, a 2-seater and the TV unit, zeros for the absent slots
    np.testing.assert_allclose(c[2, :7], [4.0 / 8, 3.5 / 8, 0, 0, 1, 0, 0.2])
    np.testing.assert_array_equal(c[2, 7:13], [1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(c[2, 13:19], [1.60 / 3, 1.50 / 3, 0, 0, 0, 0])
    np.testing.assert_allclose(c[2, 19:25], [0.90 / 3, 0.40 / 3, 0, 0, 0, 0])


def test_condition_uses_catalog_sizes_whatever_the_rotation(catalog, good_layout):
    c, _ = _encode([good_layout, good_layout.with_item(BOOKSHELF, rot=0)], catalog)
    np.testing.assert_array_equal(c[0], c[1])  # the rotation is a target, not a condition


def test_target_vector(catalog, good_layout):
    _, x = _encode([good_layout], catalog)
    per_slot = x[0].reshape(6, 6)
    np.testing.assert_allclose(per_slot[SOFA], [2.8 / 5, 3.55 / 4, 0, 0, 1, 0])  # facing south
    np.testing.assert_allclose(per_slot[BOOKSHELF], [4.85 / 5, 1.5 / 4, 0, 0, 0, 1])  # facing west


def test_absent_slots_are_zero_filled(catalog, good_layout, layouts):
    _, x = _encode(layouts, catalog)
    np.testing.assert_array_equal(x[2].reshape(6, 6)[2:], 0)
    # an absent slot whose row still holds an old position encodes as zeros too
    stale = dataclasses.replace(good_layout, mask=np.arange(6) != ARMCHAIR)
    c, x = _encode([stale], catalog)
    np.testing.assert_array_equal(x[0].reshape(6, 6)[ARMCHAIR], 0)
    assert c[0, 7 + ARMCHAIR] == c[0, 13 + ARMCHAIR] == c[0, 19 + ARMCHAIR] == 0


@pytest.mark.parametrize("slot, rotations", [
    (COFFEE_TABLE, [0, 2]),  # front and back look alike
    (COFFEE_TABLE, [1, 3]),
    (SIDE_TABLE, [0, 1, 2, 3]),  # every side looks alike
])
def test_equivalent_rotations_give_one_target(catalog, good_layout, slot, rotations):
    targets = [_encode([good_layout.with_item(slot, rot=r)], catalog)[1] for r in rotations]
    for target in targets[1:]:
        np.testing.assert_array_equal(target, targets[0])


@pytest.mark.parametrize("slot, a, b", [(SOFA, 2, 0), (COFFEE_TABLE, 0, 1), (ARMCHAIR, 3, 1)])
def test_distinct_rotations_give_distinct_targets(catalog, good_layout, slot, a, b):
    _, first = _encode([good_layout.with_item(slot, rot=a)], catalog)
    _, second = _encode([good_layout.with_item(slot, rot=b)], catalog)
    assert not np.array_equal(first, second)


# --------------------------------------------------------------------------- decoding

def _conditions_only(batch):
    """The batch with its centres and rotations wiped, as the pipeline has it before decoding."""
    return dataclasses.replace(batch, center=np.zeros_like(batch.center), rot=np.zeros_like(batch.rot))


@pytest.mark.parametrize("dtype, atol", [(np.float64, 1e-12), (np.float32, 1e-5)])
def test_round_trip_gives_the_canonical_layout(catalog, layouts, dtype, atol):
    batch = stack_layouts(layouts, catalog)
    uv, rot_scores = split_targets(encode_targets(batch, catalog, dtype))
    decoded = decode_targets(uv, rot_scores, _conditions_only(batch), catalog)
    for i, layout in enumerate(layouts):
        assert_same_layout(decoded.layout(i, catalog), canonicalize(layout, catalog), atol)


def test_round_trip_through_float32_keeps_the_check_results(catalog, rules, layouts):
    batch = stack_layouts(layouts, catalog)
    decoded = decode_targets(*split_targets(encode_targets(batch, catalog)), batch, catalog)
    for i, layout in enumerate(layouts):
        assert check_layout(decoded.layout(i, catalog), catalog, rules) == check_layout(layout, catalog, rules)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA GPU"))])
def test_decoding_logits_takes_the_likeliest_rotation_in_canonical_form(catalog, good_layout, device):
    batch = stack_layouts([good_layout], catalog)
    uv, _ = split_targets(torch.as_tensor(encode_targets(batch, catalog), device=device))
    logits = torch.zeros(1, 6, 4, device=device, requires_grad=True)
    logits = logits + torch.tensor([0.0, 0.0, 0.0, 2.0], device=device)  # every item most likely faces West
    decoded = decode_targets(uv, logits, batch, catalog)
    assert decoded.rot[0].tolist() == [3, 3, 1, 3, 3, 0]  # coffee table 3 mod 2, side table 0
    np.testing.assert_allclose(decoded.center, batch.center, atol=1e-5)


# --------------------------------------------------------------------------- the batch

def test_batch_rows_sizes_and_subsets(catalog, layouts):
    batch = stack_layouts(layouts, catalog)
    part = batch.subset([2, 0])
    assert len(part) == 2 and part.door_wall.tolist() == [2, 3]  # S, W
    assert part.variant[0].tolist() == [1, 0, -1, -1, -1, -1]  # the 2-seater is the sofa's second variant
    np.testing.assert_allclose(part.sizes(catalog)[0, :2], [[1.60, 0.90], [1.50, 0.40]])
    np.testing.assert_array_equal(part.sizes(catalog)[0, 2:], 0)
    np.testing.assert_allclose(part.sizes(catalog)[1], np.stack([WIDTHS, DEPTHS], axis=1))
    assert_same_layout(part.layout(1, catalog), layouts[0])


def test_batch_is_stored_in_canonical_form(catalog, layouts):
    batch = stack_layouts(layouts, catalog)
    assert batch.rot[1, COFFEE_TABLE] == 0 and batch.rot[1, SIDE_TABLE] == 0


def test_batch_rejects_mismatched_shapes(catalog, layouts):
    batch = stack_layouts(layouts, catalog)
    with pytest.raises(ValueError, match="center"):
        dataclasses.replace(batch, center=batch.center[:, :5])
    with pytest.raises(ValueError, match="no layouts"):
        stack_layouts([], catalog)


# --------------------------------------------------------------------------- tensors for training

def test_training_tensors(catalog, layouts):
    batch = stack_layouts(layouts, catalog)
    data = training_tensors(batch, catalog)
    assert len(data) == 3 and data.x.shape == (3, 36) and data.c.shape == (3, 25)
    assert data.x.dtype == data.c.dtype == data.mask.dtype == torch.float32
    np.testing.assert_array_equal(data.x.numpy(), encode_targets(batch, catalog))
    assert data.mask[2].tolist() == [1, 1, 0, 0, 0, 0]
    assert data.rot_mask[0].tolist() == [1, 1, 1, 1, 1, 0]  # the side table's rotation is not learned
    assert data.rot_mask[2].tolist() == [1, 1, 0, 0, 0, 0]
    rows = data[torch.tensor([2, 0])]
    assert torch.equal(rows.x[1], data.x[0]) and torch.equal(rows.c[0], data.c[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_training_tensors_on_the_gpu(catalog, layouts):
    data = training_tensors(stack_layouts(layouts, catalog), catalog, device="cuda")
    rows = data[next(minibatches(len(data), 2, device="cuda"))]
    assert rows.x.is_cuda and rows.c.is_cuda and rows.x.shape == (2, 36)


def test_minibatches_cover_every_row_once_in_a_seeded_order():
    def epoch(seed, **options):
        return [b.tolist() for b in minibatches(10, 4, torch.Generator().manual_seed(seed), **options)]

    batches = epoch(0)
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(sum(batches, [])) == list(range(10))
    assert epoch(0) == batches and epoch(1) != batches
    assert [len(b) for b in epoch(0, drop_last=True)] == [4, 4]
