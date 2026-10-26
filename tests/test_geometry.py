"""Geometry tests: analytic overlaps, NumPy vs PyTorch, loss gradients."""
import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from spacegen import geometry as g


def t(values):
    """A float64 tensor, the precision gradcheck needs."""
    return torch.tensor(values, dtype=torch.float64)


def random_layouts(batch=16, k=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 4.5, (batch, k, 2)), rng.uniform(0.3, 2.0, (batch, k, 2))


# Sofa, a side table inside the sofa's footprint, a coffee table overlapping the sofa, and
# an armchair clear of everything. No pair sits on a kink of ReLU or min, so finite
# differences are valid.
LAYOUT_CENTER = [[2.0, 1.0], [2.3, 1.1], [3.3, 1.6], [5.0, 3.0]]
LAYOUT_SIZE = [[2.1, 0.9], [0.45, 0.45], [1.0, 0.55], [0.8, 0.8]]


# --------------------------------------------------------------------------- rotation

def test_effective_size_swaps_width_and_depth_for_east_and_west():
    size = np.array([[2.1, 0.9]] * 4)
    eff = g.effective_size(size, np.arange(4))
    np.testing.assert_allclose(eff, [[2.1, 0.9], [0.9, 2.1], [2.1, 0.9], [0.9, 2.1]])


def test_facing_vectors_follow_the_rotation_classes():
    np.testing.assert_allclose(g.facing_vector(np.arange(4), np.zeros(2)),
                               [[0, 1], [1, 0], [0, -1], [-1, 0]])


@pytest.mark.parametrize("rot, expected", [
    (0, [3.0, 1.8]),  # facing North: front face at y = 1.45, point 0.35 m beyond it
    (1, [3.8, 1.0]),  # facing East: effective size (0.9, 2.1), front face at x = 3.45
    (2, [3.0, 0.2]),
    (3, [2.2, 1.0]),
])
def test_front_point_lies_in_front_of_the_front_face(rot, expected):
    center = np.array([3.0, 1.0])
    eff = g.effective_size(np.array([2.1, 0.9]), rot)
    np.testing.assert_allclose(g.front_point(center, eff, rot, offset=0.35), expected)


@pytest.mark.parametrize("rot_symmetry, rot, expected", [
    (1, 3, [3]),
    (2, 1, [1, 3]),
    (2, 2, [2, 0]),
    (4, 2, [2, 3, 0, 1]),
])
def test_equivalent_rotations(rot_symmetry, rot, expected):
    assert g.equivalent_rotations(rot, rot_symmetry) == expected


def test_equivalent_rotations_rejects_other_symmetries():
    with pytest.raises(ValueError):
        g.equivalent_rotations(0, 3)


def test_canonical_rotation_picks_one_class_per_equivalent_set():
    rot = np.arange(4)
    np.testing.assert_array_equal(g.canonical_rotation(rot, 1), [0, 1, 2, 3])
    np.testing.assert_array_equal(g.canonical_rotation(rot, 2), [0, 1, 0, 1])  # coffee table
    np.testing.assert_array_equal(g.canonical_rotation(rot, 4), [0, 0, 0, 0])  # side table
    for sym in (1, 2, 4):
        for r in range(4):
            assert g.canonical_rotation(r, sym) in g.equivalent_rotations(r, sym)


# --------------------------------------------------------------------------- overlap area

@pytest.mark.parametrize("c1, s1, c2, s2, area", [
    ((0, 0), (2, 2), (1, 1), (2, 2), 1.0),      # corners overlap: [0, 1] x [0, 1]
    ((0, 0), (2, 1), (1, 0), (2, 1), 1.0),      # half overlap along x: [0, 1] x [-0.5, 0.5]
    ((0, 0), (2, 2), (3, 0), (2, 2), 0.0),      # separated
    ((0, 0), (2, 2), (2, 0), (2, 2), 0.0),      # edges touch
    ((0, 0), (4, 4), (0.5, 0.5), (1, 1), 1.0),  # contained: the smaller box's area
])
def test_overlap_area_of_known_boxes(c1, s1, c2, s2, area):
    args = [np.array(v, dtype=float) for v in (c1, s1, c2, s2)]
    assert g.overlap_area(*args) == pytest.approx(area)
    assert g.overlap_area(*args[2:], *args[:2]) == pytest.approx(area)


def test_pairwise_overlap_is_symmetric_with_zero_diagonal():
    center, size = random_layouts()
    ov = g.pairwise_overlap_area(center, size)
    assert ov.shape == (16, 6, 6)
    np.testing.assert_allclose(ov, np.swapaxes(ov, -1, -2))
    np.testing.assert_array_equal(np.diagonal(ov, axis1=-2, axis2=-1), 0)
    assert (ov > 0).any()


def test_absent_items_do_not_count():
    center = np.array([[1.0, 1.0], [1.2, 1.1], [3.0, 3.0]])
    size = np.array([[1.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    mask = np.array([True, False, True])
    assert g.total_overlap_area(center, size) > 0  # items 0 and 1 overlap
    assert g.total_overlap_area(center, size, mask) == 0
    assert g.overlap_penalty(center, size, margin=0.05, mask=mask) == 0


def test_numpy_and_torch_give_the_same_results():
    center, size = random_layouts()
    rng = np.random.default_rng(1)
    rot = rng.integers(0, 4, (16, 6))
    mask = rng.random((16, 6)) > 0.3
    room = np.tile([5.0, 4.0], (16, 1))
    cases = [
        (g.effective_size, (size, rot)),
        (g.front_point, (center, size, rot, 0.35)),
        (g.pairwise_overlap_area, (center, size, mask)),
        (g.total_overlap_area, (center, size, mask)),
        (g.overlap_penalty, (center, size, 0.05, mask)),
        (g.out_of_room, (center, size, room)),
        (g.inside_room, (center, size, room, 1e-3)),
    ]
    for fn, args in cases:
        expected = fn(*args)
        got = fn(*[torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in args]).numpy()
        if expected.dtype == bool:
            np.testing.assert_array_equal(got, expected, err_msg=fn.__name__)
        else:
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12, err_msg=fn.__name__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_gpu_tensors_give_the_same_penalty():
    center, size = t(LAYOUT_CENTER), t(LAYOUT_SIZE)
    on_gpu = g.overlap_penalty(center.cuda(), size.cuda(), margin=0.05)
    assert on_gpu.device.type == "cuda"
    assert on_gpu.item() == pytest.approx(g.overlap_penalty(center, size, margin=0.05).item())


# --------------------------------------------------------------------------- penetration depth

def test_penetration_depth_is_the_smaller_axis_overlap():
    # [-1, 1] x [-1, 1] against [0.5, 2.5] x [-0.8, 1.2]: overlap 0.5 along x, 1.8 along y
    depth = g.penetration_depth(np.zeros(2), np.full(2, 2.0), np.array([1.5, 0.2]), np.full(2, 2.0))
    assert depth == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("gap, expected", [(0.10, 0.0), (0.03, 0.02), (-0.04, 0.09)])
def test_margin_penalises_near_misses(gap, expected):
    # two 1 m boxes side by side along x; a negative gap is an overlap; margin 0.05 m
    depth = g.penetration_depth(np.zeros(2), np.ones(2), np.array([1.0 + gap, 0.0]), np.ones(2),
                                margin=0.05)
    assert depth == pytest.approx(expected, abs=1e-5)


def test_contained_item_gets_a_push_that_the_exact_area_cannot_give():
    """A side table fully inside the sofa's footprint."""
    sofa_c, sofa_s, table_s = t([2.0, 1.0]), t([2.1, 0.9]), t([0.45, 0.45])
    table_c = t([2.3, 1.1]).requires_grad_()

    area = g.overlap_area(table_c, table_s, sofa_c, sofa_s)
    assert area.item() == pytest.approx(0.45 * 0.45)
    (area_grad,) = torch.autograd.grad(area, table_c)
    assert torch.all(area_grad == 0)  # the area gives no direction to move

    center, size = torch.stack([sofa_c, table_c]), torch.stack([sofa_s, table_s])
    (pen_grad,) = torch.autograd.grad(g.overlap_penalty(center, size, margin=0.05), table_c)
    # the depth is smallest along y, so the push is along y, away from the sofa's centre
    assert pen_grad[0] == 0
    assert pen_grad[1] < 0


def test_coincident_centres_give_a_finite_zero_gradient():
    center = t([[1.0, 1.0], [1.0, 1.0]]).requires_grad_()
    penalty = g.overlap_penalty(center, t([[0.8, 0.8], [0.45, 0.45]]), margin=0.05)
    (grad,) = torch.autograd.grad(penalty, center)
    assert torch.isfinite(grad).all()
    assert torch.all(grad == 0)


def test_overlap_penalty_gradient_matches_finite_differences():
    size = t(LAYOUT_SIZE)
    center = t(LAYOUT_CENTER).requires_grad_()
    assert gradcheck(lambda c: g.overlap_penalty(c, size, margin=0.05), (center,))


# --------------------------------------------------------------------------- room

def test_out_of_room_measures_protrusion_per_wall():
    center = np.array([[2.5, 2.0], [0.3, 2.0], [4.9, 3.9]])
    size = np.array([[1.0, 1.0], [1.0, 1.0], [0.4, 0.4]])
    np.testing.assert_allclose(g.out_of_room(center, size, [5.0, 4.0]), [0.0, 0.2, 0.2])


def test_out_of_room_gradient_is_zero_inside_and_matches_finite_differences():
    room, size = t([5.0, 4.0]), t([[1.0, 1.0], [1.0, 1.0]])
    center = t([[2.5, 2.0], [0.3, 2.0]]).requires_grad_()
    assert gradcheck(lambda c: g.out_of_room(c, size, room).sum(), (center,))
    (grad,) = torch.autograd.grad(g.out_of_room(center, size, room).sum(), center)
    assert torch.all(grad[0] == 0)  # fully inside: nothing to fix
    assert grad[1, 0] < 0  # sticks out on the left: gradient descent moves it right


def test_inside_room_allows_the_tolerance():
    size = np.array([[1.0, 1.0]] * 3)
    center = np.array([[2.5, 2.0], [0.4995, 2.0], [0.498, 2.0]])  # inside, 0.5 mm out, 2 mm out
    np.testing.assert_array_equal(g.inside_room(center, size, [5.0, 4.0], tol=1e-3), [True, True, False])


def test_room_functions_broadcast_over_a_batch_of_rooms():
    center, size = random_layouts(batch=4, k=3)
    rooms = np.array([[5.0, 4.0], [3.5, 3.0], [7.0, 6.0], [4.0, 4.0]])
    assert g.inside_room(center, size, rooms).shape == (4, 3)
    assert g.out_of_room(center, size, rooms).shape == (4, 3)
