"""Hand-built layouts and a comparison helper shared by the tests."""
import numpy as np

# Slot numbers of the living room (tests/test_catalog.py checks this order).
SOFA, TV_UNIT, COFFEE_TABLE, BOOKSHELF, ARMCHAIR, SIDE_TABLE = range(6)

# A 5.0 x 4.0 m room with the door in the middle of the west wall: door centre (0, 2.0),
# clearance zone x 0 to 0.9, y 1.55 to 2.45. Every item has its default variant, every
# hard check passes, and every relation in rules.yaml holds.
GOOD_ITEMS = {
    "sofa": (2.8, 3.55, 2),  # against the north wall, facing south
    "tv_unit": (2.8, 0.2, 0),  # against the south wall, facing the sofa 2.7 m away
    "coffee_table": (2.8, 2.425, 0),  # 0.40 m in front of the sofa
    "bookshelf": (4.85, 1.5, 3),  # against the east wall, facing west
    "armchair": (4.2, 2.6, 3),  # at 90 degrees to the sofa, 0.5 m from the coffee table
    "side_table": (1.475, 3.775, 0),  # 0.05 m from the sofa's west end
}


def assert_same_layout(actual, expected, atol=0.0):
    """Equal rooms, doors, items and rotations; centres equal within `atol` meters."""
    assert (actual.room_type, actual.width, actual.depth, actual.door_wall) == \
        (expected.room_type, expected.width, expected.depth, expected.door_wall)
    assert actual.door_offset == expected.door_offset
    assert actual.variant_ids == expected.variant_ids
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(actual.size, expected.size)
    np.testing.assert_array_equal(actual.rot, expected.rot)
    np.testing.assert_allclose(actual.center, expected.center, rtol=0, atol=atol)
