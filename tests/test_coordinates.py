import numpy as np

from src.utils.coordinates import (
    degrees_to_local_meters,
    horizontal_distance_m,
    distance_3d_m,
)


def test_zero_distance():
    """
    Identical geographic coordinates should give zero distance.
    """

    x, y = degrees_to_local_meters(
        longitude=12.0,
        latitude=4.0,
        reference_longitude=12.0,
        reference_latitude=4.0,
    )

    assert np.isclose(x, 0.0)
    assert np.isclose(y, 0.0)


def test_one_degree_latitude():
    """
    One degree of latitude should be approximately 111.2 km.
    """

    x, y = degrees_to_local_meters(
        longitude=12.0,
        latitude=5.0,
        reference_longitude=12.0,
        reference_latitude=4.0,
    )

    assert np.isclose(y, 111194.9, atol=200)
    assert np.isclose(x, 0.0)


def test_one_degree_longitude():
    """
    One degree of longitude at latitude 4°.
    """

    x, y = degrees_to_local_meters(
        longitude=13.0,
        latitude=4.0,
        reference_longitude=12.0,
        reference_latitude=4.0,
    )

    expected = 111194.9 * np.cos(np.deg2rad(4.0))

    assert np.isclose(x, expected, atol=200)
    assert np.isclose(y, 0.0)


def test_horizontal_distance():
    """
    Horizontal distance must be positive.
    """

    d = horizontal_distance_m(
        longitude=13.0,
        latitude=4.0,
        reference_longitude=12.0,
        reference_latitude=4.0,
    )

    assert d > 100000


def test_3d_distance():
    """
    3D distance should be larger than horizontal distance.
    """

    dh = horizontal_distance_m(
        longitude=13.0,
        latitude=4.0,
        reference_longitude=12.0,
        reference_latitude=4.0,
    )

    d3 = distance_3d_m(
        longitude=13.0,
        latitude=4.0,
        elevation_m=500,
        reference_longitude=12.0,
        reference_latitude=4.0,
        reference_elevation_m=0,
    )

    assert d3 > dh
