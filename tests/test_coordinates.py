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

def test_inverse_coordinate_conversion():
    from src.utils.coordinates import local_meters_to_degrees

    lon_original = 10.2
    lat_original = 4.3

    lon0 = 10.0
    lat0 = 4.0

    x, y = degrees_to_local_meters(
        longitude=lon_original,
        latitude=lat_original,
        reference_longitude=lon0,
        reference_latitude=lat0,
    )

    lon_recovered, lat_recovered = local_meters_to_degrees(
        x_m=x,
        y_m=y,
        reference_longitude=lon0,
        reference_latitude=lat0,
    )

    assert np.isclose(
        lon_recovered,
        lon_original,
        atol=1e-12,
    )

    assert np.isclose(
        lat_recovered,
        lat_original,
        atol=1e-12,
    )


def test_inverse_zero_offset():
    from src.utils.coordinates import local_meters_to_degrees

    lon, lat = local_meters_to_degrees(
        x_m=0.0,
        y_m=0.0,
        reference_longitude=9.3,
        reference_latitude=4.5,
    )

    assert np.isclose(lon, 9.3)
    assert np.isclose(lat, 4.5)


def test_inverse_array_shape():
    from src.utils.coordinates import local_meters_to_degrees

    x = np.array([
        [0.0, 500.0],
        [0.0, 500.0],
    ])

    y = np.array([
        [0.0, 0.0],
        [500.0, 500.0],
    ])

    lon, lat = local_meters_to_degrees(
        x_m=x,
        y_m=y,
        reference_longitude=9.0,
        reference_latitude=4.0,
    )

    assert lon.shape == (2, 2)
    assert lat.shape == (2, 2)
