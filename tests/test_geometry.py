import numpy as np

from src.utils.geometry import local_distances


def test_local_distances_zero_at_station():
    lon = np.array([[9.3]])
    lat = np.array([[4.0]])

    dx, dy, r = local_distances(
        lon,
        lat,
        lon_station=9.3,
        lat_station=4.0,
    )

    assert np.isclose(dx[0, 0], 0.0)
    assert np.isclose(dy[0, 0], 0.0)
    assert np.isclose(r[0, 0], 0.0)


def test_local_distances_shape():
    lon = np.array([
        [9.0, 9.1],
        [9.0, 9.1],
    ])

    lat = np.array([
        [4.0, 4.0],
        [4.1, 4.1],
    ])

    dx, dy, r = local_distances(
        lon,
        lat,
        lon_station=9.05,
        lat_station=4.05,
    )

    assert dx.shape == (2, 2)
    assert dy.shape == (2, 2)
    assert r.shape == (2, 2)


def test_radial_distance_definition():
    lon = np.array([[9.1]])
    lat = np.array([[4.1]])

    dx, dy, r = local_distances(
        lon,
        lat,
        lon_station=9.0,
        lat_station=4.0,
    )

    expected = np.sqrt(dx**2 + dy**2)

    assert np.allclose(r, expected)


def test_distance_is_non_negative():
    lon = np.array([
        [8.9, 9.0, 9.1],
        [8.9, 9.0, 9.1],
    ])

    lat = np.array([
        [3.9, 3.9, 3.9],
        [4.1, 4.1, 4.1],
    ])

    _, _, r = local_distances(
        lon,
        lat,
        lon_station=9.0,
        lat_station=4.0,
    )

    assert np.all(r >= 0.0)
