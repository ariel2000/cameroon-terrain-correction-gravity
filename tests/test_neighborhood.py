import numpy as np
import pytest

from src.utils.neighborhood import build_station_neighborhood


def make_test_cells():
    """
    Small synthetic 2 x 2 terrain-cell model.
    """
    return {
        "lon_center": np.array([
            [9.00, 9.01],
            [9.00, 9.01],
        ]),
        "lat_center": np.array([
            [4.00, 4.00],
            [4.01, 4.01],
        ]),
        "elevation_center_m": np.array([
            [100.0, 110.0],
            [120.0, 130.0],
        ]),
    }


def test_all_cells_retained_without_radius():
    cells = make_test_cells()

    neighborhood = build_station_neighborhood(
        cells=cells,
        station_longitude=9.0,
        station_latitude=4.0,
        station_elevation_m=100.0,
    )

    assert neighborhood["number_of_cells"] == 4


def test_elevation_difference():
    cells = make_test_cells()

    neighborhood = build_station_neighborhood(
        cells=cells,
        station_longitude=9.0,
        station_latitude=4.0,
        station_elevation_m=100.0,
    )

    expected = np.array([
        0.0,
        10.0,
        20.0,
        30.0,
    ])

    assert np.allclose(
        np.sort(neighborhood["elevation_difference_m"]),
        expected,
    )


def test_radius_selection():
    cells = make_test_cells()

    neighborhood = build_station_neighborhood(
        cells=cells,
        station_longitude=9.0,
        station_latitude=4.0,
        station_elevation_m=100.0,
        max_radius_m=500.0,
    )

    assert neighborhood["number_of_cells"] == 1

    assert np.all(
        neighborhood["horizontal_distance_m"] <= 500.0
    )


def test_horizontal_distances_non_negative():
    cells = make_test_cells()

    neighborhood = build_station_neighborhood(
        cells=cells,
        station_longitude=9.0,
        station_latitude=4.0,
        station_elevation_m=100.0,
    )

    assert np.all(
        neighborhood["horizontal_distance_m"] >= 0.0
    )


def test_invalid_radius():
    cells = make_test_cells()

    with pytest.raises(ValueError):
        build_station_neighborhood(
            cells=cells,
            station_longitude=9.0,
            station_latitude=4.0,
            station_elevation_m=100.0,
            max_radius_m=0.0,
        )
