import numpy as np
import pytest

from src.utils.station_centered_grid import (
    terrain_relative_to_station,
    enforce_station_reference,
)


def test_relative_elevation_basic():
    terrain = np.array([
        [100.0, 120.0],
        [80.0, 110.0],
    ])

    result = terrain_relative_to_station(
        terrain_elevation=terrain,
        station_elevation=100.0,
    )

    expected = np.array([
        [0.0, 20.0],
        [-20.0, 10.0],
    ])

    assert np.allclose(result, expected)


def test_station_elevation_not_zero():
    terrain = np.array([50.0, 75.0, 100.0])

    result = terrain_relative_to_station(
        terrain_elevation=terrain,
        station_elevation=75.0,
    )

    expected = np.array([-25.0, 0.0, 25.0])

    assert np.allclose(result, expected)


def test_invalid_station_elevation():
    terrain = np.array([100.0, 120.0])

    with pytest.raises(ValueError):
        terrain_relative_to_station(
            terrain_elevation=terrain,
            station_elevation=np.nan,
        )


def test_invalid_terrain_values():
    terrain = np.array([
        [100.0, np.nan],
        [120.0, 130.0],
    ])

    with pytest.raises(ValueError):
        terrain_relative_to_station(
            terrain_elevation=terrain,
            station_elevation=100.0,
        )


def test_enforce_station_reference():
    relative = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    result = enforce_station_reference(
        relative_elevation=relative,
        station_index=(1, 0),
    )

    assert np.isclose(result[1, 0], 0.0)

    # Other values must remain unchanged
    assert np.isclose(result[0, 0], 10.0)
    assert np.isclose(result[0, 1], 20.0)
    assert np.isclose(result[1, 1], 40.0)


def test_original_array_not_modified():
    relative = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    original = relative.copy()

    enforce_station_reference(
        relative_elevation=relative,
        station_index=(0, 1),
    )

    assert np.array_equal(relative, original)


def test_invalid_station_index():
    relative = np.zeros((2, 2))

    with pytest.raises(IndexError):
        enforce_station_reference(
            relative_elevation=relative,
            station_index=(3, 0),
        )


def test_requires_two_dimensional_array():
    relative = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        enforce_station_reference(
            relative_elevation=relative,
            station_index=(0, 0),
        )
