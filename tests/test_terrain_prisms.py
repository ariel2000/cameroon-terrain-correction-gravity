import numpy as np
import pandas as pd
import pytest

from src.nagy_method.terrain_prisms import build_terrain_prisms


def test_prism_above_station():
    neighborhood = pd.DataFrame(
        {
            "dx_m": [1000.0],
            "dy_m": [0.0],
            "terrain_elevation_m": [150.0],
        }
    )

    prisms, signs = build_terrain_prisms(
        neighborhood,
        station_elevation_m=100.0,
        cell_dx_m=500.0,
        cell_dy_m=600.0,
    )

    assert prisms.shape == (1, 6)

    west, east, south, north, bottom, top = prisms[0]

    assert np.isclose(west, 750.0)
    assert np.isclose(east, 1250.0)

    assert np.isclose(south, -300.0)
    assert np.isclose(north, 300.0)

    assert np.isclose(bottom, 0.0)
    assert np.isclose(top, 50.0)

    assert signs[0] == 1.0


def test_prism_below_station():
    neighborhood = pd.DataFrame(
        {
            "dx_m": [1000.0],
            "dy_m": [0.0],
            "terrain_elevation_m": [70.0],
        }
    )

    prisms, signs = build_terrain_prisms(
        neighborhood,
        station_elevation_m=100.0,
        cell_dx_m=500.0,
        cell_dy_m=600.0,
    )

    assert np.isclose(prisms[0, 4], -30.0)
    assert np.isclose(prisms[0, 5], 0.0)

    assert signs[0] == -1.0


def test_zero_height_cell_removed():
    neighborhood = pd.DataFrame(
        {
            "dx_m": [0.0],
            "dy_m": [0.0],
            "terrain_elevation_m": [100.0],
        }
    )

    prisms, signs = build_terrain_prisms(
        neighborhood,
        station_elevation_m=100.0,
        cell_dx_m=500.0,
        cell_dy_m=500.0,
    )

    assert prisms.shape == (0, 6)
    assert signs.size == 0


def test_multiple_cells():
    neighborhood = pd.DataFrame(
        {
            "dx_m": [0.0, 500.0, 1000.0],
            "dy_m": [0.0, 0.0, 0.0],
            "terrain_elevation_m": [
                120.0,
                80.0,
                100.0,
            ],
        }
    )

    prisms, signs = build_terrain_prisms(
        neighborhood,
        station_elevation_m=100.0,
        cell_dx_m=500.0,
        cell_dy_m=500.0,
    )

    # Third cell has dh = 0 and must disappear
    assert prisms.shape == (2, 6)

    assert np.array_equal(
        signs,
        np.array([1.0, -1.0]),
    )


def test_invalid_cell_size():
    neighborhood = pd.DataFrame(
        {
            "dx_m": [0.0],
            "dy_m": [0.0],
            "terrain_elevation_m": [120.0],
        }
    )

    with pytest.raises(ValueError):
        build_terrain_prisms(
            neighborhood,
            station_elevation_m=100.0,
            cell_dx_m=-500.0,
            cell_dy_m=500.0,
        )
