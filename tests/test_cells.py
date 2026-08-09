from pathlib import Path

import numpy as np

from src.utils.read_data import read_kriged_grid
from src.utils.cells import build_terrain_cells


ROOT = Path(__file__).resolve().parents[1]

GRID_FILE = (
    ROOT
    / "data"
    / "raw"
    / "kriged_grid_surfer.dat"
)


def test_number_of_cells():
    grid = read_kriged_grid(GRID_FILE)

    cells = build_terrain_cells(grid)

    assert cells["nx_cells"] == 597
    assert cells["ny_cells"] == 597

    assert cells["number_of_cells"] == 597 * 597
    assert cells["number_of_cells"] == 356409


def test_cell_array_shapes():
    grid = read_kriged_grid(GRID_FILE)

    cells = build_terrain_cells(grid)

    expected_shape = (597, 597)

    assert cells["lon_center"].shape == expected_shape
    assert cells["lat_center"].shape == expected_shape
    assert cells["elevation_center_m"].shape == expected_shape


def test_first_cell_center():
    grid = read_kriged_grid(GRID_FILE)

    cells = build_terrain_cells(grid)

    expected_lon = (
        grid["x"][0]
        + grid["x"][1]
    ) / 2.0

    expected_lat = (
        grid["y"][0]
        + grid["y"][1]
    ) / 2.0

    assert np.isclose(
        cells["lon_center"][0, 0],
        expected_lon,
    )

    assert np.isclose(
        cells["lat_center"][0, 0],
        expected_lat,
    )


def test_first_cell_elevation():
    grid = read_kriged_grid(GRID_FILE)

    cells = build_terrain_cells(grid)

    expected_elevation = np.mean(
        [
            grid["z"][0, 0],
            grid["z"][0, 1],
            grid["z"][1, 0],
            grid["z"][1, 1],
        ]
    )

    assert np.isclose(
        cells["elevation_center_m"][0, 0],
        expected_elevation,
    )


def test_cell_boundaries_are_ordered():
    grid = read_kriged_grid(GRID_FILE)

    cells = build_terrain_cells(grid)

    assert np.all(
        cells["lon_east"] > cells["lon_west"]
    )

    assert np.all(
        cells["lat_north"] > cells["lat_south"]
    )
