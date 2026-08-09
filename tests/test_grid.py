from pathlib import Path

import numpy as np

from src.utils.read_data import read_kriged_grid
from src.utils.grid import (
    get_grid_spacing_degrees,
    get_grid_spacing_meters,
    get_grid_extent,
    check_regular_grid,
)


ROOT = Path(__file__).resolve().parents[1]
GRID_FILE = ROOT / "data" / "raw" / "kriged_grid_surfer.dat"


def test_grid_dimensions():
    grid = read_kriged_grid(GRID_FILE)

    assert grid["nx"] == 598
    assert grid["ny"] == 598
    assert grid["z"].shape == (598, 598)


def test_grid_regular():
    grid = read_kriged_grid(GRID_FILE)

    assert check_regular_grid(grid)


def test_grid_spacing_degrees():
    grid = read_kriged_grid(GRID_FILE)

    spacing = get_grid_spacing_degrees(grid)

    assert spacing["dx_mean_deg"] > 0
    assert spacing["dy_mean_deg"] > 0

    assert np.isclose(
        spacing["dx_min_deg"],
        spacing["dx_max_deg"],
        rtol=1e-7,
        atol=1e-12,
    )

    assert np.isclose(
        spacing["dy_min_deg"],
        spacing["dy_max_deg"],
        rtol=1e-7,
        atol=1e-12,
    )


def test_grid_spacing_meters():
    grid = read_kriged_grid(GRID_FILE)

    spacing = get_grid_spacing_meters(grid)

    assert spacing["dx_m"] > 0
    assert spacing["dy_m"] > 0


def test_grid_extent():
    grid = read_kriged_grid(GRID_FILE)

    extent = get_grid_extent(grid)

    assert extent["longitude_max_deg"] > extent["longitude_min_deg"]
    assert extent["latitude_max_deg"] > extent["latitude_min_deg"]
    assert extent["elevation_max_m"] >= extent["elevation_min_m"]
