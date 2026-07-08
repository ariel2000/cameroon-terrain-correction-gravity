from pathlib import Path

from src.utils.read_data import (
    read_original_stations,
    read_kriged_grid,
    summarize_stations,
    summarize_grid,
    find_nearest_grid_node,
)


ROOT = Path(__file__).resolve().parents[1]

STATIONS_FILE = ROOT / "data" / "raw" / "original_stations.xlsx"
GRID_FILE = ROOT / "data" / "raw" / "kriged_grid_surfer.dat"


def test_read_original_stations():
    stations = read_original_stations(STATIONS_FILE)

    assert len(stations) == 1991

    expected_columns = [
        "longitude",
        "latitude",
        "altitude_m",
        "free_air_anomaly_mgal",
    ]

    assert list(stations.columns) == expected_columns


def test_read_kriged_grid():
    grid = read_kriged_grid(GRID_FILE)

    assert grid["nx"] > 0
    assert grid["ny"] > 0
    assert grid["z"].shape == (grid["ny"], grid["nx"])


def test_nearest_grid_node():
    stations = read_original_stations(STATIONS_FILE)
    grid = read_kriged_grid(GRID_FILE)

    first_station = stations.iloc[0]

    i, j, grid_lon, grid_lat, grid_elev = find_nearest_grid_node(
        first_station["longitude"],
        first_station["latitude"],
        grid,
    )

    assert 0 <= i < grid["ny"]
    assert 0 <= j < grid["nx"]
    assert isinstance(grid_lon, float)
    assert isinstance(grid_lat, float)
    assert isinstance(grid_elev, float)


def test_summaries():
    stations = read_original_stations(STATIONS_FILE)
    grid = read_kriged_grid(GRID_FILE)

    station_summary = summarize_stations(stations)
    grid_summary = summarize_grid(grid)

    assert station_summary["number_of_stations"] == 1991
    assert grid_summary["number_of_grid_nodes"] == grid["nx"] * grid["ny"]
