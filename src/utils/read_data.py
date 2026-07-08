"""
Data readers for the Cameroon terrain correction project.

Files expected in data/raw/:
- original_stations.xlsx
- kriged_grid_surfer.dat

Coordinate convention:
- longitude: degrees
- latitude: degrees
- altitude/elevation: meters
- free-air anomaly: probably mGal
"""

from pathlib import Path

import numpy as np
import pandas as pd


def read_original_stations(file_path: str | Path) -> pd.DataFrame:
    """
    Read the 1991 original gravity stations from Excel.

    Returns
    -------
    pandas.DataFrame
        Columns:
        longitude, latitude, altitude_m, free_air_anomaly_mgal
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_excel(file_path)

    expected_columns = ["Longitude", "Latitude", "Altitude", "A(air libre)"]

    missing = [col for col in expected_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in station file: {missing}")

    data = data[expected_columns].copy()

    data.columns = [
        "longitude",
        "latitude",
        "altitude_m",
        "free_air_anomaly_mgal",
    ]

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna().reset_index(drop=True)

    if len(data) != 1991:
        raise ValueError(
            f"Expected 1991 stations, but found {len(data)} valid rows."
        )

    return data


def read_kriged_grid(file_path: str | Path) -> dict:
    """
    Read the kriged Surfer XYZ grid.

    The file must contain three columns:

    longitude latitude elevation

    Returns
    -------
    dict
        Dictionary with x, y, z, xx, yy and metadata.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    xyz = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "elevation_m"],
        engine="python",
    )

    xyz = xyz.apply(pd.to_numeric, errors="coerce")
    xyz = xyz.dropna().reset_index(drop=True)

    x = np.sort(xyz["longitude"].unique())
    y = np.sort(xyz["latitude"].unique())

    nx = len(x)
    ny = len(y)

    if nx * ny != len(xyz):
        raise ValueError(
            "The grid is not regular: nx * ny is different from the number of rows."
        )

    grid_table = xyz.pivot(
        index="latitude",
        columns="longitude",
        values="elevation_m",
    )

    grid_table = grid_table.reindex(index=y, columns=x)

    z = grid_table.to_numpy()

    xx, yy = np.meshgrid(x, y)

    return {
        "x": x,
        "y": y,
        "z": z,
        "xx": xx,
        "yy": yy,
        "nx": nx,
        "ny": ny,
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "z_min": float(np.nanmin(z)),
        "z_max": float(np.nanmax(z)),
    }


def find_nearest_grid_node(
    longitude: float,
    latitude: float,
    grid: dict,
) -> tuple[int, int, float, float, float]:
    """
    Locate the nearest grid node to a gravity station.

    Returns
    -------
    tuple
        i, j, grid_longitude, grid_latitude, grid_elevation_m

        i = latitude index
        j = longitude index
    """
    x = grid["x"]
    y = grid["y"]
    z = grid["z"]

    j = int(np.argmin(np.abs(x - longitude)))
    i = int(np.argmin(np.abs(y - latitude)))

    return i, j, float(x[j]), float(y[i]), float(z[i, j])


def summarize_stations(stations: pd.DataFrame) -> dict:
    """
    Summary of the original station dataset.
    """
    return {
        "number_of_stations": len(stations),
        "longitude_min": float(stations["longitude"].min()),
        "longitude_max": float(stations["longitude"].max()),
        "latitude_min": float(stations["latitude"].min()),
        "latitude_max": float(stations["latitude"].max()),
        "altitude_min_m": float(stations["altitude_m"].min()),
        "altitude_max_m": float(stations["altitude_m"].max()),
        "free_air_anomaly_min_mgal": float(
            stations["free_air_anomaly_mgal"].min()
        ),
        "free_air_anomaly_max_mgal": float(
            stations["free_air_anomaly_mgal"].max()
        ),
    }


def summarize_grid(grid: dict) -> dict:
    """
    Summary of the kriged grid.
    """
    return {
        "nx": grid["nx"],
        "ny": grid["ny"],
        "number_of_grid_nodes": grid["nx"] * grid["ny"],
        "longitude_min": grid["x_min"],
        "longitude_max": grid["x_max"],
        "latitude_min": grid["y_min"],
        "latitude_max": grid["y_max"],
        "elevation_min_m": grid["z_min"],
        "elevation_max_m": grid["z_max"],
  }
