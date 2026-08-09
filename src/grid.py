"""
Grid geometry utilities for the Cameroon terrain correction project.

The kriged terrain grid is defined by:
- longitude in decimal degrees
- latitude in decimal degrees
- elevation in meters

This module characterizes the geometry of the grid before it is
used by terrain-correction methods.
"""

from __future__ import annotations

import numpy as np

from src.utils.coordinates import degrees_to_local_meters


def get_grid_spacing_degrees(grid: dict) -> dict:
    """
    Compute grid spacing in longitude and latitude.

    Parameters
    ----------
    grid : dict
        Grid dictionary returned by read_kriged_grid().

    Returns
    -------
    dict
        Mean, minimum, and maximum spacing in degrees.
    """
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["y"], dtype=float)

    if x.size < 2 or y.size < 2:
        raise ValueError("Grid must contain at least two nodes in each direction.")

    dx = np.diff(x)
    dy = np.diff(y)

    return {
        "dx_mean_deg": float(np.mean(dx)),
        "dx_min_deg": float(np.min(dx)),
        "dx_max_deg": float(np.max(dx)),
        "dy_mean_deg": float(np.mean(dy)),
        "dy_min_deg": float(np.min(dy)),
        "dy_max_deg": float(np.max(dy)),
    }


def check_regular_grid(
    grid: dict,
    rtol: float = 1e-7,
    atol: float = 1e-12,
) -> bool:
    """
    Check whether longitude and latitude spacing are regular.

    Parameters
    ----------
    grid : dict
        Grid dictionary returned by read_kriged_grid().

    rtol : float
        Relative tolerance.

    atol : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if the grid is regular in both directions.
    """
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["y"], dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)

    regular_x = np.allclose(dx, dx[0], rtol=rtol, atol=atol)
    regular_y = np.allclose(dy, dy[0], rtol=rtol, atol=atol)

    return bool(regular_x and regular_y)


def get_grid_extent(grid: dict) -> dict:
    """
    Return geographic and elevation limits of the grid.
    """
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["y"], dtype=float)
    z = np.asarray(grid["z"], dtype=float)

    return {
        "longitude_min_deg": float(np.min(x)),
        "longitude_max_deg": float(np.max(x)),
        "latitude_min_deg": float(np.min(y)),
        "latitude_max_deg": float(np.max(y)),
        "elevation_min_m": float(np.nanmin(z)),
        "elevation_max_m": float(np.nanmax(z)),
    }


def get_grid_spacing_meters(grid: dict) -> dict:
    """
    Estimate grid spacing in meters at the center of the study area.

    The conversion is performed using the local spherical-Earth
    approximation implemented in coordinates.py.

    Returns
    -------
    dict
        Approximate east-west and north-south spacing in meters.
    """
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["y"], dtype=float)

    if x.size < 2 or y.size < 2:
        raise ValueError("Grid must contain at least two nodes in each direction.")

    lon0 = float((x.min() + x.max()) / 2.0)
    lat0 = float((y.min() + y.max()) / 2.0)

    dx_deg = float(np.mean(np.diff(x)))
    dy_deg = float(np.mean(np.diff(y)))

    x_m, _ = degrees_to_local_meters(
        longitude=lon0 + dx_deg,
        latitude=lat0,
        reference_longitude=lon0,
        reference_latitude=lat0,
    )

    _, y_m = degrees_to_local_meters(
        longitude=lon0,
        latitude=lat0 + dy_deg,
        reference_longitude=lon0,
        reference_latitude=lat0,
    )

    return {
        "dx_m": float(abs(x_m)),
        "dy_m": float(abs(y_m)),
        "reference_longitude_deg": lon0,
        "reference_latitude_deg": lat0,
    }
