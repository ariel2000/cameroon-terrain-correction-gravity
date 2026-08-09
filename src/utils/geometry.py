"""
Geometry utilities for terrain correction calculations.

This module builds station-to-terrain geometry using the coordinate
conversion functions already validated in coordinates.py.
"""

from __future__ import annotations

import numpy as np

from src.utils.coordinates import degrees_to_local_meters


def local_distances(
    lon_grid,
    lat_grid,
    lon_station,
    lat_station,
):
    """
    Compute horizontal local distances between a gravity station
    and terrain-grid nodes.

    Parameters
    ----------
    lon_grid : array_like
        Grid longitudes in decimal degrees.

    lat_grid : array_like
        Grid latitudes in decimal degrees.

    lon_station : float
        Station longitude in decimal degrees.

    lat_station : float
        Station latitude in decimal degrees.

    Returns
    -------
    dx_m : ndarray
        East-west offsets relative to the station, in meters.

    dy_m : ndarray
        North-south offsets relative to the station, in meters.

    r_m : ndarray
        Horizontal radial distance from the station, in meters.
    """

    dx_m, dy_m = degrees_to_local_meters(
        longitude=lon_grid,
        latitude=lat_grid,
        reference_longitude=lon_station,
        reference_latitude=lat_station,
    )

    r_m = np.hypot(dx_m, dy_m)

    return dx_m, dy_m, r_m
