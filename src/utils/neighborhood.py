"""
Station-to-terrain neighborhood utilities.

This module combines:
- terrain cells,
- station coordinates,
- station elevation,

to construct the geometric quantities required by terrain-correction
methods.

No gravity formula is implemented here. The purpose of this module is
to provide a common geometric basis for Hammer, Nagy, prism-based,
spherical-cap, and Vannes calculations.
"""

from __future__ import annotations

import numpy as np

from src.utils.geometry import local_distances


def build_station_neighborhood(
    cells: dict,
    station_longitude: float,
    station_latitude: float,
    station_elevation_m: float,
    max_radius_m: float | None = None,
) -> dict:
    """
    Build station-to-terrain geometry.

    Parameters
    ----------
    cells : dict
        Terrain-cell dictionary returned by build_terrain_cells().

    station_longitude : float
        Station longitude in decimal degrees.

    station_latitude : float
        Station latitude in decimal degrees.

    station_elevation_m : float
        Station elevation in meters.

    max_radius_m : float or None
        Maximum horizontal radius around the station, in meters.
        If None, all terrain cells are retained.

    Returns
    -------
    dict
        One-dimensional arrays describing the selected terrain cells.

        Returned quantities include:
        - longitude
        - latitude
        - terrain_elevation_m
        - dx_m
        - dy_m
        - horizontal_distance_m
        - elevation_difference_m
        - number_of_cells
        - max_radius_m
    """

    lon = np.asarray(cells["lon_center"], dtype=float)
    lat = np.asarray(cells["lat_center"], dtype=float)
    terrain_elevation = np.asarray(
        cells["elevation_center_m"],
        dtype=float,
    )

    if lon.shape != lat.shape:
        raise ValueError(
            "Cell longitude and latitude arrays must have identical shapes."
        )

    if lon.shape != terrain_elevation.shape:
        raise ValueError(
            "Cell coordinate and elevation arrays must have identical shapes."
        )

    if max_radius_m is not None and max_radius_m <= 0:
        raise ValueError("max_radius_m must be strictly positive.")

    dx_m, dy_m, r_m = local_distances(
        lon_grid=lon,
        lat_grid=lat,
        lon_station=station_longitude,
        lat_station=station_latitude,
    )

    dh_m = terrain_elevation - float(station_elevation_m)

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(terrain_elevation)
        & np.isfinite(dx_m)
        & np.isfinite(dy_m)
        & np.isfinite(r_m)
        & np.isfinite(dh_m)
    )

    if max_radius_m is not None:
        valid &= r_m <= float(max_radius_m)

    return {
        "longitude": lon[valid],
        "latitude": lat[valid],
        "terrain_elevation_m": terrain_elevation[valid],
        "dx_m": dx_m[valid],
        "dy_m": dy_m[valid],
        "horizontal_distance_m": r_m[valid],
        "elevation_difference_m": dh_m[valid],
        "number_of_cells": int(np.count_nonzero(valid)),
        "max_radius_m": (
            None if max_radius_m is None else float(max_radius_m)
        ),
    }
