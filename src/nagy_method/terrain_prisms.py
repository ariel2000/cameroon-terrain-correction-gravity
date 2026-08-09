"""
Construction of terrain prisms relative to a gravity station.

The function accepts the dictionary returned directly by
build_station_neighborhood().

Coordinates are local Cartesian coordinates in meters:
    x -> east
    y -> north
    z -> upward

The station elevation defines the local vertical reference z = 0.
"""

from __future__ import annotations

import numpy as np


def build_terrain_prisms(
    neighborhood: dict,
    station_elevation_m: float,
    cell_dx_m: float,
    cell_dy_m: float,
):
    """
    Build rectangular terrain prisms from a station neighborhood.

    Parameters
    ----------
    neighborhood : dict
        Dictionary returned by build_station_neighborhood().
        Required arrays:
        - dx_m
        - dy_m
        - terrain_elevation_m

    station_elevation_m : float
        Elevation of the gravity station in meters.

    cell_dx_m : float
        East-west cell dimension in meters.

    cell_dy_m : float
        North-south cell dimension in meters.

    Returns
    -------
    prisms : ndarray, shape (n, 6)
        Prism boundaries:
        west, east, south, north, bottom, top.

    signs : ndarray
        +1 for terrain above station level.
        -1 for terrain below station level.
    """

    required = {
        "dx_m",
        "dy_m",
        "terrain_elevation_m",
    }

    missing = required.difference(neighborhood.keys())

    if missing:
        raise ValueError(
            f"Missing neighborhood fields: {sorted(missing)}"
        )

    if cell_dx_m <= 0 or cell_dy_m <= 0:
        raise ValueError("Cell dimensions must be positive.")

    x = np.asarray(neighborhood["dx_m"], dtype=float)
    y = np.asarray(neighborhood["dy_m"], dtype=float)

    terrain_z = np.asarray(
        neighborhood["terrain_elevation_m"],
        dtype=float,
    )

    if not (
        x.shape == y.shape == terrain_z.shape
    ):
        raise ValueError(
            "Neighborhood arrays must have identical shapes."
        )

    dh = terrain_z - float(station_elevation_m)

    # Remove zero-volume cells
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(dh)
        & ~np.isclose(dh, 0.0)
    )

    x = x[valid]
    y = y[valid]
    dh = dh[valid]

    half_dx = cell_dx_m / 2.0
    half_dy = cell_dy_m / 2.0

    west = x - half_dx
    east = x + half_dx

    south = y - half_dy
    north = y + half_dy

    # Local vertical system:
    # station elevation corresponds to z = 0
    bottom = np.minimum(0.0, dh)
    top = np.maximum(0.0, dh)

    signs = np.sign(dh)

    prisms = np.column_stack(
        (
            west,
            east,
            south,
            north,
            bottom,
            top,
        )
    )

    return prisms, signs
