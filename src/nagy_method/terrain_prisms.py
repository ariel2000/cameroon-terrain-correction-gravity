"""
Construction of terrain prisms relative to a gravity station.

Each terrain-grid cell is represented by a rectangular prism whose
vertical extent is defined relative to the station elevation.

Coordinates are local Cartesian coordinates in meters:
    x -> east
    y -> north
    z -> upward
"""

from __future__ import annotations

import numpy as np


def build_terrain_prisms(
    neighborhood,
    station_elevation_m: float,
    cell_dx_m: float,
    cell_dy_m: float,
):
    """
    Build rectangular terrain prisms from a station neighborhood.

    Parameters
    ----------
    neighborhood : pandas.DataFrame
        Must contain:
        dx_m, dy_m, terrain_elevation_m

    station_elevation_m : float
        Elevation of the gravity station in meters.

    cell_dx_m, cell_dy_m : float
        Grid-cell dimensions in meters.

    Returns
    -------
    prisms : ndarray, shape (n, 6)
        Prism boundaries:
        [west, east, south, north, bottom, top]

    signs : ndarray
        +1 for terrain above station level.
        -1 for terrain below station level.
    """

    required = {
        "dx_m",
        "dy_m",
        "terrain_elevation_m",
    }

    missing = required.difference(neighborhood.columns)

    if missing:
        raise ValueError(
            f"Missing neighborhood columns: {sorted(missing)}"
        )

    if cell_dx_m <= 0 or cell_dy_m <= 0:
        raise ValueError("Cell dimensions must be positive.")

    x = neighborhood["dx_m"].to_numpy(dtype=float)
    y = neighborhood["dy_m"].to_numpy(dtype=float)

    terrain_z = neighborhood[
        "terrain_elevation_m"
    ].to_numpy(dtype=float)

    dh = terrain_z - float(station_elevation_m)

    # Cells exactly at station level have zero volume
    valid = ~np.isclose(dh, 0.0)

    x = x[valid]
    y = y[valid]
    dh = dh[valid]

    half_dx = cell_dx_m / 2.0
    half_dy = cell_dy_m / 2.0

    west = x - half_dx
    east = x + half_dx

    south = y - half_dy
    north = y + half_dy

    # Local vertical coordinate:
    # station elevation = z = 0
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
