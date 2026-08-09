"""
Terrain-cell construction utilities.

The Surfer terrain grid contains elevation values at grid nodes.
For terrain-correction calculations, adjacent nodes are converted
into topographic cells.

For a grid of ny x nx nodes, the resulting terrain-cell array has:

    (ny - 1) x (nx - 1) cells

Each cell is characterized by:
- geographic boundaries,
- geographic center,
- representative terrain elevation.

The representative elevation is currently defined as the arithmetic
mean of the four corner-node elevations.
"""

from __future__ import annotations

import numpy as np


def build_terrain_cells(grid: dict) -> dict:
    """
    Build terrain cells from a regular node-based grid.

    Parameters
    ----------
    grid : dict
        Grid returned by read_kriged_grid().

        Required entries:
        - x : 1-D longitude array
        - y : 1-D latitude array
        - z : 2-D elevation array with shape (ny, nx)

    Returns
    -------
    dict
        Terrain-cell geometry and representative elevations.
    """

    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["y"], dtype=float)
    z = np.asarray(grid["z"], dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Grid coordinates x and y must be one-dimensional.")

    if z.shape != (y.size, x.size):
        raise ValueError(
            "Grid elevation array shape is inconsistent with x and y."
        )

    if x.size < 2 or y.size < 2:
        raise ValueError(
            "At least two grid nodes are required in each direction."
        )

    # Geographic cell boundaries
    lon_west = x[:-1]
    lon_east = x[1:]

    lat_south = y[:-1]
    lat_north = y[1:]

    # Geographic cell centers
    lon_center_1d = 0.5 * (lon_west + lon_east)
    lat_center_1d = 0.5 * (lat_south + lat_north)

    lon_center, lat_center = np.meshgrid(
        lon_center_1d,
        lat_center_1d,
    )

    # Mean elevation of the four corner nodes
    z_sw = z[:-1, :-1]
    z_se = z[:-1, 1:]
    z_nw = z[1:, :-1]
    z_ne = z[1:, 1:]

    elevation_center_m = (
        z_sw
        + z_se
        + z_nw
        + z_ne
    ) / 4.0

    ny_cells = y.size - 1
    nx_cells = x.size - 1

    return {
        "lon_west": lon_west,
        "lon_east": lon_east,
        "lat_south": lat_south,
        "lat_north": lat_north,
        "lon_center": lon_center,
        "lat_center": lat_center,
        "elevation_center_m": elevation_center_m,
        "nx_cells": nx_cells,
        "ny_cells": ny_cells,
        "number_of_cells": nx_cells * ny_cells,
    }
