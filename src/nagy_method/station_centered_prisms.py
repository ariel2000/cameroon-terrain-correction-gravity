"""
Build rectangular terrain prisms from a station-centred local grid.

The local grid contains terrain elevations at nodes.
Adjacent nodes define rectangular cells.

For each cell:
- horizontal boundaries come directly from x_m and y_m;
- representative relative elevation is the mean of the four corners;
- the reference plane is z = 0, i.e. station elevation.

The station itself is located exactly at x = y = z = 0.
"""

from __future__ import annotations

import numpy as np


def build_station_centered_prisms(
    local_grid: dict,
    max_radius_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build terrain prisms from a station-centred node grid.

    Parameters
    ----------
    local_grid : dict
        Output of build_station_centered_grid().
        Required fields:
        - x_m
        - y_m
        - relative_elevation_m

    max_radius_m : float or None
        Optional circular selection radius based on cell centres.

    Returns
    -------
    prisms : ndarray, shape (n, 6)
        Prism limits:
        [west, east, south, north, bottom, top]

    signs : ndarray, shape (n,)
        +1 for terrain above the station reference plane.
        -1 for terrain below the station reference plane.

    radius_m : ndarray, shape (n,)
        Horizontal distance of each retained cell centre
        from the station.
    """

    x = np.asarray(local_grid["x_m"], dtype=float)
    y = np.asarray(local_grid["y_m"], dtype=float)
    h = np.asarray(
        local_grid["relative_elevation_m"],
        dtype=float,
    )

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "x_m and y_m must be one-dimensional arrays."
        )

    if h.shape != (y.size, x.size):
        raise ValueError(
            "relative_elevation_m shape is inconsistent with x_m and y_m."
        )

    if x.size < 2 or y.size < 2:
        raise ValueError(
            "At least two nodes are required in each direction."
        )

    if max_radius_m is not None and max_radius_m <= 0:
        raise ValueError(
            "max_radius_m must be strictly positive."
        )

    # --------------------------------------------------------
    # Cell boundaries
    # --------------------------------------------------------

    west_1d = x[:-1]
    east_1d = x[1:]

    south_1d = y[:-1]
    north_1d = y[1:]

    west, south = np.meshgrid(
        west_1d,
        south_1d,
    )

    east, north = np.meshgrid(
        east_1d,
        north_1d,
    )

    # --------------------------------------------------------
    # Cell centres
    # --------------------------------------------------------

    x_center = 0.5 * (west + east)
    y_center = 0.5 * (south + north)

    radius = np.hypot(
        x_center,
        y_center,
    )

    # --------------------------------------------------------
    # Representative terrain elevation in each cell:
    # arithmetic mean of the four corner nodes
    # --------------------------------------------------------

    h_sw = h[:-1, :-1]
    h_se = h[:-1, 1:]
    h_nw = h[1:, :-1]
    h_ne = h[1:, 1:]

    h_cell = (
        h_sw
        + h_se
        + h_nw
        + h_ne
    ) / 4.0

    # --------------------------------------------------------
    # Selection mask
    # --------------------------------------------------------

    valid = (
        np.isfinite(h_cell)
        & ~np.isclose(h_cell, 0.0)
    )

    if max_radius_m is not None:
        valid &= radius <= float(max_radius_m)

    h_selected = h_cell[valid]

    bottom = np.minimum(
        0.0,
        h_selected,
    )

    top = np.maximum(
        0.0,
        h_selected,
    )

    signs = np.sign(h_selected)

    prisms = np.column_stack(
        (
            west[valid],
            east[valid],
            south[valid],
            north[valid],
            bottom,
            top,
        )
    )

    return (
        prisms,
        signs,
        radius[valid],
    )
