"""
Exact gravitational attraction of rectangular prisms.

This module uses Harmonica as a validated implementation of the
right-rectangular prism gravity formulation associated with Nagy.

Coordinates:
- easting: meters
- northing: meters
- upward: meters

Density:
- kg/m^3

Output:
- vertical gravity component g_z in mGal
"""

from __future__ import annotations

import numpy as np
import harmonica as hm


def prism_vertical_gravity(
    observation_easting_m: float,
    observation_northing_m: float,
    observation_upward_m: float,
    west_m: float,
    east_m: float,
    south_m: float,
    north_m: float,
    bottom_m: float,
    top_m: float,
    density_kg_m3: float,
) -> float:
    """
    Compute the downward vertical gravitational acceleration
    produced by one rectangular prism.

    Returns
    -------
    float
        Vertical gravitational attraction in mGal.
    """

    if east_m <= west_m:
        raise ValueError("east_m must be greater than west_m.")

    if north_m <= south_m:
        raise ValueError("north_m must be greater than south_m.")

    if top_m <= bottom_m:
        raise ValueError("top_m must be greater than bottom_m.")

    prism = np.array(
        [[
            west_m,
            east_m,
            south_m,
            north_m,
            bottom_m,
            top_m,
        ]],
        dtype=float,
    )

    density = np.array(
        [density_kg_m3],
        dtype=float,
    )

    gravity = hm.prism_gravity(
        coordinates=(
            np.array([observation_easting_m]),
            np.array([observation_northing_m]),
            np.array([observation_upward_m]),
        ),
        prisms=prism,
        density=density,
        field="g_z",
        parallel=False,
    )

    return float(gravity[0])
