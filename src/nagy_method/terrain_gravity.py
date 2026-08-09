"""
Gravity effect of terrain prisms.

This module connects terrain-prism geometry with the exact
rectangular-prism gravity calculation.
"""

from __future__ import annotations

import numpy as np

from src.nagy_method.prism_gravity import prism_vertical_gravity


def terrain_gravity_from_prisms(
    prisms: np.ndarray,
    signs: np.ndarray,
    density_kg_m3: float = 2670.0,
) -> dict:
    """
    Compute the vertical gravity effect of terrain prisms.

    The observation point is the station:
        x = 0
        y = 0
        z = 0

    Parameters
    ----------
    prisms : ndarray, shape (n, 6)
        Columns:
        west, east, south, north, bottom, top.

    signs : ndarray
        +1 for excess terrain above station level.
        -1 for terrain deficit below station level.

    density_kg_m3 : float
        Terrain density in kg/m^3.

    Returns
    -------
    dict
        Total and separated gravity contributions in mGal.
    """

    prisms = np.asarray(prisms, dtype=float)
    signs = np.asarray(signs, dtype=float)

    if prisms.ndim != 2 or prisms.shape[1] != 6:
        raise ValueError("prisms must have shape (n, 6).")

    if len(prisms) != len(signs):
        raise ValueError(
            "prisms and signs must contain the same number of elements."
        )

    contributions = np.zeros(len(prisms), dtype=float)

    for i, prism in enumerate(prisms):

        west, east, south, north, bottom, top = prism

        g = prism_vertical_gravity(
            observation_easting_m=0.0,
            observation_northing_m=0.0,
            observation_upward_m=0.0,
            west_m=west,
            east_m=east,
            south_m=south,
            north_m=north,
            bottom_m=bottom,
            top_m=top,
            density_kg_m3=density_kg_m3,
        )

        contributions[i] = signs[i] * abs(g)

    above = signs > 0
    below = signs < 0

    g_above = contributions[above].sum()
    g_below = contributions[below].sum()

    return {
        "total_mgal": float(contributions.sum()),
        "above_mgal": float(g_above),
        "below_mgal": float(g_below),
        "n_prisms": int(len(prisms)),
        "contributions_mgal": contributions,
    }
