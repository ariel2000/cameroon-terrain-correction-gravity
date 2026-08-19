"""
Gravity effect and terrain correction from rectangular terrain prisms.

The station is the local observation point:

    x = 0
    y = 0
    z = 0

Terrain is represented relative to the horizontal plane passing
through the gravity station.

Sign convention
---------------
Harmonica returns g_z positive downward.

For terrain relative to the station-level reference plane:

    sign = +1
        excess rock above station level
        density contrast = +rho

    sign = -1
        missing rock below station level
        density contrast = -rho

The terrain gravity effect is the signed perturbation relative
to the reference horizontal slab.

The terrain correction is defined here as:

    TC = - terrain_gravity_effect

and is returned in mGal.
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
    Compute terrain gravity effect and terrain correction.

    Parameters
    ----------
    prisms : ndarray, shape (n, 6)
        Prism boundaries:
        west, east, south, north, bottom, top.

    signs : ndarray
        +1 for excess terrain above station level.
        -1 for missing terrain below station level.

    density_kg_m3 : float
        Positive reference rock density in kg/m^3.

    Returns
    -------
    dict
        Signed gravity effects and corresponding terrain correction
        in mGal.
    """

    prisms = np.asarray(prisms, dtype=float)
    signs = np.asarray(signs, dtype=float)

    if prisms.ndim != 2 or prisms.shape[1] != 6:
        raise ValueError(
            "prisms must have shape (n, 6)."
        )

    if len(prisms) != len(signs):
        raise ValueError(
            "prisms and signs must contain the same number of elements."
        )

    if density_kg_m3 < 0:
        raise ValueError(
            "density_kg_m3 must be non-negative."
        )

    if not np.all(
        np.isin(signs, [-1.0, 1.0])
    ):
        raise ValueError(
            "signs must contain only +1 or -1."
        )

    contributions = np.zeros(
        len(prisms),
        dtype=float,
    )

    for i, prism in enumerate(prisms):

        west, east, south, north, bottom, top = prism

        density_contrast = (
            density_kg_m3 * signs[i]
        )

        contributions[i] = prism_vertical_gravity(
            observation_easting_m=0.0,
            observation_northing_m=0.0,
            observation_upward_m=0.0,
            west_m=west,
            east_m=east,
            south_m=south,
            north_m=north,
            bottom_m=bottom,
            top_m=top,
            density_kg_m3=density_contrast,
        )

    above = signs > 0
    below = signs < 0

    above_effect = contributions[above].sum()
    below_effect = contributions[below].sum()

    terrain_effect = contributions.sum()

    terrain_correction = -terrain_effect

    return {
        "terrain_effect_mgal": float(terrain_effect),
        "terrain_correction_mgal": float(terrain_correction),

        "above_effect_mgal": float(above_effect),
        "below_effect_mgal": float(below_effect),

        "above_correction_mgal": float(-above_effect),
        "below_correction_mgal": float(-below_effect),

        "n_prisms": int(len(prisms)),
        "contributions_mgal": contributions,
    }
