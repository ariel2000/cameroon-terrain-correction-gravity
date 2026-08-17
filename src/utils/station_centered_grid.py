"""
Station-centred terrain utilities.

This module defines transformations used to express terrain
elevations relative to the gravity station elevation.

The station elevation is the vertical reference:

    h_rel = h_terrain - h_station

Thus:
    h_rel > 0 : terrain above station level
    h_rel < 0 : terrain below station level
    h_rel = 0 : terrain at station level
"""

from __future__ import annotations

import numpy as np


def terrain_relative_to_station(
    terrain_elevation,
    station_elevation: float,
):
    """
    Convert absolute terrain elevations to elevations relative
    to the gravity station.

    Parameters
    ----------
    terrain_elevation : array-like
        Terrain elevations in metres.

    station_elevation : float
        Gravity station elevation in metres.

    Returns
    -------
    numpy.ndarray
        Terrain elevation relative to station level, in metres.
    """

    terrain_elevation = np.asarray(
        terrain_elevation,
        dtype=float,
    )

    if not np.isfinite(station_elevation):
        raise ValueError(
            "station_elevation must be finite."
        )

    if not np.all(np.isfinite(terrain_elevation)):
        raise ValueError(
            "terrain_elevation contains non-finite values."
        )

    return terrain_elevation - float(station_elevation)


def enforce_station_reference(
    relative_elevation,
    station_index,
):
    """
    Force the terrain reference at the station location to zero.

    This function should only be applied when station_index
    identifies the terrain sample representing the station
    itself.

    Parameters
    ----------
    relative_elevation : array-like
        Terrain elevations relative to station level.

    station_index : tuple
        Index (iy, ix) of the terrain sample representing
        the station.

    Returns
    -------
    numpy.ndarray
        Copy of the relative terrain with the station sample
        exactly equal to zero.
    """

    relative_elevation = np.asarray(
        relative_elevation,
        dtype=float,
    ).copy()

    if relative_elevation.ndim != 2:
        raise ValueError(
            "relative_elevation must be a 2-D array."
        )

    iy, ix = station_index

    if not (
        0 <= iy < relative_elevation.shape[0]
        and
        0 <= ix < relative_elevation.shape[1]
    ):
        raise IndexError(
            "station_index lies outside the terrain array."
        )

    relative_elevation[iy, ix] = 0.0

    return relative_elevation
