"""
Coordinate utilities for terrain correction calculations.

All input station and grid coordinates are geographic:
- longitude in degrees
- latitude in degrees
- elevation in meters

This module converts geographic coordinate differences into
local metric distances suitable for gravity terrain correction.
"""

from __future__ import annotations

import numpy as np


EARTH_RADIUS_M = 6_371_000.0


def degrees_to_local_meters(
    longitude: float | np.ndarray,
    latitude: float | np.ndarray,
    reference_longitude: float,
    reference_latitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert longitude/latitude differences into local Cartesian offsets.

    Parameters
    ----------
    longitude : float or ndarray
        Longitude(s) in decimal degrees.

    latitude : float or ndarray
        Latitude(s) in decimal degrees.

    reference_longitude : float
        Longitude of the local origin, in decimal degrees.

    reference_latitude : float
        Latitude of the local origin, in decimal degrees.

    Returns
    -------
    x_m, y_m : ndarray
        Local east-west and north-south distances in meters.

    Notes
    -----
    This uses a local spherical-Earth approximation:

        dx = R * cos(latitude0) * dlon
        dy = R * dlat

    where angular differences are expressed in radians.

    This approximation is appropriate for local terrain-correction
    calculations and avoids treating angular degrees as linear distances.
    """
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)

    lon0_rad = np.deg2rad(reference_longitude)
    lat0_rad = np.deg2rad(reference_latitude)

    lon_rad = np.deg2rad(longitude)
    lat_rad = np.deg2rad(latitude)

    dlon = lon_rad - lon0_rad
    dlat = lat_rad - lat0_rad

    x_m = EARTH_RADIUS_M * np.cos(lat0_rad) * dlon
    y_m = EARTH_RADIUS_M * dlat

    return x_m, y_m


def horizontal_distance_m(
    longitude: float | np.ndarray,
    latitude: float | np.ndarray,
    reference_longitude: float,
    reference_latitude: float,
) -> np.ndarray:
    """
    Compute horizontal distance from a reference point.

    Returns
    -------
    ndarray
        Horizontal distance in meters.
    """
    x_m, y_m = degrees_to_local_meters(
        longitude=longitude,
        latitude=latitude,
        reference_longitude=reference_longitude,
        reference_latitude=reference_latitude,
    )

    return np.hypot(x_m, y_m)


def distance_3d_m(
    longitude: float | np.ndarray,
    latitude: float | np.ndarray,
    elevation_m: float | np.ndarray,
    reference_longitude: float,
    reference_latitude: float,
    reference_elevation_m: float,
) -> np.ndarray:
    """
    Compute 3D distance from a reference station.

    Returns
    -------
    ndarray
        Three-dimensional distance in meters.
    """
    x_m, y_m = degrees_to_local_meters(
        longitude=longitude,
        latitude=latitude,
        reference_longitude=reference_longitude,
        reference_latitude=reference_latitude,
    )

    elevation_m = np.asarray(elevation_m, dtype=float)
    dz_m = elevation_m - float(reference_elevation_m)

    return np.sqrt(x_m**2 + y_m**2 + dz_m**2)
