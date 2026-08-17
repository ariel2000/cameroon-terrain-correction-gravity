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

from scipy.interpolate import RegularGridInterpolator

from src.utils.coordinates import local_meters_to_degrees


def build_station_centered_grid(
    grid: dict,
    station_longitude: float,
    station_latitude: float,
    station_elevation_m: float,
    radius_m: float,
    spacing_m: float,
) -> dict:
    """
    Build a local regular terrain grid centered exactly on a gravity station.

    The station is located at:
        x = 0 m
        y = 0 m

    The original geographic terrain grid is interpolated onto a local
    Cartesian grid. The terrain node at the exact station position is
    constrained to the measured station elevation.

    Parameters
    ----------
    grid : dict
        Terrain grid returned by read_kriged_grid().

    station_longitude : float
        Station longitude in decimal degrees.

    station_latitude : float
        Station latitude in decimal degrees.

    station_elevation_m : float
        Measured station elevation in meters.

    radius_m : float
        Half-width of the local square grid in meters.

    spacing_m : float
        Local grid spacing in meters.

    Returns
    -------
    dict
        Local grid containing:
        - x_m
        - y_m
        - longitude
        - latitude
        - elevation_m
        - relative_elevation_m
        - station_index
        - spacing_m
        - radius_m
    """

    if radius_m <= 0:
        raise ValueError("radius_m must be strictly positive.")

    if spacing_m <= 0:
        raise ValueError("spacing_m must be strictly positive.")

    if spacing_m > radius_m:
        raise ValueError(
            "spacing_m must not exceed radius_m."
        )

    x_original = np.asarray(grid["x"], dtype=float)
    y_original = np.asarray(grid["y"], dtype=float)
    z_original = np.asarray(grid["z"], dtype=float)

    if z_original.shape != (
        y_original.size,
        x_original.size,
    ):
        raise ValueError(
            "Grid elevation shape is inconsistent with x and y."
        )

    # --------------------------------------------------------
    # Build symmetric local axes.
    #
    # We deliberately force zero to be one of the nodes.
    # --------------------------------------------------------

    n = int(np.floor(radius_m / spacing_m))

    x_m = np.arange(
        -n,
        n + 1,
        dtype=float,
    ) * spacing_m

    y_m = np.arange(
        -n,
        n + 1,
        dtype=float,
    ) * spacing_m

    x_mesh, y_mesh = np.meshgrid(
        x_m,
        y_m,
    )

    # --------------------------------------------------------
    # Convert local metric coordinates back to geographic
    # coordinates for interpolation of the Surfer grid.
    # --------------------------------------------------------

    longitude, latitude = local_meters_to_degrees(
        x_m=x_mesh,
        y_m=y_mesh,
        reference_longitude=station_longitude,
        reference_latitude=station_latitude,
    )

    # --------------------------------------------------------
    # Bilinear interpolation of original terrain.
    # --------------------------------------------------------

    interpolator = RegularGridInterpolator(
        (y_original, x_original),
        z_original,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    query_points = np.column_stack(
        (
            latitude.ravel(),
            longitude.ravel(),
        )
    )

    elevation = interpolator(
        query_points
    ).reshape(x_mesh.shape)

    if np.any(~np.isfinite(elevation)):
        raise ValueError(
            "The requested local grid extends outside the terrain grid."
        )

    # --------------------------------------------------------
    # Exact station node.
    # --------------------------------------------------------

    center = n

    station_index = (
        center,
        center,
    )

    elevation[
        station_index
    ] = float(station_elevation_m)

    relative_elevation = (
        elevation
        - float(station_elevation_m)
    )

    # Numerical guarantee
    relative_elevation[
        station_index
    ] = 0.0

    return {
        "x_m": x_m,
        "y_m": y_m,
        "x_mesh_m": x_mesh,
        "y_mesh_m": y_mesh,
        "longitude": longitude,
        "latitude": latitude,
        "elevation_m": elevation,
        "relative_elevation_m": relative_elevation,
        "station_index": station_index,
        "spacing_m": float(spacing_m),
        "radius_m": float(radius_m),
    }
