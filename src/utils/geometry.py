import numpy as np

EARTH_RADIUS_M = 6371008.8


def local_distances(lon_grid, lat_grid, lon_station, lat_station):
    """
    Compute local horizontal distances between a gravity station
    and terrain-grid nodes.

    Parameters
    ----------
    lon_grid : array_like
        Grid longitudes in degrees.
    lat_grid : array_like
        Grid latitudes in degrees.
    lon_station : float
        Station longitude in degrees.
    lat_station : float
        Station latitude in degrees.

    Returns
    -------
    dx : ndarray
        East-West distances from station, in meters.
    dy : ndarray
        North-South distances from station, in meters.
    r : ndarray
        Horizontal radial distances from station, in meters.
    """

    lon_grid = np.asarray(lon_grid, dtype=float)
    lat_grid = np.asarray(lat_grid, dtype=float)

    lat0 = np.deg2rad(lat_station)

    dx = (
        EARTH_RADIUS_M
        * np.cos(lat0)
        * np.deg2rad(lon_grid - lon_station)
    )

    dy = (
        EARTH_RADIUS_M
        * np.deg2rad(lat_grid - lat_station)
    )

    r = np.sqrt(dx**2 + dy**2)

    return dx, dy, r
