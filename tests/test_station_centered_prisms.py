import numpy as np
import pytest

from src.nagy_method.station_centered_prisms import (
    build_station_centered_prisms,
)


def make_local_grid():
    return {
        "x_m": np.array(
            [-500.0, 0.0, 500.0]
        ),
        "y_m": np.array(
            [-500.0, 0.0, 500.0]
        ),
        "relative_elevation_m": np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 10.0],
            ]
        ),
    }


def test_number_of_cells():
    local = make_local_grid()

    prisms, signs, radius = (
        build_station_centered_prisms(local)
    )

    # 3 x 3 nodes -> 2 x 2 cells
    assert len(prisms) == 4
    assert len(signs) == 4
    assert len(radius) == 4


def test_station_is_cell_vertex_not_cell_center():
    local = make_local_grid()

    prisms, _, _ = (
        build_station_centered_prisms(local)
    )

    # Every cell touches the origin through a boundary,
    # but no cell has the origin as its geometric centre.
    centers_x = (
        prisms[:, 0] + prisms[:, 1]
    ) / 2.0

    centers_y = (
        prisms[:, 2] + prisms[:, 3]
    ) / 2.0

    centers_r = np.hypot(
        centers_x,
        centers_y,
    )

    assert np.all(centers_r > 0.0)


def test_cell_height_uses_four_corner_mean():
    local = make_local_grid()

    prisms, _, _ = (
        build_station_centered_prisms(local)
    )

    # First cell corners:
    # 10, 10, 10, 0
    expected_height = 7.5

    assert np.isclose(
        prisms[0, 5] - prisms[0, 4],
        expected_height,
    )


def test_positive_terrain_sign():
    local = make_local_grid()

    _, signs, _ = (
        build_station_centered_prisms(local)
    )

    assert np.all(signs == 1.0)


def test_radius_selection():
    local = make_local_grid()

    prisms, signs, radius = (
        build_station_centered_prisms(
            local,
            max_radius_m=400.0,
        )
    )

    # All four cell centres are at
    # sqrt(250² + 250²) ≈ 353.6 m
    assert len(prisms) == 4
    assert np.all(radius <= 400.0)


def test_invalid_radius():
    local = make_local_grid()

    with pytest.raises(ValueError):
        build_station_centered_prisms(
            local,
            max_radius_m=0.0,
        )
