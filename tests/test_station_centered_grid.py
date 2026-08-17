import numpy as np
import pytest

from src.utils.station_centered_grid import (
    terrain_relative_to_station,
    enforce_station_reference,
)


def test_relative_elevation_basic():
    terrain = np.array([
        [100.0, 120.0],
        [80.0, 110.0],
    ])

    result = terrain_relative_to_station(
        terrain_elevation=terrain,
        station_elevation=100.0,
    )

    expected = np.array([
        [0.0, 20.0],
        [-20.0, 10.0],
    ])

    assert np.allclose(result, expected)


def test_station_elevation_not_zero():
    terrain = np.array([50.0, 75.0, 100.0])

    result = terrain_relative_to_station(
        terrain_elevation=terrain,
        station_elevation=75.0,
    )

    expected = np.array([-25.0, 0.0, 25.0])

    assert np.allclose(result, expected)


def test_invalid_station_elevation():
    terrain = np.array([100.0, 120.0])

    with pytest.raises(ValueError):
        terrain_relative_to_station(
            terrain_elevation=terrain,
            station_elevation=np.nan,
        )


def test_invalid_terrain_values():
    terrain = np.array([
        [100.0, np.nan],
        [120.0, 130.0],
    ])

    with pytest.raises(ValueError):
        terrain_relative_to_station(
            terrain_elevation=terrain,
            station_elevation=100.0,
        )


def test_enforce_station_reference():
    relative = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    result = enforce_station_reference(
        relative_elevation=relative,
        station_index=(1, 0),
    )

    assert np.isclose(result[1, 0], 0.0)

    # Other values must remain unchanged
    assert np.isclose(result[0, 0], 10.0)
    assert np.isclose(result[0, 1], 20.0)
    assert np.isclose(result[1, 1], 40.0)


def test_original_array_not_modified():
    relative = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    original = relative.copy()

    enforce_station_reference(
        relative_elevation=relative,
        station_index=(0, 1),
    )

    assert np.array_equal(relative, original)


def test_invalid_station_index():
    relative = np.zeros((2, 2))

    with pytest.raises(IndexError):
        enforce_station_reference(
            relative_elevation=relative,
            station_index=(3, 0),
        )


def test_requires_two_dimensional_array():
    relative = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        enforce_station_reference(
            relative_elevation=relative,
            station_index=(0, 0),
        )
from src.utils.station_centered_grid import (
    build_station_centered_grid,
)


def make_simple_geographic_grid():
    x = np.linspace(9.0, 9.1, 11)
    y = np.linspace(4.0, 4.1, 11)

    xx, yy = np.meshgrid(x, y)

    # Simple planar synthetic terrain
    z = (
        100.0
        + 10.0 * (xx - 9.0)
        + 20.0 * (yy - 4.0)
    )

    return {
        "x": x,
        "y": y,
        "z": z,
    }


def test_station_is_exactly_centered():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    iy, ix = local["station_index"]

    assert np.isclose(
        local["x_mesh_m"][iy, ix],
        0.0,
    )

    assert np.isclose(
        local["y_mesh_m"][iy, ix],
        0.0,
    )


def test_station_elevation_is_enforced():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    idx = local["station_index"]

    assert np.isclose(
        local["elevation_m"][idx],
        150.0,
    )

    assert np.isclose(
        local["relative_elevation_m"][idx],
        0.0,
    )


def test_local_grid_is_symmetric():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    assert np.isclose(
        local["x_m"][0],
        -local["x_m"][-1],
    )

    assert np.isclose(
        local["y_m"][0],
        -local["y_m"][-1],
    )


def test_expected_local_grid_size():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    # -1000 ... 0 ... +1000 every 250 m
    # gives 9 nodes in each direction
    assert local["elevation_m"].shape == (9, 9)


def test_invalid_local_grid_parameters():
    grid = make_simple_geographic_grid()

    with pytest.raises(ValueError):
        build_station_centered_grid(
            grid=grid,
            station_longitude=9.05,
            station_latitude=4.05,
            station_elevation_m=150.0,
            radius_m=1000.0,
            spacing_m=-100.0,
        )
        from src.utils.station_centered_grid import (
    build_station_centered_grid,
)


def make_simple_geographic_grid():
    x = np.linspace(9.0, 9.1, 11)
    y = np.linspace(4.0, 4.1, 11)

    xx, yy = np.meshgrid(x, y)

    # Simple planar synthetic terrain
    z = (
        100.0
        + 10.0 * (xx - 9.0)
        + 20.0 * (yy - 4.0)
    )

    return {
        "x": x,
        "y": y,
        "z": z,
    }


def test_station_is_exactly_centered():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    iy, ix = local["station_index"]

    assert np.isclose(
        local["x_mesh_m"][iy, ix],
        0.0,
    )

    assert np.isclose(
        local["y_mesh_m"][iy, ix],
        0.0,
    )


def test_station_elevation_is_enforced():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    idx = local["station_index"]

    assert np.isclose(
        local["elevation_m"][idx],
        150.0,
    )

    assert np.isclose(
        local["relative_elevation_m"][idx],
        0.0,
    )


def test_local_grid_is_symmetric():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    assert np.isclose(
        local["x_m"][0],
        -local["x_m"][-1],
    )

    assert np.isclose(
        local["y_m"][0],
        -local["y_m"][-1],
    )


def test_expected_local_grid_size():
    grid = make_simple_geographic_grid()

    local = build_station_centered_grid(
        grid=grid,
        station_longitude=9.05,
        station_latitude=4.05,
        station_elevation_m=150.0,
        radius_m=1000.0,
        spacing_m=250.0,
    )

    # -1000 ... 0 ... +1000 every 250 m
    # gives 9 nodes in each direction
    assert local["elevation_m"].shape == (9, 9)


def test_invalid_local_grid_parameters():
    grid = make_simple_geographic_grid()

    with pytest.raises(ValueError):
        build_station_centered_grid(
            grid=grid,
            station_longitude=9.05,
            station_latitude=4.05,
            station_elevation_m=150.0,
            radius_m=1000.0,
            spacing_m=-100.0,
        )
