import numpy as np
import pytest

from src.nagy_method.terrain_gravity import (
    terrain_gravity_from_prisms,
)


def test_empty_prisms():
    prisms = np.empty((0, 6))
    signs = np.empty(0)

    result = terrain_gravity_from_prisms(
        prisms,
        signs,
    )

    assert result["n_prisms"] == 0
    assert result["total_mgal"] == 0.0


def test_positive_terrain():
    prisms = np.array([
        [
            500.0,
            1000.0,
            -250.0,
            250.0,
            0.0,
            100.0,
        ]
    ])

    signs = np.array([1.0])

    result = terrain_gravity_from_prisms(
        prisms,
        signs,
    )

    assert result["total_mgal"] > 0.0
    assert result["above_mgal"] > 0.0
    assert result["below_mgal"] == 0.0


def test_negative_terrain():
    prisms = np.array([
        [
            500.0,
            1000.0,
            -250.0,
            250.0,
            -100.0,
            0.0,
        ]
    ])

    signs = np.array([-1.0])

    result = terrain_gravity_from_prisms(
        prisms,
        signs,
    )

    assert result["total_mgal"] < 0.0
    assert result["below_mgal"] < 0.0


def test_density_linearity():
    prisms = np.array([
        [
            500.0,
            1000.0,
            -250.0,
            250.0,
            0.0,
            100.0,
        ]
    ])

    signs = np.array([1.0])

    r1 = terrain_gravity_from_prisms(
        prisms,
        signs,
        density_kg_m3=1000.0,
    )

    r2 = terrain_gravity_from_prisms(
        prisms,
        signs,
        density_kg_m3=2000.0,
    )

    assert np.isclose(
        r2["total_mgal"],
        2.0 * r1["total_mgal"],
    )


def test_incompatible_lengths():
    prisms = np.array([
        [0, 1, 0, 1, 0, 1],
        [1, 2, 1, 2, 0, 1],
    ])

    signs = np.array([1.0])

    with pytest.raises(ValueError):
        terrain_gravity_from_prisms(
            prisms,
            signs,
        )
