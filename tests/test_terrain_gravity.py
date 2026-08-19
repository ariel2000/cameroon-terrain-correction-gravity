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
    assert result["terrain_effect_mgal"] == 0.0
    assert result["terrain_correction_mgal"] == 0.0


def test_terrain_above_station():
    """
    Positive rock above the observation point pulls upward.

    Since Harmonica g_z is positive downward,
    the terrain gravity effect must be negative.
    """

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

    assert result["terrain_effect_mgal"] < 0.0
    assert result["terrain_correction_mgal"] > 0.0

    assert result["above_effect_mgal"] < 0.0
    assert result["above_correction_mgal"] > 0.0


def test_terrain_below_station():
    """
    Terrain below station level represents missing rock.

    It is therefore modelled with negative density contrast.
    The resulting terrain perturbation is also negative.
    """

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

    assert result["terrain_effect_mgal"] < 0.0
    assert result["terrain_correction_mgal"] > 0.0

    assert result["below_effect_mgal"] < 0.0
    assert result["below_correction_mgal"] > 0.0


def test_above_and_below_both_increase_correction():
    prisms = np.array([
        [
            500.0,
            1000.0,
            -250.0,
            250.0,
            0.0,
            100.0,
        ],
        [
            -1000.0,
            -500.0,
            -250.0,
            250.0,
            -100.0,
            0.0,
        ],
    ])

    signs = np.array([
        1.0,
        -1.0,
    ])

    result = terrain_gravity_from_prisms(
        prisms,
        signs,
    )

    assert result["above_correction_mgal"] > 0.0
    assert result["below_correction_mgal"] > 0.0

    assert result["terrain_correction_mgal"] > 0.0


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
        r2["terrain_correction_mgal"],
        2.0 * r1["terrain_correction_mgal"],
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


def test_invalid_sign():
    prisms = np.array([
        [0, 1, 0, 1, 0, 1],
    ])

    signs = np.array([0.0])

    with pytest.raises(ValueError):
        terrain_gravity_from_prisms(
            prisms,
            signs,
        )
