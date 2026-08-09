import numpy as np
import pytest

from src.nagy_method.prism_gravity import prism_vertical_gravity


def test_positive_prism_attraction():
    gravity = prism_vertical_gravity(
        observation_easting_m=0.0,
        observation_northing_m=0.0,
        observation_upward_m=1000.0,
        west_m=-500.0,
        east_m=500.0,
        south_m=-500.0,
        north_m=500.0,
        bottom_m=0.0,
        top_m=500.0,
        density_kg_m3=2670.0,
    )

    assert gravity > 0.0


def test_zero_density_gives_zero_gravity():
    gravity = prism_vertical_gravity(
        observation_easting_m=0.0,
        observation_northing_m=0.0,
        observation_upward_m=1000.0,
        west_m=-500.0,
        east_m=500.0,
        south_m=-500.0,
        north_m=500.0,
        bottom_m=0.0,
        top_m=500.0,
        density_kg_m3=0.0,
    )

    assert np.isclose(gravity, 0.0)


def test_density_linearity():
    g1 = prism_vertical_gravity(
        0.0, 0.0, 1000.0,
        -500.0, 500.0,
        -500.0, 500.0,
        0.0, 500.0,
        1000.0,
    )

    g2 = prism_vertical_gravity(
        0.0, 0.0, 1000.0,
        -500.0, 500.0,
        -500.0, 500.0,
        0.0, 500.0,
        2000.0,
    )

    assert np.isclose(g2, 2.0 * g1)


def test_invalid_horizontal_bounds():
    with pytest.raises(ValueError):
        prism_vertical_gravity(
            0.0, 0.0, 1000.0,
            500.0, -500.0,
            -500.0, 500.0,
            0.0, 500.0,
            2670.0,
        )


def test_invalid_vertical_bounds():
    with pytest.raises(ValueError):
        prism_vertical_gravity(
            0.0, 0.0, 1000.0,
            -500.0, 500.0,
            -500.0, 500.0,
            500.0, 0.0,
            2670.0,
        )
