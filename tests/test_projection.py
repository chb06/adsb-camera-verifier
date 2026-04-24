from __future__ import annotations

import numpy as np

from geo.projection import SiteRef, geodetic_to_enu, yaw_pitch_roll_to_R_enu_cam


def test_rotation_matrix_shape():
    R = yaw_pitch_roll_to_R_enu_cam(0.0, 0.0, 0.0)
    assert R.shape == (3, 3)
    # should be orthonormal-ish
    I = R @ R.T
    assert np.allclose(I, np.eye(3), atol=1e-6)


def test_geodetic_to_enu_returns_expected_direction():
    site = SiteRef(lat0_deg=37.0, lon0_deg=-122.0, alt0_m=0.0)
    p_east = geodetic_to_enu(site, 37.0, -121.999, 0.0)
    p_north = geodetic_to_enu(site, 37.001, -122.0, 0.0)

    assert p_east[0] > 0.0
    assert abs(p_east[1]) < 50.0
    assert p_north[1] > 0.0
    assert abs(p_north[0]) < 50.0
