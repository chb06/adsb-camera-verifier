from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import pymap3d as pm  # type: ignore
except Exception:  # pragma: no cover - exercised indirectly in environments without pymap3d
    pm = None


@dataclass
class CameraModel:
    """Pinhole camera model with simple extrinsics.

    Coordinate conventions (consistent, but document and stick to them):
    - ENU: +E east, +N north, +U up (meters)
    - Camera frame: +X right, +Y down, +Z forward

    R_enu_cam maps ENU vectors into camera coords.
    """

    K: np.ndarray          # 3x3
    dist: np.ndarray       # (5,) k1,k2,p1,p2,k3
    R_enu_cam: np.ndarray  # 3x3
    t_enu_cam: np.ndarray  # (3,) camera origin in ENU


@dataclass
class SiteRef:
    lat0_deg: float
    lon0_deg: float
    alt0_m: float


# WGS84 constants for the fallback geodetic->ENU conversion.
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    alt = float(alt_m)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _ecef_delta_to_enu(dx: np.ndarray, lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))

    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return rot @ dx.reshape(3,)


def _geodetic_to_enu_fallback(site: SiteRef, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    p = _geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    p0 = _geodetic_to_ecef(site.lat0_deg, site.lon0_deg, site.alt0_m)
    return _ecef_delta_to_enu(p - p0, site.lat0_deg, site.lon0_deg)


def yaw_pitch_roll_to_R_enu_cam(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build rotation matrix from ENU -> camera given yaw/pitch/roll.

    This is a *starting* convention. You may need to adjust sign/axis depending
    on how you mount the camera. We'll validate with overlay and fix if needed.

    Convention used here:
    - yaw: rotation about +U (ENU up), positive yaw rotates +N toward +E
    - pitch: rotation about +E (east), positive pitch rotates +U toward +N
    - roll: rotation about +N (north), positive roll rotates +U toward +E

    Then an additional fixed rotation maps ENU axes into camera axes.
    """
    import math

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # Rotation matrices in ENU basis
    R_yaw = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    R_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )

    R_roll = np.array(
        [
            [math.cos(roll), 0.0, math.sin(roll)],
            [0.0, 1.0, 0.0],
            [-math.sin(roll), 0.0, math.cos(roll)],
        ],
        dtype=np.float64,
    )

    R_enu_mount = R_yaw @ R_pitch @ R_roll

    # Map ENU -> camera axes (base orientation)
    # ENU basis vectors expressed in camera coords when yaw=pitch=roll=0.
    # We want camera +Z to look toward +N, camera +X toward +E, camera +Y down.
    # ENU: E,N,U  -> Camera: X,Z,-Y (because camera Y is down)
    R_enu_to_cam_base = np.array(
        [
            [1.0, 0.0, 0.0],   # E -> +X
            [0.0, 0.0, -1.0],  # U -> -Y
            [0.0, 1.0, 0.0],   # N -> +Z
        ],
        dtype=np.float64,
    )

    return R_enu_to_cam_base @ R_enu_mount


def geodetic_to_enu(site: SiteRef, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    if pm is not None:
        e, n, u = pm.geodetic2enu(lat_deg, lon_deg, alt_m, site.lat0_deg, site.lon0_deg, site.alt0_m)
        return np.array([e, n, u], dtype=np.float64)
    return _geodetic_to_enu_fallback(site, lat_deg, lon_deg, alt_m)


def project_enu_to_pixel(cam: CameraModel, p_enu: np.ndarray) -> Tuple[float, float, bool, np.ndarray]:
    """Project an ENU point into pixels.

    Returns (u, v, in_front, p_cam)
    """
    p_rel = p_enu - cam.t_enu_cam.reshape(3,)
    p_cam = cam.R_enu_cam @ p_rel

    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if z <= 0:
        return 0.0, 0.0, False, p_cam

    xn = x / z
    yn = y / z

    # Distortion ignored in MVP; add later if needed.
    u = cam.K[0, 0] * xn + cam.K[0, 2]
    v = cam.K[1, 1] * yn + cam.K[1, 2]

    return float(u), float(v), True, p_cam


def bearing_elevation_range(p_enu: np.ndarray) -> Tuple[float, float, float]:
    """Return (azimuth rad, elevation rad, range m) in ENU."""
    e, n, u = float(p_enu[0]), float(p_enu[1]), float(p_enu[2])
    rng = float(np.sqrt(e * e + n * n + u * u))
    az = float(np.arctan2(e, n))
    el = float(np.arctan2(u, np.sqrt(e * e + n * n)))
    return az, el, rng
