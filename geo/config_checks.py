from __future__ import annotations

from typing import Any, Dict, List


def collect_projection_warnings(cfg: Dict[str, Any]) -> List[str]:
    """Return plain-English warnings for obviously placeholder site/calibration settings.

    These warnings are intentionally conservative. They do not block runtime; they only
    explain why ADS-B projection may not line up yet.
    """

    warnings: List[str] = []

    site = cfg.get('site', {}) or {}
    lat0 = float(site.get('lat0_deg', 0.0) or 0.0)
    lon0 = float(site.get('lon0_deg', 0.0) or 0.0)
    alt0 = float(site.get('alt0_m', 0.0) or 0.0)
    if abs(lat0) < 1e-9 and abs(lon0) < 1e-9 and abs(alt0) < 1e-6:
        warnings.append(
            'Site reference is still the 0,0,0 placeholder. ADS-B track projection will not be useful until you set your real camera location.'
        )

    cal = cfg.get('calibration', {}) or {}
    if bool(cal.get('use_fov_guess', False)):
        warnings.append(
            'Camera intrinsics are still using an FOV guess. That is fine for software bring-up, but not accurate enough for matching aircraft to ADS-B reliably.'
        )

    yaw = float(cal.get('yaw_deg', 0.0) or 0.0)
    pitch = float(cal.get('pitch_deg', 0.0) or 0.0)
    roll = float(cal.get('roll_deg', 0.0) or 0.0)
    if abs(yaw) < 1e-6 and abs(pitch) < 1e-6 and abs(roll) < 1e-6:
        warnings.append(
            'Camera yaw/pitch/roll are all still 0. ADS-B ROIs will only line up by luck until you measure or estimate the camera pointing direction.'
        )

    return warnings


def required_site_measurements() -> List[str]:
    return [
        'Camera latitude, longitude, and altitude (or as close as you can get from phone GPS / maps).',
        'Camera mounting height above ground.',
        'Which direction the camera is pointing (rough compass heading / yaw).',
        'Whether the camera is tilted upward or downward (pitch).',
        'A note about whether the horizon is level or the camera is rotated (roll).',
    ]
