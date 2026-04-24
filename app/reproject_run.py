from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from config import deep_update, load_yaml
from data.replay import ReplayRun
from geo.config_checks import collect_projection_warnings
from geo.projection import (
    CameraModel,
    SiteRef,
    bearing_elevation_range,
    geodetic_to_enu,
    project_enu_to_pixel,
    yaw_pitch_roll_to_R_enu_cam,
)


@dataclass
class Track:
    raw: dict

    def predict_enu(self, site: SiteRef, t: float) -> np.ndarray:
        p0 = geodetic_to_enu(site, float(self.raw['lat_deg']), float(self.raw['lon_deg']), float(self.raw['alt_m']))
        dt = float(t - float(self.raw['t_rx']))
        v_enu = np.array(
            [
                float(self.raw.get('ve_mps', 0.0)),
                float(self.raw.get('vn_mps', 0.0)),
                float(self.raw.get('vu_mps', 0.0)),
            ],
            dtype=np.float64,
        )
        return p0 + v_enu * dt


def roi_half_sizes_px(range_m: float, fx: float, fy: float, cfg_roi: dict) -> tuple[int, int]:
    sigma_m = float(cfg_roi.get('default_sigma_m', 80.0))
    k = float(cfg_roi.get('k_sigma', 3.0))
    obj = float(cfg_roi.get('obj_margin_px', 48))
    min_hw = int(cfg_roi.get('min_hw_px', 160))
    min_hh = int(cfg_roi.get('min_hh_px', 160))

    if range_m < 1.0:
        range_m = 1.0

    sigma_u = fx * (sigma_m / range_m)
    sigma_v = fy * (sigma_m / range_m)
    hw = int(max(min_hw, k * sigma_u + obj))
    hh = int(max(min_hh, k * sigma_v + obj))
    return hw, hh


def clamp_roi(u: float, v: float, hw: int, hh: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(w - 1, u - hw)))
    y1 = int(max(0, min(h - 1, v - hh)))
    x2 = int(max(0, min(w - 1, u + hw)))
    y2 = int(max(0, min(h - 1, v + hh)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def reshape_roi(x1: int, y1: int, x2: int, y2: int, w: int, h: int, cfg_roi: dict) -> tuple[int, int, int, int]:
    width_scale = float(cfg_roi.get('width_scale', 1.0))
    height_scale = float(cfg_roi.get('height_scale', 1.0))
    anchor_y = str(cfg_roi.get('anchor_y', 'center')).strip().lower()

    width_scale = min(1.0, max(0.1, width_scale))
    height_scale = min(1.0, max(0.1, height_scale))

    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    new_w = max(1, int(round(roi_w * width_scale)))
    new_h = max(1, int(round(roi_h * height_scale)))

    cx = x1 + roi_w / 2.0
    new_x1 = int(round(cx - new_w / 2.0))
    new_x2 = new_x1 + new_w

    if anchor_y == 'top':
        new_y1 = y1
        new_y2 = y1 + new_h
    elif anchor_y == 'bottom':
        new_y2 = y2
        new_y1 = y2 - new_h
    else:
        cy = y1 + roi_h / 2.0
        new_y1 = int(round(cy - new_h / 2.0))
        new_y2 = new_y1 + new_h

    new_x1 = max(0, min(w - 1, new_x1))
    new_y1 = max(0, min(h - 1, new_y1))
    new_x2 = max(0, min(w - 1, new_x2))
    new_y2 = max(0, min(h - 1, new_y2))
    if new_x2 <= new_x1:
        new_x2 = min(w - 1, new_x1 + 1)
    if new_y2 <= new_y1:
        new_y2 = min(h - 1, new_y1 + 1)
    return new_x1, new_y1, new_x2, new_y2


def build_config(run_dir: Path, override: str | None) -> Dict[str, Any]:
    rr = ReplayRun(run_dir)
    cfg = dict(rr.metadata.get('config', {}) or {})
    if not cfg:
        raise SystemExit(f'No config snapshot found in {run_dir / "metadata.json"}')
    if override:
        cfg = deep_update(cfg, load_yaml(override))
    return cfg


def camera_wh(run_dir: Path, rr: ReplayRun, cfg: Dict[str, Any]) -> tuple[int, int]:
    camera_meta = rr.metadata.get('camera', {}) if isinstance(rr.metadata, dict) else {}
    w = int(camera_meta.get('actual_width') or camera_meta.get('requested_width') or 0)
    h = int(camera_meta.get('actual_height') or camera_meta.get('requested_height') or 0)
    if w > 0 and h > 0:
        return w, h

    if rr.video_exists():
        cap = cv2.VideoCapture(str(run_dir / 'video.mp4'))
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if w > 0 and h > 0:
            return w, h

    raise SystemExit('Could not determine video width/height from metadata or video.mp4')


def build_camera_model(cfg: Dict[str, Any], w: int, h: int) -> CameraModel:
    cal = cfg['calibration']
    use_fov_guess = bool(cal.get('use_fov_guess', False))
    if use_fov_guess:
        import math

        hfov = math.radians(float(cal.get('hfov_deg', 41.0)))
        vfov = math.radians(float(cal.get('vfov_deg', 31.0)))
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        fy = (h / 2.0) / math.tan(vfov / 2.0)
        cx = w / 2.0
        cy = h / 2.0
    else:
        fx = float(cal['fx'])
        fy = float(cal['fy'])
        cx = float(cal['cx'])
        cy = float(cal['cy'])

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(cal.get('dist', [0, 0, 0, 0, 0]), dtype=np.float64)
    R = yaw_pitch_roll_to_R_enu_cam(
        float(cal.get('yaw_deg', 0.0)),
        float(cal.get('pitch_deg', 0.0)),
        float(cal.get('roll_deg', 0.0)),
    )
    return CameraModel(K=K, dist=dist, R_enu_cam=R, t_enu_cam=np.zeros(3, dtype=np.float64))


def reproject_run(run_dir: Path, override: str | None, stale_s: float, overwrite: bool):
    rr = ReplayRun(run_dir)
    cfg = build_config(run_dir, override)
    w, h = camera_wh(run_dir, rr, cfg)
    site = SiteRef(
        lat0_deg=float(cfg['site']['lat0_deg']),
        lon0_deg=float(cfg['site']['lon0_deg']),
        alt0_m=float(cfg['site']['alt0_m']),
    )
    cam_model = build_camera_model(cfg, w, h)
    cfg_roi = cfg.get('roi', {}) or {}
    warnings = collect_projection_warnings(cfg)

    track_path = run_dir / 'track_frames.jsonl'
    roi_path = run_dir / 'roi_index.jsonl'
    if not overwrite and (track_path.exists() or roi_path.exists()):
        raise SystemExit('Output files already exist. Pass --overwrite to replace them.')

    rows_written = 0
    roi_written = 0
    unique_icaos: set[str] = set()
    lead_s = float(cfg.get('adsb', {}).get('lead_s', 0.0))

    with track_path.open('w') as f_track, roi_path.open('w') as f_roi:
        for frame_rec in rr.frames:
            latest_rows = rr.latest_adsb_states(frame_rec.t_frame, stale_s=stale_s)
            for raw in latest_rows:
                tr = Track(raw=raw)
                p_enu = tr.predict_enu(site, frame_rec.t_frame + lead_s)
                u, v, in_front, _p_cam = project_enu_to_pixel(cam_model, p_enu)
                if not in_front:
                    continue
                if u < -200 or u > w + 200 or v < -200 or v > h + 200:
                    continue

                az_rad, el_rad, rng = bearing_elevation_range(p_enu)
                hw, hh = roi_half_sizes_px(rng, float(cam_model.K[0, 0]), float(cam_model.K[1, 1]), cfg_roi)
                x1, y1, x2, y2 = clamp_roi(u, v, hw, hh, w, h)
                x1, y1, x2, y2 = reshape_roi(x1, y1, x2, y2, w, h, cfg_roi)
                icao24 = str(raw.get('icao24', ''))
                flight = str(raw.get('flight', '') or '').strip()

                track_rec = {
                    'frame_id': int(frame_rec.frame_id),
                    'video_frame_idx': int(frame_rec.video_frame_idx),
                    't_frame': float(frame_rec.t_frame),
                    't_rel_s': float(frame_rec.t_rel_s),
                    'icao24': icao24,
                    'flight': flight,
                    'source_t_rx': float(raw.get('t_rx', 0.0)),
                    'state_age_s': float(frame_rec.t_frame - float(raw.get('t_rx', 0.0))),
                    'lat_deg': float(raw.get('lat_deg', 0.0)),
                    'lon_deg': float(raw.get('lon_deg', 0.0)),
                    'alt_m': float(raw.get('alt_m', 0.0)),
                    'range_m': float(rng),
                    'az_deg': float(np.degrees(az_rad)),
                    'el_deg': float(np.degrees(el_rad)),
                    'pixel_center': [float(u), float(v)],
                    'roi_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                    'roi_path': None,
                    'yolo_backend': 'offline_reproject',
                    'detected_this_frame': False,
                    'vision_confirmed': False,
                    'audio_ok': False,
                    'decision': 'UNKNOWN',
                    'reason': 'offline_reproject',
                    'detection_count': 0,
                    'detections': [],
                }
                f_track.write(json.dumps(track_rec) + '\n')
                f_roi.write(
                    json.dumps(
                        {
                            'icao24': icao24,
                            'frame_id': int(frame_rec.frame_id),
                            't_frame': float(frame_rec.t_frame),
                            'roi_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                            'path': None,
                        }
                    )
                    + '\n'
                )
                rows_written += 1
                roi_written += 1
                if icao24:
                    unique_icaos.add(icao24)

    print(f'[reproject] run_dir={run_dir}')
    print(f'[reproject] override={override}')
    print(f'[reproject] frames={len(rr.frames)} track_frames_written={rows_written} roi_records_written={roi_written}')
    print(f'[reproject] unique_icaos={len(unique_icaos)} video_wh={w}x{h} stale_s={stale_s:.1f} lead_s={lead_s:.1f}')
    if warnings:
        for warning in warnings:
            print(f'[reproject] warning: {warning}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--override', default=None, help='Optional YAML override to merge onto the run metadata config')
    ap.add_argument('--stale-s', type=float, default=15.0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    reproject_run(Path(args.run_dir), args.override, args.stale_s, args.overwrite)


if __name__ == '__main__':
    main()
