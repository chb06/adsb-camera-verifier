from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from rich.console import Console
except Exception:  # pragma: no cover - exercised in minimal environments without rich
    class Console:  # type: ignore
        def print(self, *args, **kwargs):
            text = ' '.join(str(a) for a in args)
            text = re.sub(r'\[/?[^\]]+\]', '', text)
            print(text)

from config import load_config
from common_types import AdsbState
from data.logger import RunLogger, RunLoggerConfig
from fusion.state_machine import Evidence, FusionConfig, FusionStateMachine, TrackDecision
from geo.config_checks import collect_projection_warnings
from geo.projection import (
    CameraModel,
    SiteRef,
    bearing_elevation_range,
    geodetic_to_enu,
    project_enu_to_pixel,
    yaw_pitch_roll_to_R_enu_cam,
)
from perception.vision_persistence import VisionPersistConfig, VisionPersistence
from perception.yolo_trt import YoloConfig, YoloDetector
from sensors.adsb_ingest import AdsbIngestConfig, AdsbIngestor
from sensors.camera_capture import CameraConfig, CameraStream, mode_to_wh_fps

console = Console()


@dataclass
class Track:
    last: AdsbState
    history: deque[AdsbState] = field(default_factory=lambda: deque(maxlen=12))

    def __post_init__(self):
        if not self.history:
            self.history.append(self.last)

    def update(self, msg: AdsbState):
        self.last = msg
        self.history.append(msg)

    def last_enu(self, site: SiteRef) -> np.ndarray:
        return geodetic_to_enu(site, self.last.lat_deg, self.last.lon_deg, self.last.alt_m)

    def estimate_velocity_enu(self, site: SiteRef) -> tuple[np.ndarray, str]:
        direct_v = np.array([self.last.ve_mps, self.last.vn_mps, self.last.vu_mps], dtype=np.float64)
        if float(np.hypot(direct_v[0], direct_v[1])) > 1.0:
            return direct_v, 'adsb_velocity'

        last_p = self.last_enu(site)
        for prev in reversed(list(self.history)[:-1]):
            dt = float(self.last.t_rx - prev.t_rx)
            if dt <= 0.05:
                continue
            prev_p = geodetic_to_enu(site, prev.lat_deg, prev.lon_deg, prev.alt_m)
            delta = last_p - prev_p
            if float(np.hypot(delta[0], delta[1])) < 1.0:
                continue
            v_hist = delta / dt
            return v_hist.astype(np.float64), 'history_fallback'

        return direct_v, 'none'

    def predict_enu(self, site: SiteRef, t: float) -> tuple[np.ndarray, np.ndarray, float, str]:
        p0 = self.last_enu(site)
        dt = max(0.0, float(t - self.last.t_rx))
        v_enu, motion_source = self.estimate_velocity_enu(site)
        return p0 + v_enu * dt, v_enu, dt, motion_source


@dataclass
class TrackSummary:
    icao24: str
    flight: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    age_s: float
    range_m: float


@dataclass
class AdsbRuntimeStats:
    total_messages: int = 0
    last_message_t: Optional[float] = None
    unique_icaos: set[str] = field(default_factory=set)


@dataclass
class ProjectionDebugInfo:
    icao24: str
    age_s: float
    u: Optional[float]
    v: Optional[float]
    center_in_frame: bool
    roi_xyxy: Optional[Tuple[int, int, int, int]]
    roi_intersects_frame: bool
    roi_w_px: Optional[int]
    roi_h_px: Optional[int]
    az_deg: Optional[float]
    el_deg: Optional[float]
    latency_extrapolated: bool
    extrapolated_dt_s: float
    horizontal_speed_mps: float
    motion_source: str
    last_enu_m: Tuple[float, float, float]
    predicted_enu_m: Tuple[float, float, float]
    can_draw_roi: bool
    skip_reason: Optional[str] = None


@dataclass
class OrientationDebugInfo:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    use_fov_guess: bool
    hfov_deg: Optional[float]
    vfov_deg: Optional[float]


@dataclass
class MatchedDrawState:
    frame_id: int
    xyxy_frame: Tuple[int, int, int, int]
    cls: str
    conf: float


def roi_half_sizes_px(
    range_m: float,
    fx: float,
    fy: float,
    cfg_roi: dict,
    cfg_projection: dict,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int]:
    sigma_m = float(cfg_roi.get('default_sigma_m', 80.0))
    k = float(cfg_roi.get('k_sigma', 3.0))
    roi_scale = float(cfg_projection.get('roi_scale', 0.6))
    roi_padding_px = float(cfg_projection.get('roi_padding_px', 24.0))
    roi_max_fraction_of_frame = float(cfg_projection.get('roi_max_fraction_of_frame', 0.35))

    if range_m < 1.0:
        range_m = 1.0

    sigma_u = fx * (sigma_m / range_m)
    sigma_v = fy * (sigma_m / range_m)

    hw = int(max(18.0, roi_scale * (k * sigma_u) + roi_padding_px))
    hh = int(max(18.0, roi_scale * (k * sigma_v) + roi_padding_px))

    max_hw = int(max(18.0, 0.5 * roi_max_fraction_of_frame * float(frame_w)))
    max_hh = int(max(18.0, 0.5 * roi_max_fraction_of_frame * float(frame_h)))
    hw = min(hw, max_hw)
    hh = min(hh, max_hh)
    return hw, hh


def clamp_roi(u: float, v: float, hw: int, hh: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(w - 1, u - hw)))
    y1 = int(max(0, min(h - 1, v - hh)))
    x2 = int(max(0, min(w - 1, u + hw)))
    y2 = int(max(0, min(h - 1, v + hh)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def make_run_dir(base_dir: str | Path) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(base_dir) / f'run_{ts}'


def collect_fresh_tracks(store: Dict[str, Track], site: SiteRef, t_now: float, stale_s: float) -> list[TrackSummary]:
    out: list[TrackSummary] = []
    for icao24, tr in store.items():
        age_s = float(t_now - tr.last.t_rx)
        if age_s < 0 or age_s > stale_s:
            continue
        p_enu, _v_enu, _dt, _motion_source = tr.predict_enu(site, t_now)
        _, _, range_m = bearing_elevation_range(p_enu)
        out.append(
            TrackSummary(
                icao24=icao24,
                flight=(tr.last.flight or '').strip(),
                lat_deg=tr.last.lat_deg,
                lon_deg=tr.last.lon_deg,
                alt_m=tr.last.alt_m,
                age_s=age_s,
                range_m=range_m,
            )
        )
    out.sort(key=lambda r: (r.age_s, r.range_m))
    return out


def print_adsb_summary(rows: list[TrackSummary], max_items: int, stats: AdsbRuntimeStats, t_now: float):
    if stats.total_messages <= 0 or stats.last_message_t is None:
        console.print('[yellow]ADS-B summary[/yellow] waiting_for_messages=1 fresh_tracks=0 total_msgs=0')
        return

    last_age_s = max(0.0, float(t_now - stats.last_message_t))
    console.print(
        f"[cyan]ADS-B summary[/cyan] total_msgs={stats.total_messages} unique={len(stats.unique_icaos)} fresh_tracks={len(rows)} last_msg_age={last_age_s:.1f}s"
    )
    for row in rows[:max_items]:
        flight_txt = f" {row.flight}" if row.flight else ''
        console.print(
            f"  {row.icao24}{flight_txt} alt={row.alt_m:.0f}m range={row.range_m/1000.0:.1f}km age={row.age_s:.1f}s lat={row.lat_deg:.5f} lon={row.lon_deg:.5f}"
        )


def draw_adsb_overlay(frame: 'object', rows: list[TrackSummary], frames: int, max_items: int, stats: AdsbRuntimeStats):
    total_msgs = stats.total_messages
    cv2.putText(
        frame,
        f"frames={frames} adsb_tracks={len(rows)} adsb_msgs={total_msgs}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def _format_adsb_label_lines(
    tr: Track,
    state_age_s: float,
    az_deg: float,
    el_deg: float,
    horizontal_speed_mps: Optional[float] = None,
) -> list[str]:
    ident = tr.last.icao24
    flight = (tr.last.flight or '').strip()
    if flight:
        ident = f"{flight} {ident}"

    lines = [ident, f"alt={tr.last.alt_m:.0f}m age={state_age_s:.1f}s"]
    gs_mps = float(horizontal_speed_mps) if horizontal_speed_mps is not None else float(np.hypot(tr.last.vn_mps, tr.last.ve_mps))
    if np.isfinite(gs_mps):
        lines[1] += f" gs={gs_mps:.1f}m/s"
    if np.isfinite(az_deg) and np.isfinite(el_deg):
        lines.append(f"az={az_deg:.1f} el={el_deg:.1f}")
    return lines


def _draw_filled_label_box(
    frame: 'object',
    lines: list[str],
    anchor_xy: tuple[int, int],
    bg_color: tuple[int, int, int],
):
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    pad = 6
    line_gap = 4
    baseline_pad = 4

    sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    max_w = max(sz[0][0] for sz in sizes)
    total_h = sum(sz[0][1] for sz in sizes) + line_gap * (len(lines) - 1)
    box_w = max_w + pad * 2
    box_h = total_h + pad * 2 + baseline_pad

    h, w = frame.shape[:2]
    anchor_x, anchor_y = int(anchor_xy[0]), int(anchor_xy[1])
    x = min(max(anchor_x, 0), max(0, w - box_w))

    y_above = anchor_y - box_h - 8
    if y_above >= 0:
        y = y_above
    else:
        y = min(max(anchor_y + 8, 0), max(0, h - box_h))

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 0, 0), 1)

    text_y = y + pad
    for line, (size, baseline) in zip(lines, sizes):
        text_y += size[1]
        cv2.putText(frame, line, (x + pad, text_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        text_y += line_gap


def _pick_best_detection(det_records: list[dict]) -> Optional[dict]:
    if not det_records:
        return None
    return max(det_records, key=lambda det: (float(det.get('conf', 0.0)), -int(det.get('xyxy_frame', [0])[0])))


def _resolve_draw_detection(
    state_store: Dict[str, MatchedDrawState],
    *,
    icao24: str,
    frame_id: int,
    best_det: Optional[dict],
    can_draw_roi: bool,
    max_draw_persist_frames: int,
) -> Optional[MatchedDrawState]:
    if not can_draw_roi:
        state_store.pop(icao24, None)
        return None

    if best_det is not None:
        state = MatchedDrawState(
            frame_id=frame_id,
            xyxy_frame=tuple(int(v) for v in best_det['xyxy_frame']),
            cls=str(best_det['cls']),
            conf=float(best_det['conf']),
        )
        state_store[icao24] = state
        return state

    state = state_store.get(icao24)
    if state is None:
        return None
    if (frame_id - state.frame_id) <= max_draw_persist_frames:
        return state

    state_store.pop(icao24, None)
    return None


def _roi_intersects_frame(roi_xyxy: tuple[int, int, int, int], w: int, h: int) -> bool:
    x1, y1, x2, y2 = roi_xyxy
    if x2 <= x1 or y2 <= y1:
        return False
    return not (x2 < 0 or y2 < 0 or x1 >= w or y1 >= h)


def _compute_projection_debug(
    tr: Track,
    *,
    site: SiteRef,
    t_frame: float,
    cam_model: CameraModel,
    fx: float,
    fy: float,
    cfg_roi: dict,
    cfg_projection: dict,
    w: int,
    h: int,
) -> ProjectionDebugInfo:
    last_enu = tr.last_enu(site)
    p_enu, v_enu, predict_dt_s, motion_source = tr.predict_enu(site, t_frame)
    u, v, in_front, _p_cam = project_enu_to_pixel(cam_model, p_enu)
    az_rad, el_rad, rng = bearing_elevation_range(p_enu)

    az_deg = float(np.degrees(az_rad))
    el_deg = float(np.degrees(el_rad))
    age_s = float(t_frame - tr.last.t_rx)
    extrapolated_dt_s = max(0.0, predict_dt_s)
    latency_extrapolated = bool(extrapolated_dt_s > 1e-3)
    horizontal_speed_mps = float(np.hypot(v_enu[0], v_enu[1]))
    max_track_age_s_draw = float(cfg_projection.get('max_track_age_s_draw', 1.0))

    if not np.isfinite(u) or not np.isfinite(v):
        return ProjectionDebugInfo(
            icao24=tr.last.icao24,
            age_s=age_s,
            u=None,
            v=None,
            center_in_frame=False,
            roi_xyxy=None,
            roi_intersects_frame=False,
            roi_w_px=None,
            roi_h_px=None,
            az_deg=az_deg,
            el_deg=el_deg,
            latency_extrapolated=latency_extrapolated,
            extrapolated_dt_s=extrapolated_dt_s,
            horizontal_speed_mps=horizontal_speed_mps,
            motion_source=motion_source,
            last_enu_m=(float(last_enu[0]), float(last_enu[1]), float(last_enu[2])),
            predicted_enu_m=(float(p_enu[0]), float(p_enu[1]), float(p_enu[2])),
            can_draw_roi=False,
            skip_reason='nonfinite_projection',
        )

    center_in_frame = bool(0.0 <= u < float(w) and 0.0 <= v < float(h))
    if not in_front:
        return ProjectionDebugInfo(
            icao24=tr.last.icao24,
            age_s=age_s,
            u=float(u),
            v=float(v),
            center_in_frame=False,
            roi_xyxy=None,
            roi_intersects_frame=False,
            roi_w_px=None,
            roi_h_px=None,
            az_deg=az_deg,
            el_deg=el_deg,
            latency_extrapolated=latency_extrapolated,
            extrapolated_dt_s=extrapolated_dt_s,
            horizontal_speed_mps=horizontal_speed_mps,
            motion_source=motion_source,
            last_enu_m=(float(last_enu[0]), float(last_enu[1]), float(last_enu[2])),
            predicted_enu_m=(float(p_enu[0]), float(p_enu[1]), float(p_enu[2])),
            can_draw_roi=False,
            skip_reason='behind_camera',
        )

    hw, hh = roi_half_sizes_px(rng, fx, fy, cfg_roi, cfg_projection, w, h)
    raw_roi_xyxy = (int(u - hw), int(v - hh), int(u + hw), int(v + hh))
    roi_intersects = _roi_intersects_frame(raw_roi_xyxy, w, h)
    roi_xyxy = clamp_roi(u, v, hw, hh, w, h) if roi_intersects else None
    roi_w_px = None if roi_xyxy is None else int(roi_xyxy[2] - roi_xyxy[0])
    roi_h_px = None if roi_xyxy is None else int(roi_xyxy[3] - roi_xyxy[1])

    skip_reason = None
    can_draw_roi = False
    if age_s > max_track_age_s_draw:
        skip_reason = 'track_stale_draw'
    elif roi_xyxy is None:
        skip_reason = 'roi_outside_frame'
    elif roi_xyxy[2] <= roi_xyxy[0] or roi_xyxy[3] <= roi_xyxy[1]:
        skip_reason = 'invalid_roi'
    elif u < -200 or u > w + 200 or v < -200 or v > h + 200:
        skip_reason = 'center_outside_guard_band'
    else:
        can_draw_roi = True

    return ProjectionDebugInfo(
        icao24=tr.last.icao24,
        age_s=age_s,
        u=float(u),
        v=float(v),
        center_in_frame=center_in_frame,
        roi_xyxy=roi_xyxy,
        roi_intersects_frame=roi_intersects,
        roi_w_px=roi_w_px,
        roi_h_px=roi_h_px,
        az_deg=az_deg,
        el_deg=el_deg,
        latency_extrapolated=latency_extrapolated,
        extrapolated_dt_s=extrapolated_dt_s,
        horizontal_speed_mps=horizontal_speed_mps,
        motion_source=motion_source,
        last_enu_m=(float(last_enu[0]), float(last_enu[1]), float(last_enu[2])),
        predicted_enu_m=(float(p_enu[0]), float(p_enu[1]), float(p_enu[2])),
        can_draw_roi=can_draw_roi,
        skip_reason=skip_reason,
    )


def _draw_projection_debug_overlay(frame: 'object', debug_infos: list[ProjectionDebugInfo]):
    if not debug_infos:
        return

    freshest = debug_infos[0]
    uv_txt = 'n/a'
    if freshest.u is not None and freshest.v is not None:
        uv_txt = f"({freshest.u:.1f},{freshest.v:.1f})"

    roi_txt = 'n/a'
    if freshest.roi_xyxy is not None:
        x1, y1, x2, y2 = freshest.roi_xyxy
        roi_txt = f"({x1},{y1},{x2},{y2})"
    roi_wh_txt = 'n/a'
    if freshest.roi_w_px is not None and freshest.roi_h_px is not None:
        roi_wh_txt = f"({freshest.roi_w_px},{freshest.roi_h_px})"

    azel_txt = 'n/a'
    if freshest.az_deg is not None and freshest.el_deg is not None:
        azel_txt = f"{freshest.az_deg:.1f}/{freshest.el_deg:.1f}"

    debug_txt = (
        f"debug {freshest.icao24} age={freshest.age_s:.1f}s uv={uv_txt} "
        f"in_frame={int(freshest.center_in_frame)} roi={roi_txt} roi_hit={int(freshest.roi_intersects_frame)} "
        f"roi_wh={roi_wh_txt} extrap={int(freshest.latency_extrapolated)} dt={freshest.extrapolated_dt_s:.2f}s "
        f"hs={freshest.horizontal_speed_mps:.1f}m/s src={freshest.motion_source} az/el={azel_txt}"
    )
    if freshest.skip_reason:
        debug_txt += f" reason={freshest.skip_reason}"

    cv2.putText(
        frame,
        debug_txt,
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    y = 122
    for info in debug_infos:
        if info.u is not None and info.v is not None and not info.center_in_frame:
            note = f"OFFSCREEN TRACK {info.icao24} u={info.u:.1f} v={info.v:.1f}"
            if info.skip_reason:
                note += f" reason={info.skip_reason}"
            cv2.putText(
                frame,
                note,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            y += 18
        elif info.skip_reason:
            note = f"SKIPPED TRACK {info.icao24} reason={info.skip_reason}"
            cv2.putText(
                frame,
                note,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
            y += 18

    last_enu = freshest.last_enu_m
    pred_enu = freshest.predicted_enu_m
    enu_txt = (
        f"enu last=({last_enu[0]:.0f},{last_enu[1]:.0f},{last_enu[2]:.0f}) "
        f"pred=({pred_enu[0]:.0f},{pred_enu[1]:.0f},{pred_enu[2]:.0f})"
    )
    cv2.putText(
        frame,
        enu_txt,
        (20, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_orientation_debug_overlay(frame: 'object', orientation: OrientationDebugInfo):
    txt = (
        f"cam ypr=({orientation.yaw_deg:.1f},{orientation.pitch_deg:.1f},{orientation.roll_deg:.1f}) "
        f"fov_guess={int(orientation.use_fov_guess)}"
    )
    if orientation.use_fov_guess and orientation.hfov_deg is not None and orientation.vfov_deg is not None:
        txt += f" hfov/vfov=({orientation.hfov_deg:.1f},{orientation.vfov_deg:.1f})"
    cv2.putText(
        frame,
        txt,
        (20, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 255, 150),
        1,
        cv2.LINE_AA,
    )


async def adsb_task(store: Dict[str, Track], logger: Optional[RunLogger], cfg_adsb: dict, stats: AdsbRuntimeStats):
    ing = AdsbIngestor(
        AdsbIngestConfig(host=cfg_adsb.get('host', '127.0.0.1'), port=int(cfg_adsb.get('port', 30003)))
    )
    async for msg in ing.messages():
        tr = store.get(msg.icao24)
        if tr is None:
            store[msg.icao24] = Track(last=msg)
        else:
            tr.update(msg)
        stats.total_messages += 1
        stats.last_message_t = float(msg.t_rx)
        stats.unique_icaos.add(msg.icao24)
        if logger:
            logger.log_adsb(msg)


async def main_async(args):
    cfg = load_config(args.config, args.override)

    cam_mode = cfg['camera'].get('mode')
    if cam_mode:
        w, h, fps = mode_to_wh_fps(cam_mode)
    else:
        w = int(cfg['camera'].get('width', 1280))
        h = int(cfg['camera'].get('height', 720))
        fps = int(cfg['camera'].get('fps', 30))

    cam_cfg = CameraConfig(
        source=cfg['camera'].get('source', cfg['camera'].get('device', 0)),
        width=w,
        height=h,
        fps=fps,
        backend=cfg['camera'].get('backend', 'opencv'),
    )
    cam = CameraStream(cam_cfg)
    actual_w, actual_h = cam.actual_wh()
    if actual_w > 0 and actual_h > 0:
        w, h = actual_w, actual_h

    audio_enabled = bool(cfg.get('audio', {}).get('enabled', False))
    aud = None
    audio_det = None
    if audio_enabled:
        from sensors.audio_stream import AudioConfig, AudioStream
        from perception.audio_detector import AudioDetectorConfig, AudioDetector

        aud_cfg = AudioConfig(
            sample_rate_hz=int(cfg['audio'].get('sample_rate_hz', 16000)),
            blocksize=int(cfg['audio'].get('blocksize', 2048)),
            device=cfg['audio'].get('device', None),
        )
        aud = AudioStream(aud_cfg)
        aud.start()
        audio_det = AudioDetector(AudioDetectorConfig(sr=aud_cfg.sample_rate_hz))
        console.print('[cyan]Audio enabled[/cyan]')
    else:
        console.print('[yellow]Audio disabled[/yellow]')

    site = SiteRef(
        lat0_deg=float(cfg['site']['lat0_deg']),
        lon0_deg=float(cfg['site']['lon0_deg']),
        alt0_m=float(cfg['site']['alt0_m']),
    )

    cal = cfg['calibration']
    use_fov_guess = bool(cal.get('use_fov_guess', False))
    yaw_deg = float(cal.get('yaw_deg', 0.0))
    pitch_deg = float(cal.get('pitch_deg', 0.0))
    roll_deg = float(cal.get('roll_deg', 0.0))
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
    dist = np.array(cfg['calibration'].get('dist', [0, 0, 0, 0, 0]), dtype=np.float64)
    R = yaw_pitch_roll_to_R_enu_cam(
        yaw_deg,
        pitch_deg,
        roll_deg,
    )
    cam_model = CameraModel(K=K, dist=dist, R_enu_cam=R, t_enu_cam=np.zeros(3, dtype=np.float64))
    orientation_debug = OrientationDebugInfo(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        use_fov_guess=use_fov_guess,
        hfov_deg=float(cal.get('hfov_deg', 41.0)) if use_fov_guess else None,
        vfov_deg=float(cal.get('vfov_deg', 31.0)) if use_fov_guess else None,
    )

    yolo = YoloDetector(
        YoloConfig(
            backend=cfg.get('yolo', {}).get('backend', 'none'),
            model_path=cfg.get('yolo', {}).get('model_path'),
            imgsz=int(cfg.get('yolo', {}).get('imgsz', 640)),
            conf=float(cfg.get('yolo', {}).get('conf', 0.25)),
        )
    )
    vp = VisionPersistence(VisionPersistConfig(confirm_frames=int(cfg['fusion'].get('vision_persist_frames', 4))))

    fusion = FusionStateMachine(
        FusionConfig(
            max_verify_time_s=float(cfg['fusion'].get('max_verify_time_s', 60.0)),
            coincidence_window_s=float(cfg['fusion'].get('coincidence_window_s', 3.0)),
            require_audio_for_verify=bool(cfg['fusion'].get('require_audio_for_verify', False)),
        )
    )

    logging_cfg = cfg.get('logging', {})
    projection_cfg = cfg.get('projection', {})
    vision_draw_cfg = cfg.get('vision', {})
    logger = None
    run_dir = None
    if args.log_dir:
        run_dir = Path(args.log_dir)
    elif bool(logging_cfg.get('enabled', False)):
        run_dir = make_run_dir(logging_cfg.get('base_dir', 'runs'))

    projection_warnings = collect_projection_warnings(cfg)

    if run_dir is not None:
        logger = RunLogger(
            RunLoggerConfig(
                out_dir=run_dir,
                save_roi_jpeg=bool(logging_cfg.get('save_roi_jpeg', True)),
                jpeg_quality=int(logging_cfg.get('jpeg_quality', 90)),
                save_video=bool(logging_cfg.get('save_video', True)),
                video_fps=float(fps),
            )
        )
        logger.write_metadata(
            {
                'created_at': datetime.now().isoformat(),
                'config': cfg,
                'camera': {
                    'source_label': getattr(cam, 'source_label', cam_cfg.source),
                    'requested_width': cam_cfg.width,
                    'requested_height': cam_cfg.height,
                    'requested_fps': cam_cfg.fps,
                    'actual_width': w,
                    'actual_height': h,
                },
                'site': asdict(site),
                'projection_warnings': projection_warnings,
            }
        )
        console.print(f"[green]Logging to[/green] {run_dir}")

    track_store: Dict[str, Track] = {}
    adsb_enabled = bool(cfg['adsb'].get('enabled', False))
    adsb_stats = AdsbRuntimeStats()
    adsb_task_handle = None
    if adsb_enabled:
        adsb_task_handle = asyncio.create_task(adsb_task(track_store, logger, cfg['adsb'], adsb_stats))
        console.print(f"[cyan]ADS-B ingest enabled[/cyan] {cfg['adsb'].get('host')}:{cfg['adsb'].get('port')}")
    else:
        console.print('[yellow]ADS-B ingest disabled[/yellow]')

    audio_scores: list[tuple[float, float]] = []
    display = bool(cfg['camera'].get('display', False) or args.display)
    summary_interval_s = float(logging_cfg.get('summary_interval_s', 5.0))
    overlay_max_items = int(logging_cfg.get('overlay_max_items', 4))
    stale_track_s = float(logging_cfg.get('stale_track_s', 15.0))
    record_only_when_tracks_present = bool(logging_cfg.get('record_only_when_tracks_present', True))
    record_stop_holdoff_s = float(logging_cfg.get('record_stop_holdoff_s', 5.0))
    max_draw_persist_frames = int(vision_draw_cfg.get('max_draw_persist_frames', 3))

    console.print(
        f"[cyan]Camera[/cyan] backend={cam_cfg.backend} source={getattr(cam, 'source_label', cam_cfg.source)} requested={cam_cfg.width}x{cam_cfg.height}@{cam_cfg.fps} actual={w}x{h}"
    )
    console.print(
        f"[cyan]Fusion[/cyan] require_audio_for_verify={fusion.cfg.require_audio_for_verify} yolo_backend={yolo.backend}"
    )
    console.print(
        f"[cyan]Projection[/cyan] yaw_deg={yaw_deg:.1f} pitch_deg={pitch_deg:.1f} roll_deg={roll_deg:.1f} "
        f"camera_axes=(+X right,+Y down,+Z forward) base_forward=+N yaw+=>toward+E pitch+=>tilt_up roll+=>clockwise_view"
    )
    if use_fov_guess:
        console.print(
            f"[cyan]Projection[/cyan] use_fov_guess=1 hfov_deg={orientation_debug.hfov_deg:.1f} vfov_deg={orientation_debug.vfov_deg:.1f}"
        )
    for warning in projection_warnings:
        console.print(f"[yellow]Config warning[/yellow] {warning}")

    t0 = time.time()
    frames = 0
    last_summary_t = 0.0
    last_tracks_present_t: Optional[float] = None
    final_decisions: Dict[str, TrackDecision] = {}
    matched_draw_states: Dict[str, MatchedDrawState] = {}

    try:
        while True:
            ok, frame, t_frame = cam.read()
            if not ok:
                console.print('[red]Camera read failed or stream ended[/red]')
                break

            frames += 1
            frame_id = int(frames)
            fresh_rows = collect_fresh_tracks(track_store, site, t_frame, stale_track_s) if adsb_enabled else []
            if fresh_rows:
                last_tracks_present_t = float(t_frame)
            fresh_debug_infos: list[ProjectionDebugInfo] = []
            projection_info_by_icao: Dict[str, ProjectionDebugInfo] = {}
            if adsb_enabled:
                for row in fresh_rows:
                    tr = track_store.get(row.icao24)
                    if tr is None:
                        continue
                    info = _compute_projection_debug(
                        tr,
                        site=site,
                        t_frame=t_frame,
                        cam_model=cam_model,
                        fx=float(K[0, 0]),
                        fy=float(K[1, 1]),
                        cfg_roi=cfg.get('roi', {}),
                        cfg_projection=projection_cfg,
                        w=w,
                        h=h,
                    )
                    fresh_debug_infos.append(info)
                    projection_info_by_icao[row.icao24] = info

            record_video_active = bool(logging_cfg.get('save_video', True))
            if logger is None:
                record_video_active = False
            elif record_only_when_tracks_present:
                record_video_active = bool(
                    fresh_rows
                    or (
                        last_tracks_present_t is not None
                        and float(t_frame - last_tracks_present_t) <= record_stop_holdoff_s
                    )
                )

            video_frame_idx = int(logger.video_frame_idx) if logger is not None and record_video_active else -1
            overlay_frame = frame.copy()

            if aud is not None and audio_det is not None:
                while True:
                    rec = aud.read_block(timeout_s=0.0)
                    if rec is None:
                        break
                    t_a, x = rec
                    s = audio_det.score(x)
                    audio_scores.append((t_a, s))
                    cutoff = t_a - 60.0
                    while audio_scores and audio_scores[0][0] < cutoff:
                        audio_scores.pop(0)
                    if logger:
                        logger.log_audio_score(t_center=t_a, score=s)

            for row in fresh_rows:
                icao = row.icao24
                tr = track_store.get(icao)
                info = projection_info_by_icao.get(icao)
                if tr is None or info is None:
                    continue

                state_age_s = float(info.age_s)
                az_deg = float(info.az_deg) if info.az_deg is not None else 0.0
                el_deg = float(info.el_deg) if info.el_deg is not None else 0.0
                rng = float(row.range_m)
                u = float(info.u) if info.u is not None else 0.0
                v = float(info.v) if info.v is not None else 0.0

                roi_xyxy = info.roi_xyxy if info.roi_xyxy is not None else (0, 0, 0, 0)
                x1, y1, x2, y2 = roi_xyxy
                roi = None
                roi_path = None
                dets = []
                if info.can_draw_roi and info.roi_xyxy is not None:
                    roi = frame[y1:y2, x1:x2].copy()
                    dets = yolo.infer_bgr(roi)
                    if logger:
                        roi_path = logger.log_roi_crop(
                            icao24=icao,
                            t_frame=t_frame,
                            frame_id=frame_id,
                            roi_xyxy=roi_xyxy,
                            roi_bgr=roi,
                        )

                detected_this_frame = any(d.conf >= yolo.cfg.conf for d in dets)
                vp.update(icao, t_frame, detected_this_frame)

                a_thresh = float(cfg['fusion'].get('audio_energy_thresh', 0.25))
                audio_ok = False
                if aud is not None:
                    audio_ok = any(s >= a_thresh and (t_frame - 2.0) <= ta <= t_frame for (ta, s) in audio_scores)

                ev = Evidence(vision_ok=vp.confirmed(icao), audio_ok=audio_ok)
                in_corridor = True
                decision = fusion.update(icao, t_frame, in_corridor=in_corridor, ev=ev)

                if decision in (TrackDecision.VERIFIED, TrackDecision.UNVERIFIED) and icao not in final_decisions:
                    final_decisions[icao] = decision
                    rec = {
                        't': float(t_frame),
                        'frame_id': frame_id,
                        'video_frame_idx': int(video_frame_idx),
                        'icao24': icao,
                        'decision': decision.name,
                        'vision_ok': ev.vision_ok,
                        'audio_ok': ev.audio_ok,
                        'reason': fusion.tracks[icao].last_reason,
                        't_entry': fusion.tracks[icao].t_entry,
                        't_verified': fusion.tracks[icao].t_verified,
                    }
                    if logger:
                        logger.log_decision(rec)
                    if decision == TrackDecision.VERIFIED:
                        console.print(f"[green]VERIFIED[/green] {icao} rng={rng:.0f}m")
                    else:
                        console.print(f"[yellow]UNVERIFIED[/yellow] {icao} reason={fusion.tracks[icao].last_reason}")

                det_records = []
                for det in dets:
                    if det.conf < yolo.cfg.conf:
                        continue
                    dx1, dy1, dx2, dy2 = [int(vv) for vv in det.xyxy]
                    det_records.append(
                        {
                            'cls': det.cls,
                            'conf': float(det.conf),
                            'xyxy_roi': [dx1, dy1, dx2, dy2],
                            'xyxy_frame': [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2],
                        }
                    )
                best_det = _pick_best_detection(det_records)
                draw_det = _resolve_draw_detection(
                    matched_draw_states,
                    icao24=icao,
                    frame_id=frame_id,
                    best_det=best_det,
                    can_draw_roi=bool(info.can_draw_roi),
                    max_draw_persist_frames=max_draw_persist_frames,
                )

                if logger:
                    logger.log_track_frame(
                        {
                            'frame_id': frame_id,
                            'video_frame_idx': int(video_frame_idx),
                            't_frame': float(t_frame),
                            't_rel_s': float(t_frame - logger.first_frame_t) if logger and logger.first_frame_t is not None else 0.0,
                            'saved_in_video': bool(record_video_active),
                            'icao24': icao,
                            'flight': (tr.last.flight or '').strip(),
                            'source_t_rx': float(tr.last.t_rx),
                            'state_age_s': float(t_frame - tr.last.t_rx),
                            'lat_deg': float(tr.last.lat_deg),
                            'lon_deg': float(tr.last.lon_deg),
                            'alt_m': float(tr.last.alt_m),
                            'range_m': float(rng),
                            'az_deg': az_deg,
                            'el_deg': el_deg,
                            'pixel_center': [float(u), float(v)],
                            'roi_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                            'roi_path': roi_path,
                            'projection_latency_extrapolated': bool(info.latency_extrapolated),
                            'projection_extrapolated_dt_s': float(info.extrapolated_dt_s),
                            'projection_horizontal_speed_mps': float(info.horizontal_speed_mps),
                            'projection_motion_source': info.motion_source,
                            'projection_last_enu_m': [float(vv) for vv in info.last_enu_m],
                            'projection_predicted_enu_m': [float(vv) for vv in info.predicted_enu_m],
                            'projection_skip_reason': info.skip_reason,
                            'projection_roi_w_px': info.roi_w_px,
                            'projection_roi_h_px': info.roi_h_px,
                            'yolo_backend': yolo.backend,
                            'detected_this_frame': bool(detected_this_frame),
                            'vision_confirmed': bool(ev.vision_ok),
                            'audio_ok': bool(ev.audio_ok),
                            'decision': decision.name,
                            'reason': fusion.tracks[icao].last_reason,
                            'detection_count': len(det_records),
                            'detections': det_records,
                            'matched_detection': best_det,
                        }
                    )

                if info.can_draw_roi and info.roi_xyxy is not None:
                    cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.circle(overlay_frame, (int(u), int(v)), 4, (0, 255, 0), -1)
                    label_lines = _format_adsb_label_lines(
                        tr,
                        state_age_s=state_age_s,
                        az_deg=az_deg,
                        el_deg=el_deg,
                        horizontal_speed_mps=info.horizontal_speed_mps,
                    )
                    label_anchor = (x1, y1)
                    label_bg = (0, 255, 255)
                    if draw_det is not None:
                        dx1, dy1, dx2, dy2 = draw_det.xyxy_frame
                        cv2.rectangle(overlay_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        label_anchor = (dx1, dy1)
                        label_bg = (0, 255, 0)
                    _draw_filled_label_box(overlay_frame, label_lines, label_anchor, label_bg)

            if adsb_enabled and summary_interval_s > 0 and (t_frame - last_summary_t) >= summary_interval_s:
                print_adsb_summary(fresh_rows, overlay_max_items, adsb_stats, t_frame)
                last_summary_t = t_frame

            draw_adsb_overlay(overlay_frame, fresh_rows, frames, overlay_max_items, adsb_stats)
            _draw_projection_debug_overlay(overlay_frame, fresh_debug_infos)
            _draw_orientation_debug_overlay(overlay_frame, orientation_debug)
            if logger:
                logger.log_frame(
                    overlay_frame,
                    t_frame=t_frame,
                    frame_id=frame_id,
                    save_in_video=record_video_active,
                )

            if display:
                cv2.imshow('realtime', overlay_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Yield to the asyncio event loop so the background ADS-B ingest task can run.
            await asyncio.sleep(0)

            if args.seconds and (time.time() - t0) > args.seconds:
                break

    finally:
        cam.release()
        if aud is not None:
            aud.stop()
        if display:
            cv2.destroyAllWindows()
        if adsb_task_handle:
            adsb_task_handle.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await adsb_task_handle
        if logger:
            logger.close(
                {
                    'runtime_seconds': float(time.time() - t0),
                    'adsb_unique_icaos_seen': len(adsb_stats.unique_icaos),
                    'adsb_total_messages_seen': int(adsb_stats.total_messages),
                    'projection_warning_count': len(projection_warnings),
                }
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--override', default=None)
    ap.add_argument('--display', action='store_true')
    ap.add_argument('--seconds', type=float, default=0.0, help='0 = run forever')
    ap.add_argument('--log-dir', default=None, help='e.g., runs/2026-03-10_test1')
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
