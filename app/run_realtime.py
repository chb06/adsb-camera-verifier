from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
import time
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

    def predict_enu(self, site: SiteRef, t: float) -> np.ndarray:
        p0 = geodetic_to_enu(site, self.last.lat_deg, self.last.lon_deg, self.last.alt_m)
        dt = float(t - self.last.t_rx)
        v_enu = np.array([self.last.ve_mps, self.last.vn_mps, self.last.vu_mps], dtype=np.float64)
        return p0 + v_enu * dt


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


def roi_half_sizes_px(range_m: float, fx: float, fy: float, cfg_roi: dict) -> Tuple[int, int]:
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


def reshape_roi(x1: int, y1: int, x2: int, y2: int, w: int, h: int, cfg_roi: dict) -> Tuple[int, int, int, int]:
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


def make_run_dir(base_dir: str | Path) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(base_dir) / f'run_{ts}'


def collect_fresh_tracks(store: Dict[str, Track], site: SiteRef, t_now: float, stale_s: float) -> list[TrackSummary]:
    out: list[TrackSummary] = []
    for icao24, tr in store.items():
        age_s = float(t_now - tr.last.t_rx)
        if age_s < 0 or age_s > stale_s:
            continue
        p_enu = tr.predict_enu(site, t_now)
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
    y = 75
    for row in rows[:max_items]:
        txt = f"{row.icao24}"
        if row.flight:
            txt += f" {row.flight}"
        txt += f" {row.alt_m:.0f}m {row.range_m/1000.0:.1f}km"
        cv2.putText(frame, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += 24


def detection_label(det: dict, icao24: str, flight: str, alt_m: float, range_m: float) -> str:
    parts = [icao24]
    if flight:
        parts.append(flight)
    parts.append(f"alt={alt_m:.0f}m")
    parts.append(f"rng={range_m/1000.0:.1f}km")
    parts.append(str(det['cls']))
    parts.append(f"{float(det['conf']):.2f}")
    return " ".join(p for p in parts if p)


async def adsb_task(store: Dict[str, Track], logger: Optional[RunLogger], cfg_adsb: dict, stats: AdsbRuntimeStats):
    ing = AdsbIngestor(
        AdsbIngestConfig(host=cfg_adsb.get('host', '127.0.0.1'), port=int(cfg_adsb.get('port', 30003)))
    )
    async for msg in ing.messages():
        store[msg.icao24] = Track(last=msg)
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
        float(cfg['calibration'].get('yaw_deg', 0.0)),
        float(cfg['calibration'].get('pitch_deg', 0.0)),
        float(cfg['calibration'].get('roll_deg', 0.0)),
    )
    cam_model = CameraModel(K=K, dist=dist, R_enu_cam=R, t_enu_cam=np.zeros(3, dtype=np.float64))

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

    console.print(
        f"[cyan]Camera[/cyan] backend={cam_cfg.backend} source={getattr(cam, 'source_label', cam_cfg.source)} requested={cam_cfg.width}x{cam_cfg.height}@{cam_cfg.fps} actual={w}x{h}"
    )
    console.print(
        f"[cyan]Fusion[/cyan] require_audio_for_verify={fusion.cfg.require_audio_for_verify} yolo_backend={yolo.backend}"
    )
    for warning in projection_warnings:
        console.print(f"[yellow]Config warning[/yellow] {warning}")

    t0 = time.time()
    frames = 0
    last_summary_t = 0.0
    final_decisions: Dict[str, TrackDecision] = {}

    try:
        while True:
            ok, frame, t_frame = cam.read()
            if not ok:
                detail = cam.last_error() if hasattr(cam, 'last_error') else ''
                if detail:
                    console.print(f'[red]Camera read failed or stream ended[/red] {detail}')
                else:
                    console.print('[red]Camera read failed or stream ended[/red]')
                break

            frames += 1
            frame_id = int(frames)

            raw_frame_for_log = frame.copy() if logger is not None and display else frame
            video_frame_idx = logger.log_frame(raw_frame_for_log, t_frame=t_frame, frame_id=frame_id) if logger else (frame_id - 1)

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

            for icao, tr in list(track_store.items()):
                lead_s = float(cfg['adsb'].get('lead_s', 0.0))
                p_enu = tr.predict_enu(site, t_frame + lead_s)
                u, v, in_front, _p_cam = project_enu_to_pixel(cam_model, p_enu)
                if not in_front:
                    continue
                if u < -200 or u > w + 200 or v < -200 or v > h + 200:
                    continue

                az_rad, el_rad, rng = bearing_elevation_range(p_enu)
                cfg_roi = cfg.get('roi', {})
                hw, hh = roi_half_sizes_px(rng, float(K[0, 0]), float(K[1, 1]), cfg_roi)
                x1, y1, x2, y2 = clamp_roi(u, v, hw, hh, w, h)
                x1, y1, x2, y2 = reshape_roi(x1, y1, x2, y2, w, h, cfg_roi)
                roi = frame[y1:y2, x1:x2].copy()

                dets = yolo.infer_bgr(roi)
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

                roi_path = None
                if logger:
                    roi_path = logger.log_roi_crop(
                        icao24=icao,
                        t_frame=t_frame,
                        frame_id=frame_id,
                        roi_xyxy=(x1, y1, x2, y2),
                        roi_bgr=roi,
                    )

                det_records = []
                for det in dets:
                    dx1, dy1, dx2, dy2 = [int(vv) for vv in det.xyxy]
                    det_records.append(
                        {
                            'cls': det.cls,
                            'conf': float(det.conf),
                            'xyxy_roi': [dx1, dy1, dx2, dy2],
                            'xyxy_frame': [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2],
                        }
                    )

                if logger:
                    logger.log_track_frame(
                        {
                            'frame_id': frame_id,
                            'video_frame_idx': int(video_frame_idx),
                            't_frame': float(t_frame),
                            't_rel_s': float(t_frame - logger.first_frame_t) if logger and logger.first_frame_t is not None else 0.0,
                            'icao24': icao,
                            'flight': (tr.last.flight or '').strip(),
                            'source_t_rx': float(tr.last.t_rx),
                            'state_age_s': float(t_frame - tr.last.t_rx),
                            'lat_deg': float(tr.last.lat_deg),
                            'lon_deg': float(tr.last.lon_deg),
                            'alt_m': float(tr.last.alt_m),
                            'range_m': float(rng),
                            'az_deg': float(np.degrees(az_rad)),
                            'el_deg': float(np.degrees(el_rad)),
                            'pixel_center': [float(u), float(v)],
                            'roi_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                            'roi_path': roi_path,
                            'yolo_backend': yolo.backend,
                            'detected_this_frame': bool(detected_this_frame),
                            'vision_confirmed': bool(ev.vision_ok),
                            'audio_ok': bool(ev.audio_ok),
                            'decision': decision.name,
                            'reason': fusion.tracks[icao].last_reason,
                            'detection_count': len(det_records),
                            'detections': det_records,
                        }
                    )

                if display:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.circle(frame, (int(u), int(v)), 4, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        icao,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    for det in det_records:
                        dx1, dy1, dx2, dy2 = det['xyxy_frame']
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            detection_label(det, icao, (tr.last.flight or '').strip(), float(tr.last.alt_m), float(rng)),
                            (dx1, max(0, dy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

            fresh_rows = collect_fresh_tracks(track_store, site, t_frame, stale_track_s) if adsb_enabled else []

            if adsb_enabled and summary_interval_s > 0 and (t_frame - last_summary_t) >= summary_interval_s:
                print_adsb_summary(fresh_rows, overlay_max_items, adsb_stats, t_frame)
                last_summary_t = t_frame

            if display:
                draw_adsb_overlay(frame, fresh_rows, frames, overlay_max_items, adsb_stats)
                cv2.imshow('realtime', frame)
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
