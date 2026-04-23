from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.run_realtime import main_async
from app.run_offline_detection import run_offline_detection
from data.replay import ReplayRun


def make_video(path: Path, frames: int = 60, width: int = 1280, height: int = 720, fps: int = 15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (100 + (i * 7) % (width - 200), height // 2), 30, (255, 255, 255), -1)
        cv2.putText(frame, f'frame {i}', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        writer.write(frame)
    writer.release()


def sbs_line(
    icao24: str = 'abc123',
    flight: str = 'TEST123',
    alt_ft: float = 1000.0,
    gs_kts: float = 120.0,
    track_deg: float = 0.0,
    lat: float = 37.0090,
    lon: float = -122.0,
    vr_fpm: float = 0.0,
) -> str:
    parts = [''] * 22
    parts[0] = 'MSG'
    parts[1] = '3'
    parts[2] = '111'
    parts[3] = '11111'
    parts[4] = icao24
    parts[5] = '111111'
    parts[6] = '2026/04/15'
    parts[7] = '00:00:00.000'
    parts[8] = '2026/04/15'
    parts[9] = '00:00:00.000'
    parts[10] = flight
    parts[11] = f'{alt_ft}'
    parts[12] = f'{gs_kts}'
    parts[13] = f'{track_deg}'
    parts[14] = f'{lat}'
    parts[15] = f'{lon}'
    parts[16] = f'{vr_fpm}'
    parts[17] = '0'
    parts[18] = '0'
    parts[19] = '0'
    parts[20] = '0'
    parts[21] = '0'
    return ','.join(parts) + '\n'


async def start_adsb_server(lines: list[str]):
    async def handle(reader, writer):
        for line in lines:
            writer.write(line.encode())
            await writer.drain()
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_realtime_logging_and_replay_roundtrip(tmp_path: Path):
    async def scenario():
        video_path = tmp_path / 'input.mp4'
        make_video(video_path, frames=80)

        server, port = await start_adsb_server([sbs_line() for _ in range(200)])
        run_dir = tmp_path / 'run_rt'
        cfg_path = tmp_path / 'config.yaml'
        cfg = {
            'camera': {
                'backend': 'opencv',
                'source': str(video_path),
                'mode': '720p15',
                'display': False,
            },
            'audio': {'enabled': False},
            'adsb': {'enabled': True, 'host': '127.0.0.1', 'port': port},
            'site': {'lat0_deg': 37.0, 'lon0_deg': -122.0, 'alt0_m': 0.0},
            'calibration': {
                'use_fov_guess': True,
                'hfov_deg': 60.0,
                'vfov_deg': 35.0,
                'fx': 0.0,
                'fy': 0.0,
                'cx': 0.0,
                'cy': 0.0,
                'dist': [0, 0, 0, 0, 0],
                'yaw_deg': 0.0,
                'pitch_deg': 0.0,
                'roll_deg': 0.0,
            },
            'corridor': {'enabled': False},
            'roi': {
                'k_sigma': 3.0,
                'default_sigma_m': 80.0,
                'min_hw_px': 120,
                'min_hh_px': 120,
                'obj_margin_px': 24,
            },
            'fusion': {
                'max_verify_time_s': 5.0,
                'coincidence_window_s': 3.0,
                'vision_persist_frames': 2,
                'audio_energy_thresh': 0.25,
                'require_audio_for_verify': False,
            },
            'yolo': {'backend': 'none', 'model_path': None, 'imgsz': 640, 'conf': 0.25},
            'logging': {
                'enabled': True,
                'base_dir': str(tmp_path),
                'save_video': True,
                'save_roi_jpeg': True,
                'jpeg_quality': 90,
                'summary_interval_s': 0.2,
                'overlay_max_items': 4,
                'stale_track_s': 15.0,
            },
        }
        cfg_path.write_text(yaml.safe_dump(cfg))

        args = argparse.Namespace(config=str(cfg_path), override=None, display=False, seconds=0.0, log_dir=str(run_dir))
        try:
            await main_async(args)
        finally:
            server.close()
            await server.wait_closed()

        return run_dir

    run_dir = asyncio.run(scenario())
    assert (run_dir / 'video.mp4').exists()
    assert (run_dir / 'frame_index.jsonl').exists()
    assert (run_dir / 'track_frames.jsonl').exists()
    assert (run_dir / 'adsb.jsonl').exists()
    assert (run_dir / 'run_summary.json').exists()

    frames = load_jsonl(run_dir / 'frame_index.jsonl')
    track_frames = load_jsonl(run_dir / 'track_frames.jsonl')
    adsb = load_jsonl(run_dir / 'adsb.jsonl')
    roi_rows = load_jsonl(run_dir / 'roi_index.jsonl')
    summary = json.loads((run_dir / 'run_summary.json').read_text())

    assert len(frames) > 0
    assert len(track_frames) > 0
    assert len(adsb) > 0
    assert len(roi_rows) > 0
    assert summary['frames_logged'] == len(frames)
    assert summary['track_frames_logged'] == len(track_frames)
    assert summary['adsb_messages_logged'] == len(adsb)

    rr = ReplayRun(run_dir)
    summ = rr.summary_dict()
    assert summ['frames'] == len(frames)
    assert summ['adsb_messages'] == len(adsb)
    assert summ['track_frames'] == len(track_frames)
    assert any(rr.track_records_for_frame(fr['frame_id']) for fr in frames)
    assert rr.latest_adsb_states(frames[-1]['t_frame'], stale_s=15.0)


class FakeYoloDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.backend

    def infer_bgr(self, img_bgr):
        from common_types import Detection

        h, w = img_bgr.shape[:2]
        return [Detection(cls='aircraft', conf=0.91, xyxy=(1, 2, min(w - 1, 20), min(h - 1, 18)))]


def test_offline_detection_outputs_roi_records(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / 'run_offline'
    run_dir.mkdir()
    (run_dir / 'detections').mkdir()

    video_path = run_dir / 'video.mp4'
    make_video(video_path, frames=3, width=320, height=240, fps=15)

    frame_rows = [
        {'frame_id': 1, 'video_frame_idx': 0, 't_frame': 10.0, 't_rel_s': 0.0, 'saved_in_video': True},
        {'frame_id': 2, 'video_frame_idx': 1, 't_frame': 10.1, 't_rel_s': 0.1, 'saved_in_video': True},
        {'frame_id': 3, 'video_frame_idx': 2, 't_frame': 10.2, 't_rel_s': 0.2, 'saved_in_video': True},
    ]
    (run_dir / 'frame_index.jsonl').write_text('\n'.join(json.dumps(r) for r in frame_rows) + '\n')
    track_rows = [
        {
            'frame_id': 1,
            'video_frame_idx': 0,
            't_frame': 10.0,
            't_rel_s': 0.0,
            'icao24': 'abc123',
            'flight': 'TEST123',
            'roi_xyxy': [10, 20, 120, 140],
            'range_m': 1000.0,
            'decision': 'IN_PROGRESS',
            'detection_count': 0,
        },
        {
            'frame_id': 2,
            'video_frame_idx': 1,
            't_frame': 10.1,
            't_rel_s': 0.1,
            'icao24': 'abc123',
            'flight': 'TEST123',
            'roi_xyxy': [10, 20, 120, 140],
            'range_m': 1000.0,
            'decision': 'IN_PROGRESS',
            'detection_count': 0,
        },
    ]
    (run_dir / 'track_frames.jsonl').write_text('\n'.join(json.dumps(r) for r in track_rows) + '\n')
    (run_dir / 'metadata.json').write_text(json.dumps({'camera': {'requested_fps': 15}}, indent=2))
    (run_dir / 'run_summary.json').write_text(json.dumps({'duration_s': 0.2}, indent=2))

    monkeypatch.setattr('app.run_offline_detection.YoloDetector', FakeYoloDetector)

    args = argparse.Namespace(
        run_dir=str(run_dir),
        mode='roi',
        backend='ultralytics',
        model_path='fake.pt',
        imgsz=640,
        conf=0.25,
        stride=1,
        limit_frames=0,
        allow_class=['aircraft'],
        save_preview_video=False,
        out=None,
    )
    run_offline_detection(args)

    det_files = sorted((run_dir / 'detections').glob('*.jsonl'))
    assert det_files, 'expected offline detection output jsonl'
    rows = load_jsonl(det_files[0])
    assert len(rows) == 2
    assert rows[0]['frame_id'] == 1
    assert rows[0]['icao24'] == 'abc123'
    assert rows[0]['cls'] == 'aircraft'
    assert rows[0]['conf'] > 0.9
