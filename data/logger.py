from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

from common_types import AdsbState


@dataclass
class RunLoggerConfig:
    out_dir: Path
    save_roi_jpeg: bool = True
    jpeg_quality: int = 90
    save_video: bool = True
    video_fps: float = 15.0


class RunLogger:
    """Structured logger for synchronized sessions.

    Directory layout:
      runs/<run_id>/
        metadata.json
        run_summary.json
        video.mp4
        frame_index.jsonl
        adsb.jsonl
        track_frames.jsonl
        roi_index.jsonl
        audio_scores.jsonl
        decisions.jsonl
        detections/
        roi/icao24/<frame_id>_<frame_ts_ms>.jpg
    """

    def __init__(self, cfg: RunLoggerConfig):
        self.cfg = cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (cfg.out_dir / 'roi').mkdir(exist_ok=True)
        (cfg.out_dir / 'detections').mkdir(exist_ok=True)
        (cfg.out_dir / 'exports').mkdir(exist_ok=True)

        self.f_adsb = (cfg.out_dir / 'adsb.jsonl').open('a', buffering=1)
        self.f_frames = (cfg.out_dir / 'frame_index.jsonl').open('a', buffering=1)
        self.f_track_frames = (cfg.out_dir / 'track_frames.jsonl').open('a', buffering=1)
        self.f_roi = (cfg.out_dir / 'roi_index.jsonl').open('a', buffering=1)
        self.f_audio = (cfg.out_dir / 'audio_scores.jsonl').open('a', buffering=1)
        self.f_dec = (cfg.out_dir / 'decisions.jsonl').open('a', buffering=1)
        self.video_writer = None
        self.video_path = cfg.out_dir / 'video.mp4'

        self.video_frame_idx = 0
        self.frames_logged = 0
        self.adsb_logged = 0
        self.track_frames_logged = 0
        self.roi_logged = 0
        self.audio_logged = 0
        self.decisions_logged = 0
        self.first_frame_t: Optional[float] = None
        self.last_frame_t: Optional[float] = None
        self._closed = False

    def close(self, final_summary: Optional[Dict[str, Any]] = None):
        if self._closed:
            return
        try:
            self.finalize(final_summary)
        finally:
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None
            for f in [self.f_adsb, self.f_frames, self.f_track_frames, self.f_roi, self.f_audio, self.f_dec]:
                try:
                    f.close()
                except Exception:
                    pass
            self._closed = True

    def write_metadata(self, meta: Dict[str, Any]):
        (self.cfg.out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))

    def _ensure_video_writer(self, frame_bgr: 'object'):
        if not self.cfg.save_video or self.video_writer is not None:
            return
        h, w = frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            float(self.cfg.video_fps),
            (int(w), int(h)),
        )

    def log_frame(self, frame_bgr: 'object', *, t_frame: float, frame_id: int) -> int:
        if self.first_frame_t is None:
            self.first_frame_t = float(t_frame)
        self.last_frame_t = float(t_frame)

        video_frame_idx = int(self.video_frame_idx)
        rec = {
            'frame_id': int(frame_id),
            'video_frame_idx': video_frame_idx,
            't_frame': float(t_frame),
            't_rel_s': float(t_frame - self.first_frame_t),
            'saved_in_video': bool(self.cfg.save_video),
        }
        self.f_frames.write(json.dumps(rec) + '\n')

        if self.cfg.save_video:
            self._ensure_video_writer(frame_bgr)
            if self.video_writer is not None:
                self.video_writer.write(frame_bgr)

        self.video_frame_idx += 1
        self.frames_logged += 1
        return video_frame_idx

    def log_adsb(self, msg: AdsbState):
        d = asdict(msg)
        self.f_adsb.write(json.dumps(d) + '\n')
        self.adsb_logged += 1

    def log_track_frame(self, rec: Dict[str, Any]):
        self.f_track_frames.write(json.dumps(rec) + '\n')
        self.track_frames_logged += 1

    def log_roi_crop(
        self,
        icao24: str,
        t_frame: float,
        frame_id: int,
        roi_xyxy: Tuple[int, int, int, int],
        roi_bgr: 'object',
    ) -> Optional[str]:
        x1, y1, x2, y2 = [int(v) for v in roi_xyxy]
        rel_dir = Path('roi') / icao24
        abs_dir = self.cfg.out_dir / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        ts_ms = int(float(t_frame) * 1000)
        rel_path = rel_dir / f"{int(frame_id):06d}_{ts_ms}.jpg"
        abs_path = self.cfg.out_dir / rel_path

        path_str: Optional[str] = None
        if self.cfg.save_roi_jpeg:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)]
            ok = cv2.imwrite(str(abs_path), roi_bgr, encode_params)
            if ok:
                path_str = str(rel_path)

        rec = {
            'icao24': icao24,
            'frame_id': int(frame_id),
            't_frame': float(t_frame),
            'roi_xyxy': [x1, y1, x2, y2],
            'path': path_str,
        }
        self.f_roi.write(json.dumps(rec) + '\n')
        self.roi_logged += 1
        return path_str

    def log_audio_score(self, t_center: float, score: float):
        self.f_audio.write(json.dumps({'t': float(t_center), 'score': float(score)}) + '\n')
        self.audio_logged += 1

    def log_decision(self, rec: Dict[str, Any]):
        self.f_dec.write(json.dumps(rec) + '\n')
        self.decisions_logged += 1

    def finalize(self, extra_summary: Optional[Dict[str, Any]] = None):
        summary = {
            'video_path': 'video.mp4' if self.cfg.save_video else None,
            'frames_logged': int(self.frames_logged),
            'adsb_messages_logged': int(self.adsb_logged),
            'track_frames_logged': int(self.track_frames_logged),
            'roi_records_logged': int(self.roi_logged),
            'audio_scores_logged': int(self.audio_logged),
            'decisions_logged': int(self.decisions_logged),
            'first_frame_t': self.first_frame_t,
            'last_frame_t': self.last_frame_t,
            'duration_s': (
                float(self.last_frame_t - self.first_frame_t)
                if self.first_frame_t is not None and self.last_frame_t is not None
                else 0.0
            ),
        }
        if extra_summary:
            summary.update(extra_summary)
        (self.cfg.out_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))
