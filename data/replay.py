from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open('r') as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@dataclass
class FrameRecord:
    frame_id: int
    video_frame_idx: int
    t_frame: float
    t_rel_s: float


@dataclass
class RoiRecord:
    icao24: str
    frame_id: int
    t_frame: float
    roi_xyxy: Tuple[int, int, int, int]
    path: Optional[Path]


@dataclass
class TrackFrameRecord:
    frame_id: int
    video_frame_idx: int
    t_frame: float
    icao24: str
    roi_xyxy: Tuple[int, int, int, int]
    raw: dict


@dataclass
class DetectionRecord:
    frame_id: int
    video_frame_idx: int
    t_frame: float
    xyxy_frame: Tuple[int, int, int, int]
    raw: dict


class ReplayRun:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.metadata = load_json(run_dir / 'metadata.json', {})
        self.summary = load_json(run_dir / 'run_summary.json', {})
        self.adsb = load_jsonl(run_dir / 'adsb.jsonl')
        self.frame_index_raw = load_jsonl(run_dir / 'frame_index.jsonl')
        self.track_frames_raw = load_jsonl(run_dir / 'track_frames.jsonl')
        self.roi = load_jsonl(run_dir / 'roi_index.jsonl')
        self.audio = load_jsonl(run_dir / 'audio_scores.jsonl')
        self.decisions = load_jsonl(run_dir / 'decisions.jsonl')
        self.video_path = run_dir / 'video.mp4'
        self.detections_dir = run_dir / 'detections'

        self.frames: List[FrameRecord] = [
            FrameRecord(
                frame_id=int(r.get('frame_id', idx + 1)),
                video_frame_idx=int(r.get('video_frame_idx', idx)),
                t_frame=float(r.get('t_frame', 0.0)),
                t_rel_s=float(r.get('t_rel_s', 0.0)),
            )
            for idx, r in enumerate(self.frame_index_raw)
        ]
        self.frames.sort(key=lambda r: r.video_frame_idx)

        self.track_frames: List[TrackFrameRecord] = [
            TrackFrameRecord(
                frame_id=int(r.get('frame_id', 0)),
                video_frame_idx=int(r.get('video_frame_idx', 0)),
                t_frame=float(r.get('t_frame', 0.0)),
                icao24=str(r.get('icao24', '')),
                roi_xyxy=tuple(int(v) for v in r.get('roi_xyxy', [0, 0, 0, 0])),
                raw=r,
            )
            for r in self.track_frames_raw
        ]

        self.track_frames_by_frame: Dict[int, List[TrackFrameRecord]] = defaultdict(list)
        for rec in self.track_frames:
            self.track_frames_by_frame[rec.frame_id].append(rec)

        self.detection_files = sorted(self.detections_dir.glob('*.jsonl')) if self.detections_dir.exists() else []
        self._detections_cache: Dict[str, List[DetectionRecord]] = {}
        self._detections_by_frame_cache: Dict[str, Dict[int, List[DetectionRecord]]] = {}

    def iter_roi(self) -> Iterator[RoiRecord]:
        for r in self.roi:
            path_value = r.get('path')
            yield RoiRecord(
                icao24=str(r.get('icao24', '')),
                frame_id=int(r.get('frame_id', 0)),
                t_frame=float(r.get('t_frame', 0.0)),
                roi_xyxy=tuple(int(x) for x in r.get('roi_xyxy', [0, 0, 0, 0])),
                path=(self.run_dir / Path(path_value)) if path_value else None,
            )

    def load_roi_image(self, rec: RoiRecord):
        if rec.path is None:
            return None
        return cv2.imread(str(rec.path))

    def available_detection_files(self) -> List[str]:
        return [p.name for p in self.detection_files]

    def pick_detection_file(self, selection: Optional[str]) -> Optional[Path]:
        if selection in (None, '', 'none'):
            return None
        if selection == 'auto':
            return self.detection_files[-1] if self.detection_files else None

        cand = Path(selection)
        if cand.is_absolute() and cand.exists():
            return cand

        local = self.detections_dir / selection
        if local.exists():
            return local

        for p in self.detection_files:
            if p.name == selection:
                return p
        raise FileNotFoundError(f'Could not find detection file: {selection}')

    def _load_detection_records(self, path: Path) -> List[DetectionRecord]:
        key = str(path.resolve())
        if key in self._detections_cache:
            return self._detections_cache[key]

        rows = load_jsonl(path)
        recs = [
            DetectionRecord(
                frame_id=int(r.get('frame_id', 0)),
                video_frame_idx=int(r.get('video_frame_idx', 0)),
                t_frame=float(r.get('t_frame', 0.0)),
                xyxy_frame=tuple(int(v) for v in r.get('xyxy_frame', [0, 0, 0, 0])),
                raw=r,
            )
            for r in rows
        ]
        self._detections_cache[key] = recs
        by_frame: Dict[int, List[DetectionRecord]] = defaultdict(list)
        for rec in recs:
            by_frame[rec.frame_id].append(rec)
        self._detections_by_frame_cache[key] = by_frame
        return recs

    def detection_records_for_frame(self, frame_id: int, selection: Optional[str] = None) -> List[DetectionRecord]:
        det_path = self.pick_detection_file(selection)
        if det_path is None:
            return []
        key = str(det_path.resolve())
        if key not in self._detections_cache:
            self._load_detection_records(det_path)
        return list(self._detections_by_frame_cache.get(key, {}).get(frame_id, []))

    def track_records_for_frame(self, frame_id: int) -> List[TrackFrameRecord]:
        return list(self.track_frames_by_frame.get(frame_id, []))

    def nearby_adsb_messages(self, t_center: float, window_s: float = 2.0) -> List[dict]:
        t_min = float(t_center) - float(window_s)
        t_max = float(t_center) + float(window_s)
        rows = [r for r in self.adsb if t_min <= float(r.get('t_rx', 0.0)) <= t_max]
        rows.sort(key=lambda r: abs(float(r.get('t_rx', 0.0)) - float(t_center)))
        return rows

    def latest_adsb_states(self, t_center: float, stale_s: float = 15.0) -> List[dict]:
        latest: Dict[str, dict] = {}
        for r in self.adsb:
            icao24 = str(r.get('icao24', ''))
            if not icao24:
                continue
            t_rx = float(r.get('t_rx', 0.0))
            if t_rx > float(t_center):
                continue
            if (float(t_center) - t_rx) > float(stale_s):
                continue
            prev = latest.get(icao24)
            if prev is None or float(prev.get('t_rx', 0.0)) < t_rx:
                latest[icao24] = r
        rows = list(latest.values())
        rows.sort(key=lambda r: float(t_center) - float(r.get('t_rx', 0.0)))
        return rows

    def video_exists(self) -> bool:
        return self.video_path.exists()

    def open_video(self):
        if not self.video_exists():
            raise FileNotFoundError(f'No video.mp4 found in {self.run_dir}')
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video: {self.video_path}')
        return cap

    def inferred_video_fps(self) -> float:
        if self.frames and len(self.frames) >= 2:
            dt = self.frames[-1].t_rel_s - self.frames[0].t_rel_s
            if dt > 0:
                return max(1.0, (len(self.frames) - 1) / dt)
        if self.video_exists():
            cap = cv2.VideoCapture(str(self.video_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if fps > 0:
                return fps
        cam = self.metadata.get('camera', {}) if isinstance(self.metadata, dict) else {}
        fps = cam.get('requested_fps')
        try:
            fps_f = float(fps)
            if fps_f > 0:
                return fps_f
        except Exception:
            pass
        return 15.0

    def _fallback_frame_record(self, video_frame_idx: int) -> FrameRecord:
        fps = self.inferred_video_fps()
        t_rel_s = float(video_frame_idx) / max(fps, 1e-6)
        return FrameRecord(
            frame_id=video_frame_idx + 1,
            video_frame_idx=video_frame_idx,
            t_frame=t_rel_s,
            t_rel_s=t_rel_s,
        )

    def iter_video_frames(self, limit_frames: int = 0):
        cap = self.open_video()
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if limit_frames and idx >= limit_frames:
                    break
                if idx < len(self.frames):
                    frame_rec = self.frames[idx]
                else:
                    frame_rec = self._fallback_frame_record(idx)
                yield frame, frame_rec
                idx += 1
        finally:
            cap.release()

    def summary_dict(self) -> Dict[str, object]:
        unique_icaos = sorted({str(r.get('icao24', '')) for r in self.adsb if str(r.get('icao24', ''))})
        return {
            'run_dir': str(self.run_dir),
            'video_exists': self.video_exists(),
            'video_path': str(self.video_path),
            'frames': len(self.frames),
            'adsb_messages': len(self.adsb),
            'track_frames': len(self.track_frames),
            'roi_records': len(self.roi),
            'audio_scores': len(self.audio),
            'decisions': len(self.decisions),
            'duration_s': float(self.summary.get('duration_s', self.frames[-1].t_rel_s if self.frames else 0.0)),
            'unique_icaos': unique_icaos,
            'detection_files': self.available_detection_files(),
        }

    def summary_lines(self) -> List[str]:
        s = self.summary_dict()
        lines = [
            f"run_dir={s['run_dir']}",
            f"video={s['video_exists']} frames={s['frames']} duration_s={float(s['duration_s']):.1f}",
            f"adsb_messages={s['adsb_messages']} unique_icaos={len(s['unique_icaos'])} track_frames={s['track_frames']} roi_records={s['roi_records']}",
        ]
        det_files = s['detection_files']
        if det_files:
            lines.append('detection_files=' + ', '.join(det_files))
        return lines
