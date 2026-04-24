from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import cv2

from data.replay import ReplayRun
from perception.yolo_trt import YoloConfig, YoloDetector


def sanitize_name(text: str) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('._-')
    return s or 'model'


def filter_detections(dets, allowed_classes: set[str]):
    if not allowed_classes:
        return list(dets)
    return [d for d in dets if d.cls.lower() in allowed_classes]


def clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def default_output_path(run_dir: Path, mode: str, backend: str, model_path: Optional[str]) -> Path:
    model_stem = sanitize_name(Path(model_path).stem if model_path else backend)
    return run_dir / 'detections' / f'offline_{mode}_{model_stem}.jsonl'


def maybe_make_preview_writer(path: Optional[Path], fps: float, frame):
    if path is None:
        return None
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(path), fourcc, float(fps), (int(w), int(h)))


def draw_preview_boxes(frame, candidates, det_records):
    for cand in candidates:
        x1, y1, x2, y2 = [int(v) for v in cand.roi_xyxy]
        label = str(cand.raw.get('icao24', 'track'))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for rec in det_records:
        x1, y1, x2, y2 = [int(v) for v in rec['xyxy_frame']]
        adsb_bits = []
        icao24 = str(rec.get('icao24', '') or '').strip()
        flight = str(rec.get('flight', '') or '').strip()
        if icao24:
            adsb_bits.append(icao24)
        if flight:
            adsb_bits.append(flight)
        if rec.get('alt_m') is not None:
            adsb_bits.append(f"alt={float(rec['alt_m']):.0f}m")
        if rec.get('range_m') is not None:
            adsb_bits.append(f"rng={float(rec['range_m'])/1000.0:.1f}km")
        adsb_prefix = " ".join(adsb_bits)
        label = f"{adsb_prefix} {rec['cls']} {float(rec['conf']):.2f}".strip()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def run_offline_detection(args):
    run_dir = Path(args.run_dir)
    rr = ReplayRun(run_dir)
    if not rr.video_exists():
        raise SystemExit(f'No video.mp4 found in {run_dir}')

    mode = args.mode
    if mode == 'auto':
        mode = 'roi' if rr.track_frames else 'full_frame'
    if mode == 'roi' and not rr.track_frames:
        raise SystemExit('ROI mode requested but track_frames.jsonl is missing or empty.')

    detector = YoloDetector(
        YoloConfig(
            backend=args.backend,
            model_path=args.model_path,
            imgsz=args.imgsz,
            conf=args.conf,
        )
    )

    allowed_classes = {c.lower() for c in (args.allow_class or [])}
    out_path = Path(args.out) if args.out else default_output_path(run_dir, mode, detector.backend, args.model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = None
    if args.save_preview_video:
        preview_path = out_path.with_suffix('.mp4')

    frames_processed = 0
    roi_candidates_processed = 0
    detections_written = 0
    class_counts: Counter[str] = Counter()
    fps = rr.inferred_video_fps()
    preview_writer = None

    with out_path.open('w') as f_out:
        for frame, frame_rec in rr.iter_video_frames(limit_frames=args.limit_frames):
            if args.stride > 1 and (frame_rec.video_frame_idx % args.stride) != 0:
                continue
            frames_processed += 1
            frame_h, frame_w = frame.shape[:2]
            preview_det_records: List[dict] = []

            if mode == 'roi':
                candidates = rr.track_records_for_frame(frame_rec.frame_id)
                for cand in candidates:
                    x1, y1, x2, y2 = clamp_xyxy(*cand.roi_xyxy, frame_w, frame_h)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    roi_candidates_processed += 1
                    dets = filter_detections(detector.infer_bgr(roi), allowed_classes)
                    for det in dets:
                        dx1, dy1, dx2, dy2 = [int(v) for v in det.xyxy]
                        rec = {
                            'created_at': datetime.now().isoformat(),
                            'mode': 'roi',
                            'backend': detector.backend,
                            'model_path': args.model_path,
                            'frame_id': int(frame_rec.frame_id),
                            'video_frame_idx': int(frame_rec.video_frame_idx),
                            't_frame': float(frame_rec.t_frame),
                            't_rel_s': float(frame_rec.t_rel_s),
                            'icao24': str(cand.raw.get('icao24', '')),
                            'flight': str(cand.raw.get('flight', '') or '').strip(),
                            'alt_m': cand.raw.get('alt_m'),
                            'range_m': cand.raw.get('range_m'),
                            'roi_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                            'cls': det.cls,
                            'conf': float(det.conf),
                            'xyxy_roi': [dx1, dy1, dx2, dy2],
                            'xyxy_frame': [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2],
                        }
                        f_out.write(json.dumps(rec) + '\n')
                        preview_det_records.append(rec)
                        class_counts[det.cls] += 1
                        detections_written += 1
                if preview_path is not None:
                    if preview_writer is None:
                        preview_writer = maybe_make_preview_writer(preview_path, fps, frame)
                    preview_frame = frame.copy()
                    draw_preview_boxes(preview_frame, candidates, preview_det_records)
                    preview_writer.write(preview_frame)

            elif mode == 'full_frame':
                dets = filter_detections(detector.infer_bgr(frame), allowed_classes)
                for det in dets:
                    x1, y1, x2, y2 = [int(v) for v in det.xyxy]
                    x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, frame_w, frame_h)
                    rec = {
                        'created_at': datetime.now().isoformat(),
                        'mode': 'full_frame',
                        'backend': detector.backend,
                        'model_path': args.model_path,
                        'frame_id': int(frame_rec.frame_id),
                        'video_frame_idx': int(frame_rec.video_frame_idx),
                        't_frame': float(frame_rec.t_frame),
                        't_rel_s': float(frame_rec.t_rel_s),
                        'cls': det.cls,
                        'conf': float(det.conf),
                        'xyxy_frame': [x1, y1, x2, y2],
                    }
                    f_out.write(json.dumps(rec) + '\n')
                    preview_det_records.append(rec)
                    class_counts[det.cls] += 1
                    detections_written += 1
                if preview_path is not None:
                    if preview_writer is None:
                        preview_writer = maybe_make_preview_writer(preview_path, fps, frame)
                    preview_frame = frame.copy()
                    draw_preview_boxes(preview_frame, [], preview_det_records)
                    preview_writer.write(preview_frame)
            else:
                raise ValueError(f'Unknown mode: {mode}')

    if preview_writer is not None:
        preview_writer.release()

    summary = {
        'created_at': datetime.now().isoformat(),
        'run_dir': str(run_dir),
        'mode': mode,
        'backend': detector.backend,
        'model_path': args.model_path,
        'imgsz': int(args.imgsz),
        'conf': float(args.conf),
        'allow_classes': sorted(allowed_classes),
        'frames_processed': int(frames_processed),
        'roi_candidates_processed': int(roi_candidates_processed),
        'detections_written': int(detections_written),
        'output_jsonl': str(out_path),
        'preview_video': str(preview_path) if preview_path else None,
        'class_counts': dict(class_counts),
    }
    summary_path = out_path.with_name(out_path.stem + '_summary.json')
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f'[offline-detect] run_dir={run_dir}')
    print(f'[offline-detect] mode={mode} backend={detector.backend} frames_processed={frames_processed}')
    print(f'[offline-detect] roi_candidates_processed={roi_candidates_processed} detections_written={detections_written}')
    print(f'[offline-detect] output_jsonl={out_path}')
    print(f'[offline-detect] summary_json={summary_path}')
    if preview_path:
        print(f'[offline-detect] preview_video={preview_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--mode', default='auto', choices=['auto', 'roi', 'full_frame'])
    ap.add_argument('--backend', default='none', help='none | ultralytics | tensorrt')
    ap.add_argument('--model-path', default=None, help='Ultralytics .pt path or TensorRT .engine path')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--stride', type=int, default=1, help='Process every Nth video frame')
    ap.add_argument('--limit-frames', type=int, default=0)
    ap.add_argument('--allow-class', action='append', default=[], help='Repeat to keep only certain classes, e.g. --allow-class airplane')
    ap.add_argument('--save-preview-video', action='store_true')
    ap.add_argument('--out', default=None, help='Optional output .jsonl path')
    args = ap.parse_args()

    run_offline_detection(args)


if __name__ == '__main__':
    main()
