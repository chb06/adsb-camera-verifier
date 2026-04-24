from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.replay import ReplayRun


def build_detection_lookup(rr: ReplayRun, selection: str) -> Dict[tuple[int, str], List[dict]]:
    det_path = rr.pick_detection_file(selection)
    if det_path is None:
        return {}
    recs = rr._load_detection_records(det_path)  # internal helper is okay for this repo utility
    out: Dict[tuple[int, str], List[dict]] = {}
    for rec in recs:
        icao24 = str(rec.raw.get('icao24', '') or '')
        key = (int(rec.frame_id), icao24)
        out.setdefault(key, []).append(rec.raw)
        out.setdefault((int(rec.frame_id), ''), []).append(rec.raw)
    return out


def default_out_path(run_dir: Path) -> Path:
    out_dir = run_dir / 'exports'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / 'dataset_manifest.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--detections', default='auto', help='auto | none | file name inside detections/ | absolute path')
    ap.add_argument('--only-detected', action='store_true', help='Keep only rows with at least one offline detection')
    ap.add_argument('--stride', type=int, default=1, help='Keep every Nth row to reduce size')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rr = ReplayRun(run_dir)
    out_path = Path(args.out) if args.out else default_out_path(run_dir)
    det_lookup = build_detection_lookup(rr, args.detections)

    rows = 0
    with out_path.open('w', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow(
            [
                'run_dir',
                'frame_id',
                'video_frame_idx',
                't_frame',
                't_rel_s',
                'icao24',
                'flight',
                'roi_path',
                'roi_x1',
                'roi_y1',
                'roi_x2',
                'roi_y2',
                'range_m',
                'az_deg',
                'el_deg',
                'decision',
                'detected_this_frame',
                'vision_confirmed',
                'offline_det_count',
                'offline_best_cls',
                'offline_best_conf',
            ]
        )

        if rr.track_frames:
            for idx, rec in enumerate(rr.track_frames):
                if args.stride > 1 and (idx % args.stride) != 0:
                    continue
                dets = det_lookup.get((int(rec.frame_id), rec.icao24), det_lookup.get((int(rec.frame_id), ''), []))
                if args.only_detected and not dets:
                    continue
                best = max(dets, key=lambda d: float(d.get('conf', 0.0))) if dets else None
                x1, y1, x2, y2 = rec.roi_xyxy
                w.writerow(
                    [
                        str(run_dir),
                        rec.frame_id,
                        rec.video_frame_idx,
                        rec.t_frame,
                        float(rec.raw.get('t_rel_s', 0.0) or 0.0),
                        rec.icao24,
                        str(rec.raw.get('flight', '') or ''),
                        str(rec.raw.get('roi_path', '') or ''),
                        x1,
                        y1,
                        x2,
                        y2,
                        float(rec.raw.get('range_m', 0.0) or 0.0),
                        float(rec.raw.get('az_deg', 0.0) or 0.0),
                        float(rec.raw.get('el_deg', 0.0) or 0.0),
                        str(rec.raw.get('decision', '') or ''),
                        bool(rec.raw.get('detected_this_frame', False)),
                        bool(rec.raw.get('vision_confirmed', False)),
                        len(dets),
                        str(best.get('cls', '') if best else ''),
                        float(best.get('conf', 0.0)) if best else 0.0,
                    ]
                )
                rows += 1
        else:
            # Backward-compatible fallback for older runs that only have roi_index.jsonl.
            for idx, rec in enumerate(rr.iter_roi()):
                if args.stride > 1 and (idx % args.stride) != 0:
                    continue
                dets = det_lookup.get((int(rec.frame_id), rec.icao24), det_lookup.get((int(rec.frame_id), ''), []))
                if args.only_detected and not dets:
                    continue
                best = max(dets, key=lambda d: float(d.get('conf', 0.0))) if dets else None
                x1, y1, x2, y2 = rec.roi_xyxy
                w.writerow(
                    [
                        str(run_dir),
                        rec.frame_id,
                        '',
                        rec.t_frame,
                        '',
                        rec.icao24,
                        '',
                        str(rec.path) if rec.path else '',
                        x1,
                        y1,
                        x2,
                        y2,
                        '',
                        '',
                        '',
                        '',
                        '',
                        '',
                        len(dets),
                        str(best.get('cls', '') if best else ''),
                        float(best.get('conf', 0.0)) if best else 0.0,
                    ]
                )
                rows += 1

    print(f'Wrote {rows} rows to {out_path}')


if __name__ == '__main__':
    main()
