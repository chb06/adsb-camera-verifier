from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from data.replay import ReplayRun


def draw_overlay(frame, frame_rec, track_recs, adsb_rows, det_recs, max_tracks: int, max_adsb: int):
    cv2.putText(
        frame,
        f"frame={frame_rec.video_frame_idx} t={frame_rec.t_rel_s:.1f}s tracks={len(track_recs)} adsb={len(adsb_rows)} dets={len(det_recs)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    for rec in track_recs[:max_tracks]:
        x1, y1, x2, y2 = rec.roi_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = rec.icao24
        flight = str(rec.raw.get('flight', '') or '').strip()
        if flight:
            label += f" {flight}"
        if 'range_m' in rec.raw:
            label += f" {float(rec.raw['range_m'])/1000.0:.1f}km"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for det in det_recs:
        x1, y1, x2, y2 = det.xyxy_frame
        cls_name = str(det.raw.get('cls', 'det'))
        conf = float(det.raw.get('conf', 0.0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{cls_name} {conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    y = 65
    for row in adsb_rows[:max_adsb]:
        icao24 = str(row.get('icao24', ''))
        flight = str(row.get('flight', '') or '').strip()
        label = icao24 + (f" {flight}" if flight else '')
        if 'alt_m' in row:
            label += f" alt={float(row['alt_m']):.0f}m"
        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        y += 22


def print_replay_summary(rr: ReplayRun):
    print('Loaded run:')
    for line in rr.summary_lines():
        print('  ' + line)


def print_frame_summary(frame_rec, track_recs, adsb_rows, nearby_msgs, det_recs, max_tracks: int, max_adsb: int):
    print(
        f"[replay] frame={frame_rec.video_frame_idx} t_rel={frame_rec.t_rel_s:.1f}s projected_tracks={len(track_recs)} latest_adsb={len(adsb_rows)} nearby_msgs={len(nearby_msgs)} dets={len(det_recs)}"
    )
    for rec in track_recs[:max_tracks]:
        flight = str(rec.raw.get('flight', '') or '').strip()
        decision = str(rec.raw.get('decision', ''))
        det_count = int(rec.raw.get('detection_count', 0) or 0)
        print(
            f"  [track] {rec.icao24}{(' ' + flight) if flight else ''} range_km={float(rec.raw.get('range_m', 0.0))/1000.0:.1f} decision={decision} dets={det_count}"
        )
    for row in adsb_rows[:max_adsb]:
        flight = str(row.get('flight', '') or '').strip()
        age_s = max(0.0, float(frame_rec.t_frame) - float(row.get('t_rx', 0.0)))
        print(
            f"  [adsb] {row.get('icao24', '')}{(' ' + flight) if flight else ''} alt_m={float(row.get('alt_m', 0.0)):.0f} age_s={age_s:.1f}"
        )


def run_replay(args):
    rr = ReplayRun(Path(args.run_dir))
    print_replay_summary(rr)

    detection_selection = args.detections
    if detection_selection == 'auto' and not rr.available_detection_files():
        detection_selection = 'none'

    display = bool(args.display)
    paused = False
    last_summary_t = -1e12
    target_fps = rr.inferred_video_fps() * max(args.speed, 1e-6)
    frame_delay_ms = max(1, int(round(1000.0 / max(target_fps, 1e-6))))
    wall_t0 = time.time()

    try:
        for frame, frame_rec in rr.iter_video_frames(limit_frames=args.limit_frames):
            track_recs = rr.track_records_for_frame(frame_rec.frame_id)
            adsb_rows = rr.latest_adsb_states(frame_rec.t_frame, stale_s=args.stale_track_s)
            nearby_msgs = rr.nearby_adsb_messages(frame_rec.t_frame, window_s=args.adsb_window_s)
            det_recs = rr.detection_records_for_frame(frame_rec.frame_id, detection_selection)

            if (frame_rec.t_rel_s - last_summary_t) >= args.summary_interval_s:
                print_frame_summary(
                    frame_rec,
                    track_recs,
                    adsb_rows,
                    nearby_msgs,
                    det_recs,
                    max_tracks=args.max_tracks,
                    max_adsb=args.max_adsb,
                )
                last_summary_t = frame_rec.t_rel_s

            if display:
                draw_overlay(
                    frame,
                    frame_rec,
                    track_recs,
                    adsb_rows,
                    det_recs,
                    max_tracks=args.max_tracks,
                    max_adsb=args.max_adsb,
                )
                cv2.imshow('replay', frame)
                while True:
                    key = cv2.waitKey(0 if paused else frame_delay_ms) & 0xFF
                    if key == ord('q'):
                        return
                    if key == ord(' '):
                        paused = not paused
                        continue
                    if paused and key == ord('n'):
                        break
                    if key == 255:
                        break
                    if not paused:
                        break

            if args.seconds and (time.time() - wall_t0) >= args.seconds:
                break
    finally:
        if display:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--display', action='store_true')
    ap.add_argument('--speed', type=float, default=1.0, help='Replay speed multiplier')
    ap.add_argument('--seconds', type=float, default=0.0, help='0 = replay until video ends')
    ap.add_argument('--limit-frames', type=int, default=0)
    ap.add_argument('--summary-interval-s', type=float, default=2.0)
    ap.add_argument('--adsb-window-s', type=float, default=2.0)
    ap.add_argument('--stale-track-s', type=float, default=15.0)
    ap.add_argument('--max-adsb', type=int, default=5)
    ap.add_argument('--max-tracks', type=int, default=5)
    ap.add_argument('--detections', default='auto', help='auto | none | file name inside detections/ | absolute path')
    args = ap.parse_args()

    run_replay(args)


if __name__ == '__main__':
    main()
