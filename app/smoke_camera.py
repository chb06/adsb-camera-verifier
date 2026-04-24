from __future__ import annotations

import argparse
import time

import cv2

from sensors.camera_capture import CameraConfig, CameraStream, mode_to_wh_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='0', help='Camera index, camera name for ffmpeg_avfoundation, or a video file path')
    ap.add_argument('--backend', default='opencv', help='opencv | gstreamer | ffmpeg_avfoundation')
    ap.add_argument('--mode', default='720p30', help='4k15 | 1080p60 | 1080p30 | 720p90 | 720p60 | 720p30 | 720p15 | 480p30 | 480p15')
    ap.add_argument('--display', action='store_true')
    ap.add_argument('--seconds', type=float, default=10.0)
    args = ap.parse_args()

    w, h, fps = mode_to_wh_fps(args.mode)
    cfg = CameraConfig(source=args.source, width=w, height=h, fps=fps, backend=args.backend)
    cam = CameraStream(cfg)
    actual_w, actual_h = cam.actual_wh()

    t_start = time.time()
    n = 0
    t_last = t_start

    source_desc = cam.pipeline if cam.pipeline else getattr(cam, 'source_label', args.source)
    print(f"[camera] source: {source_desc}")
    print(f"[camera] requested {w}x{h}@{fps}  actual {actual_w}x{actual_h}")

    while True:
        ok, frame, _t_frame = cam.read()
        if not ok:
            detail = cam.last_error()
            if detail:
                print(f'[camera] read failed: {detail}')
            else:
                print('[camera] read failed')
            break
        n += 1

        now = time.time()
        if now - t_last >= 1.0:
            fps_meas = n / (now - t_start)
            print(f"[camera] frames={n} avg_fps={fps_meas:.1f}")
            t_last = now

        if args.display:
            cv2.putText(frame, f"{actual_w}x{actual_h}@{fps} frame {n}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('smoke_camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if now - t_start >= args.seconds:
            break

    cam.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
