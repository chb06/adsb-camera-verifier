from __future__ import annotations

import shutil
import sys

from sensors.camera_capture import list_avfoundation_video_devices


def main():
    if sys.platform != "darwin":
        raise SystemExit("list_cameras currently supports macOS only.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH. Install ffmpeg first.")

    devices = list_avfoundation_video_devices(ffmpeg)
    if not devices:
        raise SystemExit("No AVFoundation video devices found.")

    print("AVFoundation video devices:")
    for idx, name in devices:
        print(f"  [{idx}] {name}")


if __name__ == "__main__":
    main()
