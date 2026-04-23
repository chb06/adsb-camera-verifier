from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Cross-platform camera/video input configuration.

    backend:
    - opencv: default for Mac/Linux/Windows webcams or video files
    - gstreamer: for Linux / Jetson camera pipelines
    - ffmpeg_avfoundation: macOS device capture by camera name or AVFoundation index
    """

    source: Union[int, str] = 0
    width: Optional[int] = 1280
    height: Optional[int] = 720
    fps: Optional[int] = 30
    backend: str = "opencv"
    format: str = "YUY2"
    ffmpeg_path: str = "ffmpeg"


_PRESET_MODES = {
    "4k15": (3840, 2160, 15),
    "2160p15": (3840, 2160, 15),
    "3840x2160@15": (3840, 2160, 15),
    "1080p60": (1920, 1080, 60),
    "1080p30": (1920, 1080, 30),
    "1920x1080@60": (1920, 1080, 60),
    "1920x1080@30": (1920, 1080, 30),
    "720p90": (1280, 720, 90),
    "720p60": (1280, 720, 60),
    "720p30": (1280, 720, 30),
    "720p15": (1280, 720, 15),
    "1280x720@90": (1280, 720, 90),
    "1280x720@60": (1280, 720, 60),
    "1280x720@30": (1280, 720, 30),
    "1280x720@15": (1280, 720, 15),
    "480p30": (640, 480, 30),
    "480p15": (640, 480, 15),
    "640x480@30": (640, 480, 30),
    "640x480@15": (640, 480, 15),
}


def mode_to_wh_fps(mode: str) -> Tuple[int, int, int]:
    key = mode.lower().strip()
    if key not in _PRESET_MODES:
        raise ValueError(f"Unknown camera mode: {mode}")
    return _PRESET_MODES[key]


def parse_source(source: Union[str, int, None]) -> Union[int, str]:
    if source is None:
        return 0
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s


def list_avfoundation_video_devices(ffmpeg_path: str = "ffmpeg") -> list[tuple[int, str]]:
    if sys.platform != "darwin":
        return []
    if shutil.which(ffmpeg_path) is None:
        return []
    cmd = [ffmpeg_path, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stderr or "") + "\n" + (proc.stdout or "")
    devices: list[tuple[int, str]] = []
    in_video = False
    for line in output.splitlines():
        if "AVFoundation video devices:" in line:
            in_video = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video:
            continue
        match = re.search(r"\[(\d+)\]\s+(.*)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def resolve_avfoundation_source(source: Union[int, str], ffmpeg_path: str = "ffmpeg") -> int:
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)

    devices = list_avfoundation_video_devices(ffmpeg_path)
    if not devices:
        raise RuntimeError("No AVFoundation video devices were found.")

    norm = s.casefold()
    exact = [idx for idx, name in devices if name.casefold() == norm]
    if exact:
        return exact[0]

    partial = [idx for idx, name in devices if norm in name.casefold()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(name for _, name in devices)
        raise RuntimeError(f"Camera name '{s}' matched multiple devices. Available devices: {names}")

    names = ", ".join(name for _, name in devices)
    raise RuntimeError(f"Camera '{s}' was not found. Available devices: {names}")


def build_gst_pipeline(cfg: CameraConfig) -> str:
    device = str(cfg.source)
    width = int(cfg.width or 1920)
    height = int(cfg.height or 1080)
    fps = int(cfg.fps or 30)
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw,format={cfg.format},width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class FfmpegAVFoundationStream:
    def __init__(self, cfg: CameraConfig, source: Union[int, str]):
        if sys.platform != "darwin":
            raise RuntimeError("ffmpeg_avfoundation is only supported on macOS")
        if shutil.which(cfg.ffmpeg_path) is None:
            raise RuntimeError(
                "ffmpeg was not found on PATH. Install ffmpeg, then retry, "
                "or use backend=opencv."
            )
        self.cfg = cfg
        self.source = source
        self.resolved_source = resolve_avfoundation_source(source, cfg.ffmpeg_path)
        self.width = int(cfg.width or 1280)
        self.height = int(cfg.height or 720)
        self.fps = int(cfg.fps or 30)
        self.frame_bytes = self.width * self.height * 3
        self.proc = self._start_process()
        self.frame_id = 0

    def _input_spec(self) -> str:
        return f"{self.resolved_source}:none"

    def _start_process(self) -> subprocess.Popen:
        cmd = [
            self.cfg.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            self._input_spec(),
            "-an",
            "-sn",
            "-dn",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_bytes * 2,
        )

    def isOpened(self) -> bool:
        return self.proc.poll() is None and self.proc.stdout is not None

    def read(self):
        t0 = time.time()
        if self.proc.stdout is None:
            return False, None, t0
        raw = self.proc.stdout.read(self.frame_bytes)
        if raw is None or len(raw) != self.frame_bytes:
            return False, None, t0
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        self.frame_id += 1
        return True, frame, t0

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class CameraStream:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.backend = cfg.backend.lower().strip()
        self.source = parse_source(cfg.source)
        self.pipeline: Optional[str] = None
        self.source_label = str(cfg.source)

        if self.backend == "gstreamer":
            self.pipeline = build_gst_pipeline(cfg)
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        elif self.backend == "opencv":
            api_preference = cv2.CAP_ANY
            if sys.platform == "darwin" and isinstance(self.source, int):
                api_preference = cv2.CAP_AVFOUNDATION
            self.cap = cv2.VideoCapture(self.source, api_preference)
            if self.cap.isOpened() and isinstance(self.source, int):
                if cfg.width:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.width))
                if cfg.height:
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.height))
                if cfg.fps:
                    self.cap.set(cv2.CAP_PROP_FPS, int(cfg.fps))
        elif self.backend == "ffmpeg_avfoundation":
            self.cap = FfmpegAVFoundationStream(cfg, self.source)
            self.source_label = f"{cfg.source} -> avfoundation:{self.cap.resolved_source}"
        else:
            raise ValueError(f"Unknown camera backend: {cfg.backend}")

        if not self.cap.isOpened():
            detail = self.pipeline or self.source_label
            raise RuntimeError(
                f"Failed to open camera/video source: {detail}. "
                "On Mac, try backend=ffmpeg_avfoundation with a camera name from app.list_cameras, "
                "or use a video file path. "
                "On Jetson/Linux, use backend=gstreamer with a /dev/video* device."
            )

        self.frame_id = 0

    def read(self):
        rec = self.cap.read()
        if isinstance(rec, tuple) and len(rec) == 3:
            ok, frame, t0 = rec
        else:
            ok, frame = rec
            t0 = time.time()
        if not ok:
            return False, None, t0
        self.frame_id += 1
        return True, frame, t0

    def actual_wh(self) -> Tuple[int, int]:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
