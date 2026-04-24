from __future__ import annotations

import re
import select
import shutil
import subprocess
import sys
import threading
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


def open_opencv_capture(cfg: CameraConfig, source: Union[int, str]) -> cv2.VideoCapture:
    api_preference = cv2.CAP_ANY
    if sys.platform == "darwin" and isinstance(source, int):
        api_preference = cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(source, api_preference)
    if cap.isOpened() and isinstance(source, int):
        if cfg.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.width))
        if cfg.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.height))
        if cfg.fps:
            cap.set(cv2.CAP_PROP_FPS, int(cfg.fps))
    return cap


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


def avfoundation_pixel_format(cfg: CameraConfig) -> str:
    fmt = str(cfg.format or "").strip().lower()
    aliases = {
        "": "uyvy422",
        "yuy2": "uyvy422",
        "yuyv": "yuyv422",
        "yuyv422": "yuyv422",
        "uyvy": "uyvy422",
        "uyvy422": "uyvy422",
        "nv12": "nv12",
        "0rgb": "0rgb",
        "bgr0": "bgr0",
    }
    return aliases.get(fmt, fmt or "uyvy422")


def avfoundation_pixel_format_candidates(cfg: CameraConfig) -> list[Optional[str]]:
    preferred = avfoundation_pixel_format(cfg)
    candidates: list[Optional[str]] = [preferred, "yuyv422", "uyvy422", "nv12", None]
    seen: set[Optional[str]] = set()
    ordered: list[Optional[str]] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


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
        self.read_timeout_s = 3.0
        self.last_error: Optional[str] = None
        self.pixel_format: Optional[str] = None
        self.proc = self._start_process_with_fallback()
        self.frame_id = 0

    def _input_spec(self) -> str:
        return f"{self.resolved_source}:none"

    def _build_cmd(self, pixel_format: Optional[str]) -> list[str]:
        cmd = [
            self.cfg.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
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
        if pixel_format:
            cmd[8:8] = ["-pixel_format", pixel_format]
        return cmd

    def _start_process(self, pixel_format: Optional[str]) -> subprocess.Popen:
        cmd = self._build_cmd(pixel_format)
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 2,
        )

    def _start_process_with_fallback(self) -> subprocess.Popen:
        proc: Optional[subprocess.Popen] = None
        errors: list[str] = []
        for pixel_format in avfoundation_pixel_format_candidates(self.cfg):
            self.pixel_format = pixel_format
            proc = self._start_process(pixel_format)
            time.sleep(0.25)
            if proc.poll() is None:
                return proc
            stderr = self._read_stderr_excerpt_from_proc(proc)
            label = pixel_format or "auto"
            errors.append(f"{label}: {stderr or f'exited with code {proc.returncode}'}")
            self._close_proc(proc)
        joined = " | ".join(errors) if errors else "unable to start ffmpeg process"
        raise RuntimeError(f"Failed to start FFmpeg AVFoundation camera source after trying multiple pixel formats. {joined}")

    def _read_stderr_excerpt_from_proc(self, proc: subprocess.Popen) -> str:
        if proc.stderr is None:
            return ""
        parts: list[str] = []
        try:
            while True:
                ready, _, _ = select.select([proc.stderr], [], [], 0.0)
                if not ready:
                    break
                chunk = proc.stderr.read1(4096)
                if not chunk:
                    break
                parts.append(chunk.decode("utf-8", errors="replace"))
        except Exception:
            return ""
        return "".join(parts).strip()

    def _read_stderr_excerpt(self) -> str:
        return self._read_stderr_excerpt_from_proc(self.proc)

    def _close_proc(self, proc: subprocess.Popen):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def isOpened(self) -> bool:
        return self.proc.poll() is None and self.proc.stdout is not None

    def read(self):
        t0 = time.time()
        if self.proc.stdout is None:
            self.last_error = "ffmpeg stdout pipe was not created"
            return False, None, t0
        if self.proc.poll() is not None:
            stderr = self._read_stderr_excerpt()
            self.last_error = stderr or f"ffmpeg exited with code {self.proc.returncode}"
            return False, None, t0
        ready, _, _ = select.select([self.proc.stdout], [], [], self.read_timeout_s)
        if not ready:
            stderr = self._read_stderr_excerpt()
            hint = (
                "Timed out waiting for the first camera frame from FFmpeg. "
                "On macOS, make sure Terminal or iTerm has Camera permission and no other app is holding the camera."
            )
            fmt = self.pixel_format or "auto"
            self.last_error = f"{hint} pixel_format={fmt} {stderr}".strip()
            return False, None, t0
        raw = self.proc.stdout.read(self.frame_bytes)
        if raw is None or len(raw) != self.frame_bytes:
            stderr = self._read_stderr_excerpt()
            self.last_error = stderr or f"Expected {self.frame_bytes} bytes but received {0 if raw is None else len(raw)}"
            return False, None, t0
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        self.frame_id += 1
        self.last_error = None
        return True, frame, t0

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def release(self):
        self._close_proc(self.proc)


class OpenCVCameraStream:
    def __init__(self, cap: cv2.VideoCapture, source_label: str, read_timeout_s: float = 3.0):
        self.cap = cap
        self.source_label = source_label
        self.read_timeout_s = max(0.1, float(read_timeout_s))
        self.last_error: Optional[str] = None

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def read(self):
        result: dict[str, object] = {}

        def _worker():
            try:
                ok, frame = self.cap.read()
                result["ok"] = ok
                result["frame"] = frame
            except Exception as exc:  # pragma: no cover - defensive wrapper around OpenCV
                result["error"] = str(exc)

        t0 = time.time()
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join(self.read_timeout_s)
        if worker.is_alive():
            self.last_error = (
                f"Timed out waiting {self.read_timeout_s:.1f}s for a frame from OpenCV source "
                f"{self.source_label}. On macOS, make sure "
                "Terminal or iTerm has Camera permission."
            )
            return False, None, t0

        if "error" in result:
            self.last_error = str(result["error"])
            return False, None, t0

        ok = bool(result.get("ok", False))
        frame = result.get("frame")
        if not ok:
            self.last_error = f"OpenCV could not read a frame from source {self.source_label}"
            return False, None, t0

        self.last_error = None
        return True, frame, t0

    def get(self, prop_id: int) -> float:
        return float(self.cap.get(prop_id))

    def set(self, prop_id: int, value: float) -> bool:
        return bool(self.cap.set(prop_id, value))

    def release(self):
        self.cap.release()


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
            raw_cap = open_opencv_capture(cfg, self.source)
            self.cap = OpenCVCameraStream(raw_cap, str(self.source))
        elif self.backend == "ffmpeg_avfoundation":
            self.cap = FfmpegAVFoundationStream(cfg, self.source)
            self.source_label = f"{cfg.source} -> avfoundation:{self.cap.resolved_source}"
        else:
            raise ValueError(f"Unknown camera backend: {cfg.backend}")

        if not self.cap.isOpened():
            detail = self.pipeline or self.source_label
            raise RuntimeError(
                f"Failed to open camera/video source: {detail}. "
                "On Mac, use backend=opencv for the normal path, or force backend=ffmpeg_avfoundation "
                "with a camera name from app.list_cameras if you need the alternate AVFoundation path. "
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

    def last_error(self) -> str:
        return str(getattr(self.cap, "last_error", "") or "")

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
