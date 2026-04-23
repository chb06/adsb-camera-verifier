"""Backward-compatible camera module.

This project started with a Jetson/Linux-only GStreamer camera path.
For Mac-first development we now route camera access through the
cross-platform camera_capture module.
"""

from sensors.camera_capture import CameraConfig, CameraStream, build_gst_pipeline, mode_to_wh_fps

__all__ = [
    'CameraConfig',
    'CameraStream',
    'build_gst_pipeline',
    'mode_to_wh_fps',
]
