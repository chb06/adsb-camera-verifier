from __future__ import annotations

"""Calibration tools.

Stage 1 (do this early): **Intrinsics**
- Print a checkerboard and capture ~20-40 images with varied angles/distances.
- Use calibrate_intrinsics_checkerboard() to compute K and distortion.

Stage 2 (after ADS-B ingest + ROI overlay works): **Extrinsics**
- Fit camera yaw/pitch/roll (and optional time offset) by minimizing reprojection
  error between observed aircraft pixels and projected ADS-B.

Only Stage 1 is implemented in this scaffold.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class IntrinsicsResult:
    K: np.ndarray
    dist: np.ndarray
    rms_reproj_error: float
    image_size: Tuple[int, int]
    used_images: int


def calibrate_intrinsics_checkerboard(
    image_paths: List[Path],
    checkerboard_size: Tuple[int, int],
    square_size_m: float,
) -> IntrinsicsResult:
    """Compute intrinsics from checkerboard images.

    checkerboard_size is (cols, rows) = number of inner corners.

    Returns OpenCV-style K and dist (k1,k2,p1,p2,k3).
    """
    import cv2

    cols, rows = checkerboard_size

    # 3D object points in checkerboard coordinates
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    img_size = None
    used = 0

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w,h)

        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not ret:
            continue

        # refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)
        used += 1

    if used < 8:
        raise RuntimeError(
            f"Not enough valid checkerboard detections ({used}). "
            "Collect more images with the full board visible and sharp."
        )

    assert img_size is not None

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return IntrinsicsResult(
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        rms_reproj_error=float(ret),
        image_size=img_size,
        used_images=used,
    )


@dataclass
class ExtrinsicsResult:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    time_offset_s: float
    rms_px: float


def fit_extrinsics_from_adsb_overlay(
    adsb_samples: "object",
    pixel_observations: "object",
    initial_guess: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    fit_time_offset: bool = True,
) -> ExtrinsicsResult:
    """Fit extrinsics (and optionally time offset) to minimize reprojection error.

    TODO: implement after logging + labeling exist.
    """
    raise NotImplementedError
