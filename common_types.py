from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AdsbQuality:
    """Subset of ADS-B quality fields.

    This is intentionally conservative and configurable.
    Many feeds will omit fields or provide them under different names.
    """

    nacp: Optional[int] = None  # position accuracy category
    nic: Optional[int] = None   # integrity category
    nacv: Optional[int] = None  # velocity accuracy category
    sil: Optional[int] = None   # source integrity level


@dataclass(frozen=True)
class AdsbState:
    icao24: str
    t_rx: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    vn_mps: float
    ve_mps: float
    vu_mps: float
    quality: AdsbQuality
    flight: Optional[str] = None


@dataclass(frozen=True)
class CameraFrame:
    t_frame: float
    frame_id: int
    bgr: "object"  # numpy ndarray


@dataclass(frozen=True)
class AudioWindow:
    t_start: float
    t_end: float
    pcm_f32: "object"  # numpy float32 mono


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    xyxy: Tuple[int, int, int, int]
