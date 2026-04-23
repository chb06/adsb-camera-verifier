from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass
class AudioDetectorConfig:
    sr: int = 16000
    # Typical aircraft sound has substantial low-frequency energy, but wind/traffic
    # also lives there. Start with mid-band emphasis and tune per site.
    band_hz: tuple[float, float] = (200.0, 2500.0)
    smooth_s: float = 1.0


class AudioDetector:
    """MVP audio event detector.

    This is *not* a final model. It is a simple bandpass RMS detector that gives
    you a stable, tuneable signal for fusion.

    Later you can replace this with a trained neural model fed by log-mel features.
    """

    def __init__(self, cfg: AudioDetectorConfig):
        self.cfg = cfg
        lo, hi = cfg.band_hz
        self.sos = butter(4, [lo, hi], btype='bandpass', fs=cfg.sr, output='sos')

    def score(self, pcm_f32: np.ndarray) -> float:
        x = np.asarray(pcm_f32, dtype=np.float32).reshape(-1)
        y = sosfilt(self.sos, x)
        rms = float(np.sqrt(np.mean(y * y) + 1e-12))
        # Map RMS to [0,1] with a gentle curve; tune later.
        score = rms / (rms + 0.05)
        return float(score)
