from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class VisionPersistConfig:
    confirm_frames: int = 4
    window_s: float = 1.0


class VisionPersistence:
    """Track-level vision evidence persistence.

    MVP logic:
    - store a time-stamped boolean "detected" per frame
    - confirm if >= confirm_frames detections in the last window_s seconds

    This is intentionally simple; you can replace with EMA/HMM later.
    """

    def __init__(self, cfg: VisionPersistConfig):
        self.cfg = cfg
        self._hist: Dict[str, Deque[tuple[float, bool]]] = {}

    def update(self, track_id: str, t: float, detected: bool):
        h = self._hist.setdefault(track_id, deque())
        h.append((t, detected))
        # evict old
        t_min = t - self.cfg.window_s
        while h and h[0][0] < t_min:
            h.popleft()

    def confirmed(self, track_id: str) -> bool:
        h = self._hist.get(track_id)
        if not h:
            return False
        return sum(1 for _, d in h if d) >= self.cfg.confirm_frames
