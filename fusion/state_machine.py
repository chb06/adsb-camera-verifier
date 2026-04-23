from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional


class TrackDecision(Enum):
    IN_PROGRESS = auto()
    VERIFIED = auto()
    UNVERIFIED = auto()


class TrackStatus(Enum):
    OUTSIDE_CORRIDOR = auto()
    IN_CORRIDOR = auto()
    VERIFIED = auto()
    EXPIRED = auto()


@dataclass
class FusionConfig:
    max_verify_time_s: float = 60.0
    coincidence_window_s: float = 3.0
    require_audio_for_verify: bool = True


@dataclass
class TrackFusion:
    icao24: str
    status: TrackStatus = TrackStatus.OUTSIDE_CORRIDOR
    t_entry: Optional[float] = None
    t_verified: Optional[float] = None
    last_reason: str = "init"


@dataclass
class Evidence:
    vision_ok: bool
    audio_ok: bool


class FusionStateMachine:
    """Per-track fusion state machine producing VERIFIED/UNVERIFIED.

    Policy:
    - VERIFIED only if vision_ok is True and, when configured, audio_ok is also True.
    - UNVERIFIED if corridor ends or timeout occurs without verification.
    """

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.tracks: Dict[str, TrackFusion] = {}

    def ensure_track(self, icao24: str) -> TrackFusion:
        return self.tracks.setdefault(icao24, TrackFusion(icao24=icao24))

    def on_corridor_entry(self, icao24: str, t: float):
        tr = self.ensure_track(icao24)
        tr.status = TrackStatus.IN_CORRIDOR
        tr.t_entry = t
        tr.t_verified = None
        tr.last_reason = "entered_corridor"

    def update(self, icao24: str, t: float, in_corridor: bool, ev: Evidence) -> TrackDecision:
        tr = self.ensure_track(icao24)

        if not in_corridor:
            if tr.status == TrackStatus.IN_CORRIDOR and tr.t_verified is None:
                tr.status = TrackStatus.EXPIRED
                tr.last_reason = "left_corridor"
                return TrackDecision.UNVERIFIED
            return TrackDecision.IN_PROGRESS

        if tr.status in (TrackStatus.OUTSIDE_CORRIDOR, TrackStatus.EXPIRED):
            self.on_corridor_entry(icao24, t)

        has_required_modalities = ev.vision_ok and (ev.audio_ok or not self.cfg.require_audio_for_verify)
        if has_required_modalities:
            if tr.status != TrackStatus.VERIFIED:
                tr.status = TrackStatus.VERIFIED
                tr.t_verified = t
                tr.last_reason = "vision+audio" if self.cfg.require_audio_for_verify else "vision_only"
            return TrackDecision.VERIFIED

        if tr.t_entry is not None and (t - tr.t_entry) > self.cfg.max_verify_time_s:
            tr.status = TrackStatus.EXPIRED
            tr.last_reason = "timeout"
            return TrackDecision.UNVERIFIED

        tr.last_reason = "waiting"
        return TrackDecision.IN_PROGRESS
