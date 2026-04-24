from __future__ import annotations

from fusion.state_machine import FusionConfig, FusionStateMachine, Evidence, TrackDecision


def test_verification_requires_both_modalities_when_configured():
    f = FusionStateMachine(FusionConfig(max_verify_time_s=10.0, coincidence_window_s=3.0, require_audio_for_verify=True))
    icao = 'abc123'

    d1 = f.update(icao, t=0.0, in_corridor=True, ev=Evidence(vision_ok=True, audio_ok=False))
    assert d1 == TrackDecision.IN_PROGRESS

    d2 = f.update(icao, t=1.0, in_corridor=True, ev=Evidence(vision_ok=True, audio_ok=True))
    assert d2 == TrackDecision.VERIFIED


def test_verification_can_run_vision_only_when_configured():
    f = FusionStateMachine(FusionConfig(max_verify_time_s=10.0, coincidence_window_s=3.0, require_audio_for_verify=False))
    icao = 'def456'

    d1 = f.update(icao, t=0.0, in_corridor=True, ev=Evidence(vision_ok=True, audio_ok=False))
    assert d1 == TrackDecision.VERIFIED
