from __future__ import annotations

"""Threshold sweep scaffold.

In the full system, this will replay a run (or many runs) and sweep:
- vision persistence frames
- audio threshold
- coincidence window
- ROI sigma multiplier

For now, this file is a placeholder so the repo structure is complete.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SweepResult:
    params: dict
    precision: float
    recall: float
    false_verification_rate: float
    median_ttv_s: float
    p90_ttv_s: float


def run_sweep(run_dirs: List[str]) -> List[SweepResult]:
    raise NotImplementedError
