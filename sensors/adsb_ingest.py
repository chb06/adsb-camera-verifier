from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from common_types import AdsbQuality, AdsbState


@dataclass
class AdsbIngestConfig:
    host: str = "127.0.0.1"
    port: int = 30003  # SBS-1/BaseStation text default in many decoders
    reconnect_sec: float = 2.0


def _to_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        x = x.strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: str) -> Optional[int]:
    try:
        if x is None:
            return None
        x = x.strip()
        if x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def parse_sbs1_line(line: str, t_rx: Optional[float] = None) -> Optional[AdsbState]:
    """Parse SBS-1/BaseStation line.

    This is intentionally tolerant and only extracts fields we need for ROI projection.

    SBS-1 field order (common):
      0 MSG
      1 transmission type
      2 session id
      3 aircraft id
      4 hex ident (icao24)
      ...
      10 callsign
      11 altitude (ft)
      12 groundspeed (knots)
      13 track (deg)
      14 latitude
      15 longitude
      16 vertical rate (ft/min)

    Many decoders provide partial lines.
    """
    if not line:
        return None
    parts = line.split(',')
    if len(parts) < 22:
        return None
    if parts[0] != 'MSG':
        return None

    icao24 = parts[4].strip().lower()
    if not icao24:
        return None

    lat = _to_float(parts[14])
    lon = _to_float(parts[15])
    alt_ft = _to_float(parts[11])

    if lat is None or lon is None or alt_ft is None:
        # can't project without position
        return None

    alt_m = alt_ft * 0.3048
    flight = parts[10].strip() or None

    # Velocity: SBS-1 provides groundspeed + track, not ENU components.
    gs_kts = _to_float(parts[12])
    trk_deg = _to_float(parts[13])
    vr_fpm = _to_float(parts[16])

    vn = 0.0
    ve = 0.0
    if gs_kts is not None and trk_deg is not None:
        gs_mps = gs_kts * 0.514444
        # track is degrees clockwise from north
        import math
        trk = math.radians(trk_deg)
        vn = gs_mps * math.cos(trk)
        ve = gs_mps * math.sin(trk)

    vu = 0.0
    if vr_fpm is not None:
        vu = (vr_fpm * 0.3048) / 60.0

    if t_rx is None:
        t_rx = time.time()

    return AdsbState(
        icao24=icao24,
        t_rx=t_rx,
        lat_deg=float(lat),
        lon_deg=float(lon),
        alt_m=float(alt_m),
        vn_mps=float(vn),
        ve_mps=float(ve),
        vu_mps=float(vu),
        quality=AdsbQuality(),
        flight=flight,
    )


class AdsbIngestor:
    def __init__(self, cfg: AdsbIngestConfig):
        self.cfg = cfg

    async def messages(self) -> AsyncIterator[AdsbState]:
        """Async generator producing parsed AdsbState messages."""
        while True:
            try:
                reader, _ = await asyncio.open_connection(self.cfg.host, self.cfg.port)
                while not reader.at_eof():
                    raw = await reader.readline()
                    if not raw:
                        break
                    t = time.time()
                    msg = parse_sbs1_line(raw.decode(errors='ignore').strip(), t_rx=t)
                    if msg:
                        yield msg
            except Exception:
                await asyncio.sleep(self.cfg.reconnect_sec)
