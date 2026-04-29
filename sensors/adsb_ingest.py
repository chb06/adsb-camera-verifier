from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

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


@dataclass
class Sbs1Update:
    icao24: str
    t_rx: float
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None
    vn_mps: Optional[float] = None
    ve_mps: Optional[float] = None
    vu_mps: Optional[float] = None
    flight: Optional[str] = None


def _parse_sbs1_update(line: str, t_rx: Optional[float] = None) -> Optional[Sbs1Update]:
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

    if t_rx is None:
        t_rx = time.time()

    lat = _to_float(parts[14])
    lon = _to_float(parts[15])
    alt_ft = _to_float(parts[11])
    flight = parts[10].strip() or None

    alt_m = alt_ft * 0.3048 if alt_ft is not None else None

    gs_kts = _to_float(parts[12])
    trk_deg = _to_float(parts[13])
    vr_fpm = _to_float(parts[16])

    vn = None
    ve = None
    if gs_kts is not None and trk_deg is not None:
        gs_mps = gs_kts * 0.514444
        import math

        trk = math.radians(trk_deg)
        vn = gs_mps * math.cos(trk)
        ve = gs_mps * math.sin(trk)

    vu = None
    if vr_fpm is not None:
        vu = (vr_fpm * 0.3048) / 60.0

    return Sbs1Update(
        icao24=icao24,
        t_rx=float(t_rx),
        lat_deg=lat,
        lon_deg=lon,
        alt_m=alt_m,
        vn_mps=vn,
        ve_mps=ve,
        vu_mps=vu,
        flight=flight,
    )


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
    upd = _parse_sbs1_update(line, t_rx=t_rx)
    if upd is None:
        # can't project without position
        return None
    if upd.lat_deg is None or upd.lon_deg is None or upd.alt_m is None:
        return None

    return AdsbState(
        icao24=upd.icao24,
        t_rx=float(upd.t_rx),
        lat_deg=float(upd.lat_deg),
        lon_deg=float(upd.lon_deg),
        alt_m=float(upd.alt_m),
        vn_mps=float(upd.vn_mps or 0.0),
        ve_mps=float(upd.ve_mps or 0.0),
        vu_mps=float(upd.vu_mps or 0.0),
        quality=AdsbQuality(),
        flight=upd.flight,
    )


class AdsbIngestor:
    def __init__(self, cfg: AdsbIngestConfig):
        self.cfg = cfg
        self._state_cache: Dict[str, Dict[str, object]] = {}

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
                    upd = _parse_sbs1_update(raw.decode(errors='ignore').strip(), t_rx=t)
                    if upd is None:
                        continue

                    cache = self._state_cache.setdefault(upd.icao24, {})
                    cache['t_rx'] = float(upd.t_rx)
                    if upd.flight is not None:
                        cache['flight'] = upd.flight
                    if upd.lat_deg is not None:
                        cache['lat_deg'] = float(upd.lat_deg)
                    if upd.lon_deg is not None:
                        cache['lon_deg'] = float(upd.lon_deg)
                    if upd.alt_m is not None:
                        cache['alt_m'] = float(upd.alt_m)
                    if upd.vn_mps is not None:
                        cache['vn_mps'] = float(upd.vn_mps)
                    if upd.ve_mps is not None:
                        cache['ve_mps'] = float(upd.ve_mps)
                    if upd.vu_mps is not None:
                        cache['vu_mps'] = float(upd.vu_mps)

                    if 'lat_deg' not in cache or 'lon_deg' not in cache or 'alt_m' not in cache:
                        continue

                    yield AdsbState(
                        icao24=upd.icao24,
                        t_rx=float(cache.get('t_rx', upd.t_rx)),
                        lat_deg=float(cache['lat_deg']),
                        lon_deg=float(cache['lon_deg']),
                        alt_m=float(cache['alt_m']),
                        vn_mps=float(cache.get('vn_mps', 0.0)),
                        ve_mps=float(cache.get('ve_mps', 0.0)),
                        vu_mps=float(cache.get('vu_mps', 0.0)),
                        quality=AdsbQuality(),
                        flight=cache.get('flight') or None,
                    )
            except Exception:
                await asyncio.sleep(self.cfg.reconnect_sec)
