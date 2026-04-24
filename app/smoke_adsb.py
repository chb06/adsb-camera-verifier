from __future__ import annotations

import argparse
import asyncio

from sensors.adsb_ingest import AdsbIngestConfig, AdsbIngestor


async def run(host: str, port: int, seconds: float):
    ing = AdsbIngestor(AdsbIngestConfig(host=host, port=port))
    n = 0
    print(f"[adsb] connecting to {host}:{port} (SBS-1 text)")

    async def consume():
        nonlocal n
        async for msg in ing.messages():
            n += 1
            if n <= 5 or n % 50 == 0:
                print(
                    f"[adsb] n={n} icao={msg.icao24} lat={msg.lat_deg:.5f} lon={msg.lon_deg:.5f} alt_m={msg.alt_m:.1f}"
                )

    try:
        await asyncio.wait_for(consume(), timeout=seconds)
    except asyncio.TimeoutError:
        print(f"[adsb] done after {seconds}s messages={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=30003)
    ap.add_argument('--seconds', type=float, default=10.0)
    args = ap.parse_args()

    asyncio.run(run(args.host, args.port, args.seconds))


if __name__ == '__main__':
    main()
