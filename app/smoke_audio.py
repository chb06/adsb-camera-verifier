from __future__ import annotations

import argparse
import time

import numpy as np

from sensors.audio_stream import AudioConfig, AudioStream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--blocksize', type=int, default=2048)
    args = ap.parse_args()

    cfg = AudioConfig(sample_rate_hz=args.sr, blocksize=args.blocksize)
    stream = AudioStream(cfg)
    stream.start()

    print('[audio] capturing... (press Ctrl-C to stop)')
    t0 = time.time()
    blocks = 0

    try:
        while True:
            rec = stream.read_block(timeout_s=1.0)
            if rec is None:
                continue
            t, x = rec
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            peak = float(np.max(np.abs(x)))
            blocks += 1
            if blocks % 10 == 0:
                print(f"[audio] t={t - t0:6.2f}s rms={rms:.4f} peak={peak:.4f}")

            if time.time() - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass

    stream.stop()
    print('[audio] done')


if __name__ == '__main__':
    main()
