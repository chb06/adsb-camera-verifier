from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    blocksize: int = 2048
    device: Optional[int] = None  # None = default


class AudioStream:
    """Simple audio capture into a thread-safe queue.

    For MVP we capture raw float32 samples and timestamp blocks.
    """

    def __init__(self, cfg: AudioConfig, max_queue_blocks: int = 200):
        self.cfg = cfg
        self.q: "queue.Queue[Tuple[float, np.ndarray]]" = queue.Queue(maxsize=max_queue_blocks)
        self._stream: Optional[sd.InputStream] = None

    def start(self):
        def callback(indata, frames, time_info, status):
            # indata shape: (frames, channels)
            t = time.time()
            x = np.asarray(indata, dtype=np.float32).reshape(-1)
            try:
                self.q.put_nowait((t, x))
            except queue.Full:
                # drop if consumer is slow
                pass

        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate_hz,
            channels=self.cfg.channels,
            blocksize=self.cfg.blocksize,
            dtype='float32',
            callback=callback,
            device=self.cfg.device,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_block(self, timeout_s: float = 0.5) -> Optional[Tuple[float, np.ndarray]]:
        try:
            return self.q.get(timeout=timeout_s)
        except queue.Empty:
            return None
