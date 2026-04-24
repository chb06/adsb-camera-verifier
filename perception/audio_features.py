from __future__ import annotations

import numpy as np
from scipy.signal import stft


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """Create a mel filterbank matrix (n_mels, n_freq_bins)."""
    fmax = min(fmax, sr / 2.0)
    n_freq = n_fft // 2 + 1

    mels = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = mel_to_hz(mels)

    # FFT bin frequencies
    freqs = np.linspace(0, sr / 2.0, n_freq)

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        f_left, f_center, f_right = hz[i], hz[i + 1], hz[i + 2]
        # rising slope
        left = (freqs - f_left) / (f_center - f_left + 1e-9)
        # falling slope
        right = (f_right - freqs) / (f_right - f_center + 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(left, right))

    # Normalize to unit area per filter (optional but common)
    fb /= (fb.sum(axis=1, keepdims=True) + 1e-9)
    return fb


def log_mel(
    pcm: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """Compute log-mel spectrogram for mono float32 PCM."""
    if pcm.ndim != 1:
        pcm = pcm.reshape(-1)

    # STFT
    f, t, Zxx = stft(pcm, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, padded=False, boundary=None)
    S = np.abs(Zxx) ** 2

    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    M = fb @ S  # (n_mels, n_frames)
    return np.log(M + 1e-6).astype(np.float32)
