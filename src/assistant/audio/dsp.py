from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from assistant.audio.exceptions import AudioError


def to_mono(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    data = np.asarray(samples, dtype=np.float32)

    if data.ndim == 1:
        return np.ascontiguousarray(data)

    if data.ndim == 2:
        if data.shape[1] == 1:
            return np.ascontiguousarray(data[:, 0])
        return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32))

    raise AudioError(f"Unsupported audio shape: {data.shape}")


def rms(samples: NDArray[np.float32]) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def trim_silence(
    samples: NDArray[np.float32],
    *,
    threshold: float,
    sample_rate: int,
    pad_seconds: float = 0.2,
) -> NDArray[np.float32]:
    data = to_mono(samples)
    if data.size == 0:
        return data

    window = max(1, int(sample_rate * 0.02))
    pad = max(0, int(sample_rate * pad_seconds))
    energies = [
        float(np.sqrt(np.mean(np.square(data[index : index + window])))) for index in range(0, data.size, window)
    ]

    if not energies:
        return data

    active = [index for index, energy in enumerate(energies) if energy >= threshold]
    if not active:
        return data

    start = max(0, active[0] * window - pad)
    end = min(data.size, (active[-1] + 1) * window + pad)
    return np.ascontiguousarray(data[start:end])
