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
