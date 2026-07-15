from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from assistant.audio.exceptions import AudioError
from assistant.config import STT_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int = STT_SAMPLE_RATE
    channels: int = 1

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise AudioError(f"Invalid sample_rate: {self.sample_rate}")

        if self.channels < 1:
            raise AudioError(f"Invalid channels: {self.channels}")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    samples: NDArray[np.float32]
    format: AudioFormat


@dataclass(frozen=True, slots=True)
class AudioData:
    samples: NDArray[np.float32]
    format: AudioFormat

    @property
    def sample_rate(self) -> int:
        return self.format.sample_rate

    @property
    def channels(self) -> int:
        return self.format.channels
