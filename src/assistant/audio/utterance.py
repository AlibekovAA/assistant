from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from assistant.audio.dsp import rms, to_mono, trim_silence
from assistant.audio.models import AudioData, AudioFormat
from assistant.config import UtteranceConfig
from assistant.logger import Logger


class ReadAudio(Protocol):
    def __call__(self, *, timeout: float | None = 0.1) -> AudioData | None: ...


class UtteranceCapture:
    def __init__(self, config: UtteranceConfig) -> None:
        self._config = config
        self._logger = Logger.get(__name__)

    def capture(
        self,
        *,
        audio_format: AudioFormat,
        read_audio: ReadAudio,
        stop_event: threading.Event,
    ) -> AudioData | None:
        chunks: list[NDArray[np.float32]] = []
        state = _SpeechState()
        started_at = time.monotonic()

        while not stop_event.is_set():
            if time.monotonic() - started_at >= self._config.utterance_max_seconds:
                break

            audio = read_audio(timeout=0.1)
            if audio is None:
                continue

            samples = to_mono(audio.samples)
            chunks.append(samples)
            if state.update(rms(samples), time.monotonic(), self._config):
                break

        if stop_event.is_set() or not chunks or not state.has_speech:
            return None

        raw = np.concatenate(chunks, axis=0)
        trimmed = trim_silence(
            raw,
            threshold=self._config.speech_rms_threshold * 0.7,
            sample_rate=audio_format.sample_rate,
            pad_seconds=0.25,
        )
        duration = float(trimmed.shape[0] / audio_format.sample_rate)
        self._logger.info("Captured utterance: %.2fs", duration)

        if trimmed.size == 0:
            return None

        return AudioData(
            samples=trimmed,
            format=AudioFormat(sample_rate=audio_format.sample_rate, channels=1),
        )


@dataclass(slots=True)
class _SpeechState:
    speech_started_at: float | None = None
    loud_started_at: float | None = None
    silence_started_at: float | None = None

    @property
    def has_speech(self) -> bool:
        return self.speech_started_at is not None

    def update(self, level: float, now: float, config: UtteranceConfig) -> bool:
        if level >= config.speech_rms_threshold:
            self._handle_speech(now, config)
            return False

        self.loud_started_at = None
        return self._has_finished(now, config)

    def _handle_speech(self, now: float, config: UtteranceConfig) -> None:
        self.loud_started_at = self.loud_started_at or now
        self.silence_started_at = None

        if self.has_speech or now - self.loud_started_at < config.speech_onset_seconds:
            return

        self.speech_started_at = self.loud_started_at

    def _has_finished(self, now: float, config: UtteranceConfig) -> bool:
        speech_started_at = self.speech_started_at
        if speech_started_at is None:
            return False

        self.silence_started_at = self.silence_started_at or now
        return (
            now - speech_started_at >= config.min_speech_seconds
            and now - self.silence_started_at >= config.silence_seconds
        )
