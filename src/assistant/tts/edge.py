import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import time

import edge_tts
from edge_tts.exceptions import EdgeTTSException
import miniaudio
import numpy as np
from numpy.typing import NDArray

from assistant.audio.player import PcmQueue
from assistant.config import TtsConfig
from assistant.constants.tts import (
    TTS_DEFAULT_TIMEOUT_SECONDS,
    TTS_STREAM_DECODE_STEP_BYTES,
    TTS_STREAM_START_BYTES,
)
from assistant.core.exceptions import TtsError
from assistant.logger import Logger

_LOG = Logger.get(__name__)
_TTS_ERRORS = (EdgeTTSException, OSError, RuntimeError, ValueError, TypeError)


@dataclass(slots=True)
class StreamSpeakTiming:
    first_pcm_ms: float = 0.0
    tts_ms: float = 0.0


class EdgeTts:
    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._ready = False

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    def initialize(self) -> None:
        if self._ready:
            return

        if not self._config.voice.strip():
            raise TtsError("TTS voice must not be empty")

        self._ready = True
        _LOG.info(
            "TTS ready (engine=edge-tts, voice=%s, sample_rate=%d)",
            self._config.voice,
            self._config.sample_rate,
        )

    def shutdown(self) -> None:
        self._ready = False

    def speak_stream(self, text: str, pcm_queue: PcmQueue) -> StreamSpeakTiming:
        if not self._ready:
            raise TtsError("Text-to-speech is not initialized")

        cleaned = " ".join(text.split())
        if not cleaned:
            raise TtsError("Cannot synthesize empty text")

        timing = StreamSpeakTiming()
        started = time.perf_counter()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    self._stream_pcm(cleaned, pcm_queue, timing, started),
                    timeout=TTS_DEFAULT_TIMEOUT_SECONDS,
                )
            )
        except TtsError:
            raise
        except TimeoutError as error:
            raise TtsError(f"TTS timed out after {TTS_DEFAULT_TIMEOUT_SECONDS:.0f}s ({len(cleaned)} chars)") from error
        except (miniaudio.DecodeError, miniaudio.MiniaudioError) as error:
            raise TtsError(f"Failed to decode TTS audio: {error}") from error
        except _TTS_ERRORS as error:
            raise TtsError(f"Failed to synthesize speech: {error}") from error
        finally:
            loop.close()

        timing.tts_ms = (time.perf_counter() - started) * 1000.0
        if timing.first_pcm_ms <= 0.0:
            raise TtsError("edge-tts returned no audio")

        return timing

    async def _stream_pcm(
        self,
        text: str,
        pcm_queue: PcmQueue,
        timing: StreamSpeakTiming,
        started: float,
    ) -> None:
        communicate = edge_tts.Communicate(
            text,
            voice=self._config.voice,
            rate=self._config.rate,
        )
        mp3 = bytearray()
        emitted = 0
        last_decode_at = 0

        async for item in communicate.stream():
            payload = _as_mapping(item)
            if payload.get("type") != "audio":
                continue

            data = payload.get("data")
            if not isinstance(data, (bytes, bytearray)):
                continue

            mp3.extend(data)
            if len(mp3) < TTS_STREAM_START_BYTES:
                continue
            if len(mp3) - last_decode_at < TTS_STREAM_DECODE_STEP_BYTES:
                continue

            emitted = self._emit_decoded(mp3, emitted, pcm_queue, timing, started)
            last_decode_at = len(mp3)

        if not mp3:
            return

        self._emit_decoded(mp3, emitted, pcm_queue, timing, started)

    def _emit_decoded(
        self,
        mp3: bytearray,
        emitted: int,
        pcm_queue: PcmQueue,
        timing: StreamSpeakTiming,
        started: float,
    ) -> int:
        try:
            samples = self._decode_mp3(bytes(mp3))
        except (miniaudio.DecodeError, miniaudio.MiniaudioError):
            return emitted

        if samples.size <= emitted:
            return emitted

        chunk = np.ascontiguousarray(samples[emitted:])
        pcm_queue.put(chunk)
        if timing.first_pcm_ms <= 0.0:
            timing.first_pcm_ms = (time.perf_counter() - started) * 1000.0
        return samples.size

    def _decode_mp3(self, data: bytes) -> NDArray[np.float32]:
        decoded = miniaudio.decode(
            data,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=self._config.sample_rate,
        )
        return np.frombuffer(decoded.samples, dtype=np.float32).copy()


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TtsError(f"Unexpected edge-tts payload type: {type(value)!r}")

    return value
