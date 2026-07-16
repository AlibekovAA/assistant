from __future__ import annotations

import threading

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from assistant.audio.exceptions import AudioPlaybackError
from assistant.audio.models import AudioData
from assistant.logger import Logger


class AudioPlayer:
    def __init__(self) -> None:
        self._logger = Logger.get(__name__)
        self._stream: sd.OutputStream | None = None
        self._buffer: NDArray[np.float32] | None = None
        self._offset = 0
        self._done = threading.Event()

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    def play(
        self,
        audio: AudioData,
        *,
        device: int | None = None,
    ) -> None:
        audio.format.validate()

        if audio.samples.size == 0:
            raise AudioPlaybackError("Cannot play empty audio")

        self.stop()

        samples = self._prepare_output(audio)
        self._buffer = samples
        self._offset = 0
        self._done.clear()

        try:
            self._stream = sd.OutputStream(
                samplerate=audio.format.sample_rate,
                channels=audio.format.channels,
                device=device,
                dtype="float32",
                callback=self._on_audio,
                finished_callback=self._done.set,
            )
            self._stream.start()
        except sd.PortAudioError as error:
            self._reset_state()
            raise AudioPlaybackError(
                "Failed to play audio "
                f"(device={device}, sample_rate={audio.format.sample_rate}, "
                f"channels={audio.format.channels}, frames={samples.shape[0]}): {error}"
            ) from error

        self._wait()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._buffer = None
        self._offset = 0
        self._done.set()

        if stream is None:
            return

        try:
            if stream.active:
                stream.abort()
            stream.close()
        except sd.PortAudioError as error:
            raise AudioPlaybackError(f"Failed to stop playback: {error}") from error

    def _wait(self) -> None:
        self._done.wait()

        stream = self._stream
        self._stream = None
        self._buffer = None
        self._offset = 0

        if stream is None:
            return

        try:
            if stream.active:
                stream.stop()
            stream.close()
        except sd.PortAudioError as error:
            raise AudioPlaybackError(f"Failed while waiting for playback: {error}") from error

    def _on_audio(
        self,
        outdata: NDArray[np.float32],
        frames: int,
        _time: object,
        status: object,
    ) -> None:
        if status:
            self._logger.warning("Output stream status: %s", status)

        if self._buffer is None:
            outdata.fill(0)
            raise sd.CallbackStop

        remaining = self._buffer.shape[0] - self._offset
        if remaining <= 0:
            outdata.fill(0)
            raise sd.CallbackStop

        chunk_size = min(frames, remaining)
        outdata[:chunk_size] = self._buffer[self._offset : self._offset + chunk_size]

        if chunk_size < frames:
            outdata[chunk_size:].fill(0)
            self._offset += chunk_size
            raise sd.CallbackStop

        self._offset += chunk_size

    def _prepare_output(self, audio: AudioData) -> NDArray[np.float32]:
        samples = np.asarray(audio.samples, dtype=np.float32)
        channels = audio.format.channels

        if samples.ndim == 1:
            if channels != 1:
                raise AudioPlaybackError(f"Mono buffer cannot be played with channels={channels}")
            return np.ascontiguousarray(samples.reshape(-1, 1))

        if samples.ndim == 2:
            if samples.shape[1] != channels:
                raise AudioPlaybackError(
                    f"Audio channel mismatch: buffer has {samples.shape[1]}, format has {channels}"
                )
            return np.ascontiguousarray(samples)

        raise AudioPlaybackError(f"Unsupported audio shape for playback: {samples.shape}")

    def _reset_state(self) -> None:
        self._stream = None
        self._buffer = None
        self._offset = 0
        self._done.set()
