from __future__ import annotations

import contextlib
import queue

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from assistant.audio.exceptions import AudioRecordingError
from assistant.audio.models import AudioData, AudioFormat
from assistant.logger import Logger


class AudioRecorder:
    def __init__(self, *, queue_size: int = 64) -> None:
        self._logger = Logger.get(__name__)
        self._queue: queue.Queue[NDArray[np.float32]] = queue.Queue(maxsize=queue_size)
        self._stream: sd.InputStream | None = None
        self._format: AudioFormat | None = None

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def format(self) -> AudioFormat | None:
        return self._format

    def start(
        self,
        audio_format: AudioFormat,
        *,
        device: int | None = None,
        blocksize: int = 1024,
    ) -> None:
        audio_format.validate()

        if blocksize < 0:
            raise AudioRecordingError(f"Invalid blocksize: {blocksize}")

        if self.is_active:
            raise AudioRecordingError("Recording stream is already active")

        self._clear_queue()
        self._format = audio_format

        try:
            self._stream = sd.InputStream(
                samplerate=audio_format.sample_rate,
                channels=audio_format.channels,
                device=device,
                dtype="float32",
                blocksize=blocksize,
                callback=self._on_audio,
            )
            self._stream.start()
        except sd.PortAudioError as error:
            self._stream = None
            self._format = None
            raise AudioRecordingError(
                "Failed to start recording stream "
                f"(device={device}, sample_rate={audio_format.sample_rate}, "
                f"channels={audio_format.channels}, blocksize={blocksize}): {error}"
            ) from error

    def read(self, *, timeout: float | None = 0.1) -> AudioData | None:
        if self._format is None:
            raise AudioRecordingError("Recording stream is not started")

        try:
            samples = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        return AudioData(samples=samples, format=self._format)

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._format = None

        if stream is None:
            return

        try:
            if stream.active:
                stream.abort()
            stream.close()
        except sd.PortAudioError as error:
            raise AudioRecordingError(f"Failed to stop recording stream: {error}") from error
        finally:
            self._clear_queue()

    def _on_audio(
        self,
        indata: NDArray[np.float32],
        _frames: int,
        _time: object,
        status: object,
    ) -> None:
        if status:
            self._logger.warning("Input stream status: %s", status)

        samples = indata.copy()

        try:
            self._queue.put_nowait(samples)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()

            try:
                self._queue.put_nowait(samples)
            except queue.Full:
                self._logger.warning("Dropping audio chunk: capture queue is full")

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
