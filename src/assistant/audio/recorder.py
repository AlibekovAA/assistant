from __future__ import annotations

import contextlib
import queue
import time

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from assistant.audio.exceptions import AudioRecordingError
from assistant.audio.models import AudioChunk, AudioData, AudioFormat
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

    def read(self, *, timeout: float | None = 0.1) -> AudioChunk | None:
        if self._format is None:
            raise AudioRecordingError("Recording stream is not started")

        try:
            samples = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        return AudioChunk(samples=samples, format=self._format)

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

    def record(
        self,
        duration: float,
        audio_format: AudioFormat,
        *,
        device: int | None = None,
        blocksize: int = 1024,
    ) -> AudioData:
        audio_format.validate()

        if duration <= 0:
            raise AudioRecordingError(f"Invalid duration: {duration}")

        chunks: list[NDArray[np.float32]] = []
        target_frames = int(duration * audio_format.sample_rate)

        try:
            self.start(audio_format, device=device, blocksize=blocksize)
            deadline = time.monotonic() + duration

            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                chunk = self.read(timeout=min(0.1, max(remaining, 0.0)))

                if chunk is not None:
                    chunks.append(chunk.samples)
        finally:
            self.stop()

        if not chunks:
            raise AudioRecordingError(
                "No audio captured "
                f"(device={device}, sample_rate={audio_format.sample_rate}, "
                f"channels={audio_format.channels}, duration={duration})"
            )

        samples = np.concatenate(chunks, axis=0)

        if samples.shape[0] > target_frames:
            samples = samples[:target_frames]

        return AudioData(samples=samples, format=audio_format)

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
