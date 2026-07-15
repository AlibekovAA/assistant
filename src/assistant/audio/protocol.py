from __future__ import annotations

from typing import Protocol

from assistant.audio.models import AudioChunk, AudioData, AudioFormat


class AudioCapture(Protocol):
    @property
    def is_active(self) -> bool: ...

    def start(
        self,
        audio_format: AudioFormat,
        *,
        device: int | None = None,
        blocksize: int = 1024,
    ) -> None: ...

    def read(self, *, timeout: float | None = 0.1) -> AudioChunk | None: ...

    def stop(self) -> None: ...

    def record(
        self,
        duration: float,
        audio_format: AudioFormat,
        *,
        device: int | None = None,
        blocksize: int = 1024,
    ) -> AudioData: ...


class AudioPlayback(Protocol):
    @property
    def is_active(self) -> bool: ...

    def play(
        self,
        audio: AudioData,
        *,
        device: int | None = None,
        blocking: bool = True,
    ) -> None: ...

    def stop(self) -> None: ...

    def wait(self) -> None: ...
