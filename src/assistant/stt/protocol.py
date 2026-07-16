from __future__ import annotations

from typing import Protocol

from assistant.audio.models import AudioData
from assistant.stt.models import TranscribeOptions, Transcript


class SpeechToText(Protocol):
    @property
    def is_ready(self) -> bool: ...

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def transcribe(self, audio: AudioData, options: TranscribeOptions | None = None) -> Transcript: ...
