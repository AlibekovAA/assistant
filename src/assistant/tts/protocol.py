from __future__ import annotations

from typing import Protocol

from assistant.audio.models import AudioData


class TextToSpeech(Protocol):
    @property
    def is_ready(self) -> bool: ...

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def synthesize(self, text: str) -> AudioData: ...
