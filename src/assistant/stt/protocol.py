from __future__ import annotations

from typing import Protocol

from assistant.audio.models import AudioData
from assistant.stt.models import Transcript


class SpeechToText(Protocol):
    @property
    def is_ready(self) -> bool: ...

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def transcribe(
        self,
        audio: AudioData,
        *,
        vad_filter: bool | None = None,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        no_speech_threshold: float | None = None,
        temperature: float | None = None,
    ) -> Transcript: ...
