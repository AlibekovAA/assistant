from __future__ import annotations

from typing import Protocol

from assistant.audio.models import AudioData
from assistant.wake.models import WakeDetection


class WakeWordDetector(Protocol):
    @property
    def is_ready(self) -> bool: ...

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def reset(self) -> None: ...

    def feed(self, audio: AudioData) -> WakeDetection | None: ...
