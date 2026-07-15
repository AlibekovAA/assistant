from __future__ import annotations

from assistant.core.exceptions import AssistantError


class TtsError(AssistantError):
    pass


class TtsNotReadyError(TtsError):
    pass
