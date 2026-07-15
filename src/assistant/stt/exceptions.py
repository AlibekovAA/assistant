from __future__ import annotations

from assistant.core.exceptions import AssistantError


class SttError(AssistantError):
    pass


class SttNotReadyError(SttError):
    pass
