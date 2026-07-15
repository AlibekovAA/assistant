from __future__ import annotations

from assistant.core.exceptions import AssistantError


class WakeError(AssistantError):
    pass


class WakeNotReadyError(WakeError):
    pass
