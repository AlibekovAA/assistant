from collections.abc import Mapping
import json

from gigachat import GigaChat
from gigachat.exceptions import GigaChatException
from gigachat.models import Chat, Messages, MessagesRole

from assistant.config import GigaChatConfig
from assistant.constants.llm import GIGACHAT_MAX_TOOL_ROUNDS
from assistant.core.exceptions import BrainError
from assistant.logger import Logger
from assistant.prompts import (
    NOT_HEARD,
    REPLY_FALLBACK,
    SHUTDOWN_FAREWELL,
    SYSTEM_PROMPT,
    TOOL_ROUNDS_EXCEEDED,
)
from assistant.tools.registry import ToolRegistry

_GIGACHAT_ERRORS = (GigaChatException, OSError, TimeoutError, ValueError, TypeError, RuntimeError)
_LOG = Logger.get(__name__)


class GigaChatBrain:
    def __init__(self, config: GigaChatConfig, tools: ToolRegistry) -> None:
        self._config = config
        self._tools = tools
        self._client: GigaChat | None = None
        self._shutdown_requested = False

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def initialize(self) -> None:
        if self._client is not None:
            return

        _LOG.info(
            "Initializing GigaChat (model=%s, scope=%s)",
            self._config.model,
            self._config.scope,
        )
        client = GigaChat(
            credentials=self._config.credentials,
            scope=self._config.scope,
            model=self._config.model,
            verify_ssl_certs=self._config.verify_ssl_certs,
            timeout=self._config.timeout_seconds,
        )
        try:
            client.get_models()
        except _GIGACHAT_ERRORS as error:
            client.close()
            raise BrainError(f"Failed to connect to GigaChat: {error}") from error

        self._client = client
        _LOG.info("GigaChat ready")

    def shutdown(self) -> None:
        if self._client is None:
            return
        self._client.close()
        self._client = None

    def reply(self, user_text: str) -> str:
        if self._client is None:
            raise BrainError("GigaChat is not initialized")

        text = user_text.strip()
        if not text:
            return NOT_HEARD

        self._shutdown_requested = False
        messages: list[Messages] = [
            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
            Messages(role=MessagesRole.USER, content=text),
        ]
        functions = self._tools.specifications

        try:
            for _ in range(GIGACHAT_MAX_TOOL_ROUNDS):
                response = self._client.chat(
                    Chat(
                        model=self._config.model,
                        messages=messages,
                        functions=functions,
                        function_call="auto",
                        temperature=self._config.temperature,
                        max_tokens=self._config.max_tokens,
                    )
                )
                choice = response.choices[0]
                message = choice.message
                messages.append(message)

                if choice.finish_reason != "function_call" or message.function_call is None:
                    content = (message.content or "").strip()
                    return content or REPLY_FALLBACK

                function_name = message.function_call.name
                arguments = _function_arguments(message.function_call.arguments)
                _LOG.info("Tool call: %s(%s)", function_name, arguments)
                result = self._tools.execute(function_name, arguments)
                _LOG.debug("Tool result: %s", result)
                messages.append(
                    Messages(
                        role=MessagesRole.FUNCTION,
                        content=json.dumps(result, ensure_ascii=False),
                        name=function_name,
                    )
                )
                if result.get("shutdown") is True:
                    self._shutdown_requested = True
                    return SHUTDOWN_FAREWELL
        except _GIGACHAT_ERRORS as error:
            raise BrainError(f"GigaChat request failed: {error}") from error

        return TOOL_ROUNDS_EXCEEDED


def _function_arguments(arguments: object) -> dict[str, object]:
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return {str(key): value for key, value in arguments.items()}
    if isinstance(arguments, str):
        try:
            parsed: object = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise BrainError(f"Invalid function arguments JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise BrainError("Function arguments must be a JSON object")
        return {str(key): value for key, value in parsed.items()}
    raise BrainError(f"Unsupported function arguments type: {type(arguments).__name__}")
