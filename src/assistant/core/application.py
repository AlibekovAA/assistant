from __future__ import annotations

from assistant.config import load_config
from assistant.logger import Logger


class Application:
    def __init__(self) -> None:
        self._config = load_config()

    def run(self) -> None:
        Logger.configure()

        logger = Logger.get(__name__)
        logger.info("%s started", self._config.app_name)
