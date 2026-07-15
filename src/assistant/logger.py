from __future__ import annotations

import logging
from typing import Final


class Logger:
    _configured: bool = False
    _format: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    @classmethod
    def configure(cls, level: int = logging.INFO) -> None:
        if cls._configured:
            return

        logging.basicConfig(
            level=level,
            format=cls._format,
        )

        cls._configured = True

    @staticmethod
    def get(name: str) -> logging.Logger:
        return logging.getLogger(name)
