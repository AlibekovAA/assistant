from __future__ import annotations

import logging
import os
from typing import Final
import warnings


class Logger:
    _configured: bool = False
    _format: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _noisy_loggers: Final[tuple[str, ...]] = (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "urllib3",
        "filelock",
        "faster_whisper",
        "ctranslate2",
        "aiohttp",
        "asyncio",
        "edge_tts",
    )

    @classmethod
    def configure(cls, level: int = logging.INFO) -> None:
        if cls._configured:
            return

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"huggingface_hub(\..*)?")

        logging.basicConfig(
            level=level,
            format=cls._format,
        )

        for name in cls._noisy_loggers:
            logging.getLogger(name).setLevel(logging.ERROR)

        cls._configured = True

    @staticmethod
    def get(name: str) -> logging.Logger:
        return logging.getLogger(name)
