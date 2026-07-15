from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    app_name: str = "Assistant"
    app_version: str = "0.1.0"


def load_config() -> Config:
    return Config()
