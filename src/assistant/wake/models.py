from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WakeDetection:
    keyword: str
    index: int
