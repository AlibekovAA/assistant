from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str
    segments: tuple[TranscriptSegment, ...]
    duration: float
