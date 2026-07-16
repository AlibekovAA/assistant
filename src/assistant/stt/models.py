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


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    vad_filter: bool | None = None
    beam_size: int | None = None
    temperature: float | None = None
    no_speech_threshold: float | None = None
    initial_prompt: str | None = None
    hotwords: str | None = None
