from __future__ import annotations

from assistant.audio.devices import AudioDevice, AudioDeviceCatalog
from assistant.audio.dsp import rms, to_mono, trim_silence
from assistant.audio.exceptions import (
    AudioDeviceError,
    AudioError,
    AudioPlaybackError,
    AudioRecordingError,
)
from assistant.audio.manager import AudioManager
from assistant.audio.models import AudioChunk, AudioData, AudioFormat
from assistant.audio.player import AudioPlayer
from assistant.audio.protocol import AudioCapture, AudioPlayback
from assistant.audio.recorder import AudioRecorder

__all__ = [
    "AudioCapture",
    "AudioChunk",
    "AudioData",
    "AudioDevice",
    "AudioDeviceCatalog",
    "AudioDeviceError",
    "AudioError",
    "AudioFormat",
    "AudioManager",
    "AudioPlayback",
    "AudioPlaybackError",
    "AudioPlayer",
    "AudioRecorder",
    "AudioRecordingError",
    "rms",
    "to_mono",
    "trim_silence",
]
