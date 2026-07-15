from __future__ import annotations

import threading
import time

import numpy as np
from numpy.typing import NDArray

from assistant.audio import AudioData, AudioFormat, AudioManager, rms, to_mono, trim_silence
from assistant.config import WakeConfig
from assistant.logger import Logger
from assistant.stt import SpeechToText
from assistant.tts import TextToSpeech
from assistant.wake import WakeDetection, WakeWordDetector


class VoicePipeline:
    def __init__(
        self,
        *,
        audio: AudioManager,
        stt: SpeechToText,
        tts: TextToSpeech,
        wake: WakeWordDetector,
        wake_config: WakeConfig,
        assistant_name: str,
    ) -> None:
        self._audio = audio
        self._stt = stt
        self._tts = tts
        self._wake = wake
        self._wake_config = wake_config
        self._assistant_name = assistant_name
        self._logger = Logger.get(__name__)

    def run(self, stop_event: threading.Event) -> None:
        audio_format = self._audio.format
        self._audio.start_capture()
        self._logger.info("Listening for wake word %r...", self._wake_config.keyword)

        try:
            while not stop_event.is_set():
                chunk = self._audio.read_chunk(timeout=0.1)
                if chunk is None:
                    continue

                detection = self._wake.feed(AudioData(samples=chunk.samples, format=audio_format))
                if stop_event.is_set():
                    break

                if detection is None:
                    continue

                self._handle_detection(detection, audio_format, stop_event)
                if stop_event.is_set():
                    break

                self._wake.reset()
                self._logger.info("Listening for wake word %r...", self._wake_config.keyword)
        finally:
            self._audio.stop_capture()
            self._wake.reset()

    def _handle_detection(
        self,
        detection: WakeDetection,
        audio_format: AudioFormat,
        stop_event: threading.Event,
    ) -> None:
        self._logger.info("Wake word detected: %r", detection.keyword)
        self._prune_post_wake(stop_event)
        if stop_event.is_set():
            return

        utterance = self._capture_utterance(audio_format, stop_event)
        if stop_event.is_set():
            return

        if utterance is None or utterance.samples.size == 0:
            self._logger.info("No speech captured after wake word")
            self._speak("Я вас не расслышала.", stop_event)
            return

        transcript = self._stt.transcribe(
            utterance,
            vad_filter=True,
            beam_size=8,
            temperature=0.0,
            no_speech_threshold=0.5,
        )
        if stop_event.is_set():
            return

        if not transcript.text:
            self._logger.info("Empty transcript")
            self._speak("Я вас не расслышала.", stop_event)
            return

        self._logger.info("Heard: %s", transcript.text)
        reply = f"Привет, я голосовой помощник {self._assistant_name}. Вы сказали: {transcript.text}"
        self._speak(reply, stop_event)

    def _speak(self, text: str, stop_event: threading.Event) -> None:
        self._logger.info("Reply: %s", text)
        speech = self._tts.synthesize(text)
        if stop_event.is_set():
            return
        self._audio.play(speech, blocking=True)

    def _prune_post_wake(self, stop_event: threading.Event) -> None:
        deadline = time.monotonic() + self._wake_config.post_wake_prune_seconds
        while not stop_event.is_set() and time.monotonic() < deadline:
            self._audio.read_chunk(timeout=0.05)

    def _capture_utterance(
        self,
        audio_format: AudioFormat,
        stop_event: threading.Event,
    ) -> AudioData | None:
        chunks: list[NDArray[np.float32]] = []
        speech_started = False
        speech_started_at: float | None = None
        silence_started_at: float | None = None
        loud_started_at: float | None = None
        started_at = time.monotonic()
        threshold = self._wake_config.speech_rms_threshold

        while not stop_event.is_set():
            if time.monotonic() - started_at >= self._wake_config.utterance_max_seconds:
                break

            chunk = self._audio.read_chunk(timeout=0.1)
            if chunk is None:
                continue

            samples = to_mono(chunk.samples)
            chunks.append(samples)
            level = rms(samples)

            if level >= threshold:
                if loud_started_at is None:
                    loud_started_at = time.monotonic()

                if not speech_started and time.monotonic() - loud_started_at >= self._wake_config.speech_onset_seconds:
                    speech_started = True
                    speech_started_at = loud_started_at
                    self._logger.debug("Speech onset detected")

                silence_started_at = None
                continue

            loud_started_at = None

            if not speech_started:
                continue

            if silence_started_at is None:
                silence_started_at = time.monotonic()
                continue

            spoken = 0.0 if speech_started_at is None else time.monotonic() - speech_started_at
            silent_for = time.monotonic() - silence_started_at

            if spoken >= self._wake_config.min_speech_seconds and silent_for >= self._wake_config.silence_seconds:
                break

        if stop_event.is_set() or not chunks or not speech_started:
            return None

        raw = np.concatenate(chunks, axis=0)
        trimmed = trim_silence(
            raw,
            threshold=threshold * 0.7,
            sample_rate=audio_format.sample_rate,
            pad_seconds=0.25,
        )
        duration = float(trimmed.shape[0] / audio_format.sample_rate)
        self._logger.info("Captured utterance: %.2fs", duration)

        if trimmed.size == 0:
            return None

        return AudioData(
            samples=trimmed,
            format=AudioFormat(sample_rate=audio_format.sample_rate, channels=1),
        )
