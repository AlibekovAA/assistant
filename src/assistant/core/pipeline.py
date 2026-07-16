from __future__ import annotations

import threading
import time

from assistant.audio.manager import AudioManager
from assistant.audio.models import AudioFormat
from assistant.audio.utterance import UtteranceCapture
from assistant.config import SttConfig, UtteranceConfig, WakeConfig
from assistant.logger import Logger
from assistant.stt import SpeechToText, TranscribeOptions
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
        utterance_config: UtteranceConfig,
        stt_config: SttConfig,
        assistant_name: str,
    ) -> None:
        self._audio = audio
        self._stt = stt
        self._tts = tts
        self._wake = wake
        self._wake_config = wake_config
        self._assistant_name = assistant_name
        self._utterance = UtteranceCapture(utterance_config)
        self._command_options = TranscribeOptions(
            vad_filter=stt_config.vad_filter,
            beam_size=stt_config.beam_size,
            temperature=stt_config.temperature,
            no_speech_threshold=stt_config.no_speech_threshold,
        )
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

                detection = self._wake.feed(chunk)
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

        utterance = self._utterance.capture(
            audio_format=audio_format,
            read_audio=self._audio.read_chunk,
            stop_event=stop_event,
        )
        if stop_event.is_set():
            return

        if utterance is None or utterance.samples.size == 0:
            self._logger.info("No speech captured after wake word")
            self._speak("Я вас не расслышала.", stop_event)
            return

        transcript = self._stt.transcribe(utterance, self._command_options)
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
        self._audio.play(speech)

    def _prune_post_wake(self, stop_event: threading.Event) -> None:
        deadline = time.monotonic() + self._wake_config.post_wake_prune_seconds
        while not stop_event.is_set() and time.monotonic() < deadline:
            self._audio.read_chunk(timeout=0.05)
