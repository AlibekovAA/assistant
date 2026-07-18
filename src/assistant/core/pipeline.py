from dataclasses import dataclass
import queue
import threading
import time

from assistant.audio.manager import AudioManager
from assistant.audio.models import AudioFormat
from assistant.audio.player import PcmQueue
from assistant.audio.utterance import UtteranceCapture
from assistant.brain.gigachat import GigaChatBrain
from assistant.config import SttConfig, UtteranceConfig, WakeConfig
from assistant.constants.audio import (
    AUDIO_DEFAULT_READ_TIMEOUT_SECONDS,
    AUDIO_TTS_PRODUCER_JOIN_TIMEOUT_SECONDS,
)
from assistant.constants.speech import SPEECH_POST_WAKE_READ_TIMEOUT_SECONDS
from assistant.core.exceptions import BrainError, TtsError
from assistant.logger import Logger
from assistant.overlay.window import Live2dAvatarOverlay
from assistant.prompts import BRAIN_FAILURE, NOT_HEARD
from assistant.stt.models import TranscribeOptions
from assistant.stt.whisper import WhisperStt
from assistant.tts.edge import EdgeTts
from assistant.wake.models import WakeDetection
from assistant.wake.whisper import WhisperWakeWord

_LOG = Logger.get(__name__)


@dataclass(slots=True)
class _SpeakProducerState:
    first_pcm_ms: float = 0.0
    tts_ms: float = 0.0
    error: BaseException | None = None


class VoicePipeline:
    def __init__(
        self,
        *,
        audio: AudioManager,
        stt: WhisperStt,
        tts: EdgeTts,
        wake: WhisperWakeWord,
        brain: GigaChatBrain,
        overlay: Live2dAvatarOverlay,
        wake_config: WakeConfig,
        utterance_config: UtteranceConfig,
        stt_config: SttConfig,
    ) -> None:
        self._audio = audio
        self._stt = stt
        self._tts = tts
        self._wake = wake
        self._brain = brain
        self._overlay = overlay
        self._wake_config = wake_config
        self._utterance = UtteranceCapture(utterance_config)
        self._command_options = TranscribeOptions(
            vad_filter=stt_config.vad_filter,
            beam_size=stt_config.beam_size,
            temperature=stt_config.temperature,
            no_speech_threshold=stt_config.no_speech_threshold,
        )

    def run(self, stop_event: threading.Event) -> None:
        audio_format = self._audio.format
        self._audio.start_capture()
        self._log_listening()

        try:
            while not stop_event.is_set():
                chunk = self._audio.read_chunk(timeout=AUDIO_DEFAULT_READ_TIMEOUT_SECONDS)
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
                self._log_listening()
        finally:
            self._overlay.hide()
            self._audio.stop_capture()
            self._wake.reset()

    def _log_listening(self) -> None:
        _LOG.info("Waiting for wake word %r", self._wake_config.keyword)

    def _handle_detection(
        self,
        detection: WakeDetection,
        audio_format: AudioFormat,
        stop_event: threading.Event,
    ) -> None:
        _LOG.info("Wake word detected: %r", detection.keyword)
        self._overlay.show()
        stt_ms = 0.0
        brain_ms = 0.0
        tts_first_ms = 0.0
        tts_ms = 0.0
        play_ms = 0.0
        try:
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
                _LOG.warning("No speech captured after wake word")
                tts_first_ms, tts_ms, play_ms = self._speak(NOT_HEARD, stop_event)
                return

            started = time.perf_counter()
            transcript = self._stt.transcribe(utterance, self._command_options)
            stt_ms = (time.perf_counter() - started) * 1000.0
            if stop_event.is_set():
                return

            if not transcript.text:
                _LOG.warning("Empty transcript")
                tts_first_ms, tts_ms, play_ms = self._speak(NOT_HEARD, stop_event)
                return

            _LOG.info("Heard: %s", transcript.text)
            started = time.perf_counter()
            try:
                reply = self._brain.reply(transcript.text)
            except BrainError:
                _LOG.exception("Brain request failed")
                reply = BRAIN_FAILURE
            brain_ms = (time.perf_counter() - started) * 1000.0
            tts_first_ms, tts_ms, play_ms = self._speak(reply, stop_event)
            if self._brain.shutdown_requested:
                _LOG.warning("Shutdown requested by assistant")
                stop_event.set()
        finally:
            self._overlay.hide()
            _LOG.info(
                "Timing: stt_ms=%.0f brain_ms=%.0f tts_first_ms=%.0f tts_ms=%.0f play_ms=%.0f",
                stt_ms,
                brain_ms,
                tts_first_ms,
                tts_ms,
                play_ms,
            )

    def _speak(self, text: str, stop_event: threading.Event) -> tuple[float, float, float]:
        _LOG.info("Reply: %s", text)
        if stop_event.is_set():
            return 0.0, 0.0, 0.0

        pcm_queue: PcmQueue = queue.SimpleQueue()
        state = _SpeakProducerState()

        def producer() -> None:
            try:
                timing = self._tts.speak_stream(text, pcm_queue)
                state.first_pcm_ms = timing.first_pcm_ms
                state.tts_ms = timing.tts_ms
            except BaseException as exc:
                state.error = exc
            finally:
                pcm_queue.put(None)

        producer_thread = threading.Thread(target=producer, name="tts-stream", daemon=True)
        play_started = time.perf_counter()
        producer_thread.start()
        try:
            self._audio.play_stream(
                pcm_queue,
                sample_rate=self._tts.sample_rate,
                on_level=self._overlay.set_level,
            )
        finally:
            producer_thread.join(timeout=AUDIO_TTS_PRODUCER_JOIN_TIMEOUT_SECONDS)

        play_ms = (time.perf_counter() - play_started) * 1000.0
        if state.error is not None:
            if isinstance(state.error, TtsError):
                _LOG.warning("Speech synthesis failed: %s", state.error)
                return 0.0, 0.0, play_ms
            raise state.error

        return state.first_pcm_ms, state.tts_ms, play_ms

    def _prune_post_wake(self, stop_event: threading.Event) -> None:
        deadline = time.monotonic() + self._wake_config.post_wake_prune_seconds
        while not stop_event.is_set() and time.monotonic() < deadline:
            self._audio.read_chunk(timeout=SPEECH_POST_WAKE_READ_TIMEOUT_SECONDS)
