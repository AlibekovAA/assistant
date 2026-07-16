from __future__ import annotations

import signal
import threading
import types

from assistant.audio.manager import AudioManager
from assistant.config import load_config
from assistant.core.exceptions import AssistantError
from assistant.core.pipeline import VoicePipeline
from assistant.logger import Logger
from assistant.stt import SpeechToText, WhisperStt
from assistant.tts import EdgeTts, TextToSpeech
from assistant.wake import WakeWordDetector, WhisperWakeWord


class Application:
    def __init__(self) -> None:
        Logger.configure()

        self._config = load_config()
        self._logger = Logger.get(__name__)
        self._stop_event = threading.Event()
        self._interrupt_count = 0
        self._audio = AudioManager(self._config.audio)
        self._stt: SpeechToText = WhisperStt(self._config.stt)
        self._tts: TextToSpeech = EdgeTts(self._config.tts)
        self._wake: WakeWordDetector = WhisperWakeWord(
            self._config.wake,
            self._stt,
            stop_event=self._stop_event,
        )
        self._pipeline = VoicePipeline(
            audio=self._audio,
            stt=self._stt,
            tts=self._tts,
            wake=self._wake,
            wake_config=self._config.wake,
            utterance_config=self._config.utterance,
            stt_config=self._config.stt,
            assistant_name=self._config.app_name,
        )

    def run(self) -> None:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            self._initialize()
            self._start()
        except AssistantError:
            self._logger.exception("Application terminated due to an error")
            raise
        except KeyboardInterrupt:
            self._stop_event.set()
            self._logger.info("Application interrupted")
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            self._shutdown()

    def _handle_signal(self, _signum: int, _frame: types.FrameType | None) -> None:
        self._interrupt_count += 1
        self._stop_event.set()

        if self._interrupt_count == 1:
            self._logger.info("Stop requested")
            raise KeyboardInterrupt

        self._logger.warning("Force exit")
        raise SystemExit(130)

    def _initialize(self) -> None:
        self._logger.info("Initializing application")
        self._audio.initialize()
        self._stt.initialize()
        self._tts.initialize()
        self._wake.initialize()

    def _start(self) -> None:
        self._logger.info("%s v%s started", self._config.app_name, self._config.app_version)
        self._log_devices()
        self._logger.info(
            "STT ready (model=%s, language=%s)",
            self._config.stt.model,
            self._config.stt.language,
        )
        self._logger.info("Press Ctrl+C to stop")
        self._pipeline.run(self._stop_event)

    def _shutdown(self) -> None:
        self._stop_event.set()
        self._audio.stop_playback()
        self._audio.stop_capture()
        self._audio.shutdown()
        self._wake.shutdown()
        self._tts.shutdown()
        self._stt.shutdown()
        self._logger.info("Application stopped")

    def _log_devices(self) -> None:
        input_device = self._audio.get_default_input_device()
        output_device = self._audio.get_default_output_device()

        if input_device is not None:
            self._logger.info(
                "Default input: [%d] %s (%s)",
                input_device.index,
                input_device.name,
                input_device.hostapi_name,
            )

        if output_device is not None:
            self._logger.info(
                "Default output: [%d] %s (%s)",
                output_device.index,
                output_device.name,
                output_device.hostapi_name,
            )
