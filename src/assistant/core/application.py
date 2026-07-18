import signal
import threading
import types

from assistant.audio.manager import AudioManager
from assistant.brain.gigachat import GigaChatBrain
from assistant.config import load_config
from assistant.constants.audio import AUDIO_PIPELINE_JOIN_TIMEOUT_SECONDS
from assistant.core.exceptions import AssistantError
from assistant.core.pipeline import VoicePipeline
from assistant.logger import Logger, prepare_runtime_env
from assistant.overlay.window import Live2dAvatarOverlay
from assistant.stt.whisper import WhisperStt
from assistant.tools.registry import ToolRegistry
from assistant.tts.edge import EdgeTts
from assistant.wake.whisper import WhisperWakeWord

_LOG = Logger.get(__name__)


class Application:
    def __init__(self) -> None:
        prepare_runtime_env()
        Logger.configure()

        config = load_config()
        stop_event = threading.Event()
        audio = AudioManager(config.audio)
        stt = WhisperStt(config.stt)
        tts = EdgeTts(config.tts)
        wake = WhisperWakeWord(config.wake, stt, stop_event=stop_event)
        brain = GigaChatBrain(
            config.gigachat,
            ToolRegistry.default(
                default_city=config.tools.default_city,
                default_timezone=config.tools.default_timezone,
            ),
        )
        overlay = Live2dAvatarOverlay()

        self._config = config
        self._stop_event = stop_event
        self._interrupt_count = 0
        self._audio = audio
        self._stt = stt
        self._tts = tts
        self._wake = wake
        self._brain = brain
        self._overlay = overlay
        self._pipeline = VoicePipeline(
            audio=audio,
            stt=stt,
            tts=tts,
            wake=wake,
            brain=brain,
            overlay=overlay,
            wake_config=config.wake,
            utterance_config=config.utterance,
            stt_config=config.stt,
        )

    def run(self) -> None:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)
        pipeline_thread: threading.Thread | None = None

        try:
            self._initialize()
            pipeline_thread = threading.Thread(
                target=self._run_pipeline,
                name="voice-pipeline",
                daemon=False,
            )
            pipeline_thread.start()
            self._overlay.run()
        except AssistantError:
            _LOG.exception("Application terminated due to an error")
            raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            self._stop_event.set()
            self._overlay.shutdown()
            if pipeline_thread is not None:
                pipeline_thread.join(timeout=AUDIO_PIPELINE_JOIN_TIMEOUT_SECONDS)
                if pipeline_thread.is_alive():
                    _LOG.warning("Voice pipeline thread is still alive")
            self._shutdown()

    def _handle_signal(self, _signum: int, _frame: types.FrameType | None) -> None:
        self._interrupt_count += 1
        self._stop_event.set()
        self._overlay.shutdown()

        if self._interrupt_count == 1:
            _LOG.warning("Stop requested")
            return

        _LOG.error("Force exit")
        raise SystemExit(130)

    def _initialize(self) -> None:
        _LOG.info("Initializing application")
        self._audio.initialize()
        self._stt.initialize()
        self._tts.initialize()
        self._wake.initialize()
        self._brain.initialize()
        self._overlay.initialize()

    def _run_pipeline(self) -> None:
        if not self._overlay.wait_until_ready():
            _LOG.error("Avatar overlay failed to start")
            self._stop_event.set()
            self._overlay.shutdown()
            return

        try:
            _LOG.info("%s v%s started", self._config.app_name, self._config.app_version)
            _LOG.info(
                "STT ready (model=%s, language=%s)",
                self._config.stt.model,
                self._config.stt.language,
            )
            self._pipeline.run(self._stop_event)
        finally:
            self._overlay.shutdown()

    def _shutdown(self) -> None:
        self._stop_event.set()
        self._audio.shutdown()
        self._wake.shutdown()
        self._brain.shutdown()
        self._tts.shutdown()
        self._stt.shutdown()
        _LOG.info("Application stopped")
