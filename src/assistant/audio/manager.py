from collections.abc import Callable

from assistant.audio.devices import AudioDeviceCatalog
from assistant.audio.models import AudioData, AudioFormat
from assistant.audio.player import AudioPlayer, LevelCallback, PcmQueue
from assistant.audio.recorder import AudioRecorder
from assistant.config import AudioConfig
from assistant.constants.audio import AUDIO_DEFAULT_READ_TIMEOUT_SECONDS
from assistant.logger import Logger

_LOG = Logger.get(__name__)


class AudioManager:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._format = AudioFormat(
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self._catalog = AudioDeviceCatalog()
        self._recorder = AudioRecorder()
        self._player = AudioPlayer()
        self._input_device = config.input_device
        self._output_device = config.output_device

    @property
    def format(self) -> AudioFormat:
        return self._format

    def initialize(self) -> None:
        self._format.validate()

        if self._input_device is not None:
            configured_input = self._catalog.validate_input_device(self._input_device)
            _LOG.info(
                "Configured input: [%d] %s (%s)",
                configured_input.index,
                configured_input.name,
                configured_input.hostapi_name,
            )
        else:
            default_input = self._catalog.get_default_input_device()
            if default_input is not None:
                _LOG.info(
                    "Default input: [%d] %s (%s)",
                    default_input.index,
                    default_input.name,
                    default_input.hostapi_name,
                )

        if self._output_device is not None:
            configured_output = self._catalog.validate_output_device(self._output_device)
            _LOG.info(
                "Configured output: [%d] %s (%s)",
                configured_output.index,
                configured_output.name,
                configured_output.hostapi_name,
            )
        else:
            default_output = self._catalog.get_default_output_device()
            if default_output is not None:
                _LOG.info(
                    "Default output: [%d] %s (%s)",
                    default_output.index,
                    default_output.name,
                    default_output.hostapi_name,
                )

    def shutdown(self) -> None:
        self.stop_capture()
        self.stop_playback()

    def start_capture(self) -> None:
        self._recorder.start(
            self._format,
            device=self._input_device,
            blocksize=self._config.blocksize,
        )

    def read_chunk(self, *, timeout: float | None = AUDIO_DEFAULT_READ_TIMEOUT_SECONDS) -> AudioData | None:
        return self._recorder.read(timeout=timeout)

    def stop_capture(self) -> None:
        self._recorder.stop()

    def play(self, audio: AudioData, *, on_level: LevelCallback | None = None) -> None:
        self._with_capture_paused(lambda: self._player.play(audio=audio, device=self._output_device, on_level=on_level))

    def play_stream(
        self,
        pcm_queue: PcmQueue,
        *,
        sample_rate: int,
        on_level: LevelCallback | None = None,
    ) -> None:
        self._with_capture_paused(
            lambda: self._player.play_stream(
                pcm_queue,
                sample_rate=sample_rate,
                channels=1,
                device=self._output_device,
                on_level=on_level,
            )
        )

    def stop_playback(self) -> None:
        self._player.stop()

    def _with_capture_paused(self, action: Callable[[], None]) -> None:
        was_capturing = self._recorder.is_active

        if was_capturing:
            _LOG.debug("Pausing capture for playback")
            self._recorder.stop()

        try:
            action()
        finally:
            if was_capturing and not self._recorder.is_active:
                _LOG.debug("Resuming capture after playback")
                self.start_capture()
