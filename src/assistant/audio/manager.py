from __future__ import annotations

from assistant.audio.devices import AudioDevice, AudioDeviceCatalog
from assistant.audio.exceptions import AudioError, AudioPlaybackError
from assistant.audio.models import AudioChunk, AudioData, AudioFormat
from assistant.audio.player import AudioPlayer
from assistant.audio.protocol import AudioCapture, AudioPlayback
from assistant.audio.recorder import AudioRecorder
from assistant.config import AudioConfig
from assistant.logger import Logger


class AudioManager:
    def __init__(
        self,
        config: AudioConfig,
        *,
        catalog: AudioDeviceCatalog | None = None,
        capture: AudioCapture | None = None,
        playback: AudioPlayback | None = None,
    ) -> None:
        self._logger = Logger.get(__name__)
        self._config = config
        self._format = AudioFormat(
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self._catalog = catalog or AudioDeviceCatalog()
        self._capture: AudioCapture = capture or AudioRecorder()
        self._playback: AudioPlayback = playback or AudioPlayer()
        self._input_device = config.input_device
        self._output_device = config.output_device

    @property
    def format(self) -> AudioFormat:
        return self._format

    @property
    def is_capturing(self) -> bool:
        return self._capture.is_active

    @property
    def is_playing(self) -> bool:
        return self._playback.is_active

    def initialize(self) -> None:
        self._format.validate()

        if self._config.blocksize < 0:
            raise AudioError(f"Invalid blocksize: {self._config.blocksize}")

        if self._input_device is not None:
            device = self._catalog.validate_input_device(self._input_device)
            self._logger.info(
                "Configured input: [%d] %s (%s)",
                device.index,
                device.name,
                device.hostapi_name,
            )

        if self._output_device is not None:
            device = self._catalog.validate_output_device(self._output_device)
            self._logger.info(
                "Configured output: [%d] %s (%s)",
                device.index,
                device.name,
                device.hostapi_name,
            )

    def shutdown(self) -> None:
        self.stop_capture()
        self.stop_playback()

    def get_input_devices(self) -> list[AudioDevice]:
        return self._catalog.list_input_devices()

    def get_output_devices(self) -> list[AudioDevice]:
        return self._catalog.list_output_devices()

    def get_default_input_device(self) -> AudioDevice | None:
        return self._catalog.get_default_input_device()

    def get_default_output_device(self) -> AudioDevice | None:
        return self._catalog.get_default_output_device()

    def set_input_device(self, index: int | None) -> None:
        if index is not None:
            self._catalog.validate_input_device(index)

        self._input_device = index

    def set_output_device(self, index: int | None) -> None:
        if index is not None:
            self._catalog.validate_output_device(index)

        self._output_device = index

    def start_capture(self) -> None:
        self._capture.start(
            self._format,
            device=self._input_device,
            blocksize=self._config.blocksize,
        )

    def read_chunk(self, *, timeout: float | None = 0.1) -> AudioChunk | None:
        return self._capture.read(timeout=timeout)

    def stop_capture(self) -> None:
        self._capture.stop()

    def record(self, duration: float) -> AudioData:
        return self._capture.record(
            duration=duration,
            audio_format=self._format,
            device=self._input_device,
            blocksize=self._config.blocksize,
        )

    def play(self, audio: AudioData, *, blocking: bool = True) -> None:
        was_capturing = self._capture.is_active

        if was_capturing and not blocking:
            raise AudioPlaybackError(
                "Cannot start non-blocking playback while capture is active; stop capture first or use blocking=True"
            )

        if was_capturing:
            self._logger.info("Pausing capture for playback")
            self._capture.stop()

        try:
            self._playback.play(
                audio=audio,
                device=self._output_device,
                blocking=blocking,
            )
        finally:
            if was_capturing and not self._capture.is_active:
                self._logger.info("Resuming capture after playback")
                self._capture.start(
                    self._format,
                    device=self._input_device,
                    blocksize=self._config.blocksize,
                )

    def stop_playback(self) -> None:
        self._playback.stop()

    def wait_playback(self) -> None:
        self._playback.wait()
