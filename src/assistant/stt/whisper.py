import gc

from faster_whisper import WhisperModel
import numpy as np
from numpy.typing import NDArray

from assistant.audio.dsp import to_mono
from assistant.audio.models import AudioData
from assistant.config import SttConfig
from assistant.constants.audio import STT_SAMPLE_RATE
from assistant.constants.whisper import (
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_CPU_COMPUTE_TYPE,
    WHISPER_CUDA_COMPUTE_TYPE,
    WHISPER_LOG_PROB_THRESHOLD,
    WhisperComputeType,
    WhisperDevice,
)
from assistant.core.exceptions import SttError
from assistant.logger import Logger
from assistant.stt.models import TranscribeOptions, Transcript

_WHISPER_ERRORS = (RuntimeError, ValueError, OSError, MemoryError)
_LOG = Logger.get(__name__)


class WhisperStt:
    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._model: WhisperModel | None = None

    def initialize(self) -> None:
        if self._model is not None:
            return

        last_error: Exception | None = None

        for device, compute_type in self._device_candidates():
            _LOG.info(
                "Loading Whisper model=%s device=%s compute_type=%s",
                self._config.model,
                device,
                compute_type,
            )

            try:
                self._model = WhisperModel(
                    self._config.model,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=self._config.cpu_threads,
                    download_root=self._config.download_root,
                )
            except _WHISPER_ERRORS as error:
                self._model = None
                last_error = error

                if device == WhisperDevice.CUDA and self._config.device == WhisperDevice.AUTO:
                    _LOG.warning(
                        "CUDA load failed (%s), falling back to CPU",
                        error,
                    )
                    continue

                raise SttError(
                    f"Failed to load Whisper model {self._config.model!r} "
                    f"(device={device}, compute_type={compute_type}): {error}"
                ) from error

            _LOG.info("Whisper model loaded on %s (%s)", device, compute_type)
            return

        raise SttError(f"Failed to load Whisper model {self._config.model!r}: {last_error}") from last_error

    def shutdown(self) -> None:
        model = self._model
        self._model = None
        if model is None:
            return
        del model
        gc.collect()

    def transcribe(self, audio: AudioData, options: TranscribeOptions | None = None) -> Transcript:
        if self._model is None:
            raise SttError("Speech-to-text is not initialized")

        samples = self._prepare_samples(audio)
        if samples.size == 0:
            return Transcript(text="")

        opts = options or TranscribeOptions()
        use_vad = self._config.vad_filter if opts.vad_filter is None else opts.vad_filter
        use_beam = self._config.beam_size if opts.beam_size is None else opts.beam_size
        use_temperature = self._config.temperature if opts.temperature is None else opts.temperature
        use_no_speech = (
            self._config.no_speech_threshold if opts.no_speech_threshold is None else opts.no_speech_threshold
        )

        try:
            segments_iter, info = self._model.transcribe(
                samples,
                language=self._config.language,
                beam_size=use_beam,
                vad_filter=use_vad,
                condition_on_previous_text=False,
                initial_prompt=opts.initial_prompt,
                hotwords=opts.hotwords,
                temperature=use_temperature,
                no_speech_threshold=use_no_speech,
                compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
                log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
                without_timestamps=True,
            )
            text = " ".join(segment.text.strip() for segment in segments_iter if segment.text.strip()).strip()
        except _WHISPER_ERRORS as error:
            raise SttError(
                f"Failed to transcribe audio (frames={samples.shape[0]}, sample_rate={STT_SAMPLE_RATE}): {error}"
            ) from error

        _LOG.debug(
            "Transcribed %d frames -> %d chars (language=%s)",
            samples.shape[0],
            len(text),
            info.language,
        )

        return Transcript(text=text)

    def _device_candidates(self) -> list[tuple[WhisperDevice, WhisperComputeType]]:
        if self._config.device != WhisperDevice.AUTO:
            device = self._config.device
            return [(device, self._resolve_compute_type(device))]

        candidates: list[tuple[WhisperDevice, WhisperComputeType]] = []

        if self._cuda_available():
            candidates.append((WhisperDevice.CUDA, self._resolve_compute_type(WhisperDevice.CUDA)))

        candidates.append((WhisperDevice.CPU, self._resolve_compute_type(WhisperDevice.CPU)))
        return candidates

    def _cuda_available(self) -> bool:
        try:
            import ctranslate2

            return bool(ctranslate2.get_supported_compute_types(WhisperDevice.CUDA))
        except (*_WHISPER_ERRORS, ImportError):
            return False

    def _resolve_compute_type(self, device: WhisperDevice) -> WhisperComputeType:
        if self._config.compute_type != WhisperComputeType.AUTO:
            return self._config.compute_type

        if device == WhisperDevice.CUDA:
            return WHISPER_CUDA_COMPUTE_TYPE

        return WHISPER_CPU_COMPUTE_TYPE

    def _prepare_samples(self, audio: AudioData) -> NDArray[np.float32]:
        audio.format.validate()

        if audio.format.sample_rate != STT_SAMPLE_RATE:
            raise SttError(
                f"Unsupported sample rate for Whisper: {audio.format.sample_rate} Hz (expected {STT_SAMPLE_RATE} Hz)"
            )

        raw = np.asarray(audio.samples, dtype=np.float32)
        if raw.ndim == 2 and raw.shape[1] > 1:
            _LOG.warning("Downmixing %d channels to mono for Whisper", raw.shape[1])

        return to_mono(raw)
