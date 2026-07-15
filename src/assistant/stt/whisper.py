from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from assistant.audio.dsp import to_mono
from assistant.audio.models import AudioData
from assistant.config import STT_SAMPLE_RATE, SttConfig
from assistant.logger import Logger
from assistant.stt.exceptions import SttError, SttNotReadyError
from assistant.stt.models import Transcript, TranscriptSegment

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

_WHISPER_ERRORS = (RuntimeError, ValueError, OSError, MemoryError)


class WhisperStt:
    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._logger = Logger.get(__name__)
        self._model: WhisperModel | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def initialize(self) -> None:
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise SttError("faster-whisper is not installed. Run: uv sync") from error

        last_error: Exception | None = None

        for device, compute_type in self._device_candidates():
            self._logger.info(
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

                if device == "cuda" and self._config.device == "auto":
                    self._logger.warning(
                        "CUDA load failed (%s), falling back to CPU",
                        error,
                    )
                    continue

                raise SttError(
                    f"Failed to load Whisper model {self._config.model!r} "
                    f"(device={device}, compute_type={compute_type}): {error}"
                ) from error

            self._logger.info("Whisper model loaded on %s (%s)", device, compute_type)
            return

        raise SttError(f"Failed to load Whisper model {self._config.model!r}: {last_error}") from last_error

    def shutdown(self) -> None:
        self._model = None

    def transcribe(
        self,
        audio: AudioData,
        *,
        vad_filter: bool | None = None,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        no_speech_threshold: float | None = None,
    ) -> Transcript:
        if self._model is None:
            raise SttNotReadyError("Speech-to-text is not initialized")

        samples = self._prepare_samples(audio)

        if samples.size == 0:
            return Transcript(text="", language=self._config.language, segments=(), duration=0.0)

        use_vad = self._config.vad_filter if vad_filter is None else vad_filter
        use_beam = self._config.beam_size if beam_size is None else beam_size

        try:
            segments_iter, info = self._model.transcribe(
                samples,
                language=self._config.language,
                beam_size=use_beam,
                vad_filter=use_vad,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                no_speech_threshold=(0.6 if no_speech_threshold is None else no_speech_threshold),
                compression_ratio_threshold=2.4,
                log_prob_threshold=-0.8,
                without_timestamps=True,
            )
            raw_segments = list(segments_iter)
        except _WHISPER_ERRORS as error:
            raise SttError(
                f"Failed to transcribe audio (frames={samples.shape[0]}, sample_rate={STT_SAMPLE_RATE}): {error}"
            ) from error

        segments = tuple(
            TranscriptSegment(
                text=segment.text.strip(),
                start=float(segment.start),
                end=float(segment.end),
            )
            for segment in raw_segments
            if segment.text and segment.text.strip()
        )
        text = " ".join(segment.text for segment in segments).strip()
        duration = float(samples.shape[0] / STT_SAMPLE_RATE)

        self._logger.debug(
            "Transcribed %d frames -> %d chars (language=%s)",
            samples.shape[0],
            len(text),
            info.language,
        )

        return Transcript(
            text=text,
            language=info.language or self._config.language,
            segments=segments,
            duration=duration,
        )

    def _device_candidates(self) -> list[tuple[str, str]]:
        if self._config.device != "auto":
            device = self._config.device
            return [(device, self._resolve_compute_type(device))]

        candidates: list[tuple[str, str]] = []

        if self._cuda_available():
            candidates.append(("cuda", self._resolve_compute_type("cuda")))

        candidates.append(("cpu", self._resolve_compute_type("cpu")))
        return candidates

    def _cuda_available(self) -> bool:
        try:
            import ctranslate2

            return bool(ctranslate2.get_supported_compute_types("cuda"))
        except (*_WHISPER_ERRORS, ImportError):
            return False

    def _resolve_compute_type(self, device: str) -> str:
        if self._config.compute_type != "auto":
            return self._config.compute_type

        if device == "cuda":
            return "float16"

        return "int8"

    def _prepare_samples(self, audio: AudioData) -> NDArray[np.float32]:
        audio.format.validate()

        if audio.format.sample_rate != STT_SAMPLE_RATE:
            raise SttError(
                f"Unsupported sample rate for Whisper: {audio.format.sample_rate} Hz (expected {STT_SAMPLE_RATE} Hz)"
            )

        raw = np.asarray(audio.samples, dtype=np.float32)
        if raw.ndim == 2 and raw.shape[1] > 1:
            self._logger.warning("Downmixing %d channels to mono for Whisper", raw.shape[1])

        return to_mono(raw)
