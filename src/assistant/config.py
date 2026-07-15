from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import os

from assistant.core.exceptions import ConfigurationError

STT_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = STT_SAMPLE_RATE
    channels: int = 1
    blocksize: int = 1024


@dataclass(frozen=True, slots=True)
class SttConfig:
    model: str = "small"
    language: str = "ru"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    cpu_threads: int = 0
    download_root: str | None = None


@dataclass(frozen=True, slots=True)
class WakeConfig:
    keyword: str = "мина"
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    listen_rms_threshold: float = 0.008
    listen_peak_threshold: float = 0.02
    listen_snr: float = 2.5
    speech_rms_threshold: float = 0.008
    speech_onset_seconds: float = 0.15
    min_speech_seconds: float = 0.5
    silence_seconds: float = 1.2
    utterance_max_seconds: float = 12.0
    post_wake_prune_seconds: float = 0.35


@dataclass(frozen=True, slots=True)
class TtsConfig:
    voice: str = "ru-RU-SvetlanaNeural"
    rate: str = "+0%"
    sample_rate: int = 24_000


@dataclass(frozen=True, slots=True)
class Config:
    app_name: str = "Мина"
    app_version: str = "0.0.0"
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)


def load_config() -> Config:
    try:
        audio = AudioConfig(
            input_device=_optional_int("ASSISTANT_INPUT_DEVICE"),
            output_device=_optional_int("ASSISTANT_OUTPUT_DEVICE"),
            sample_rate=_int("ASSISTANT_SAMPLE_RATE", STT_SAMPLE_RATE),
            channels=_int("ASSISTANT_CHANNELS", 1),
            blocksize=_int("ASSISTANT_BLOCKSIZE", 1024),
        )
        stt = SttConfig(
            model=_str("ASSISTANT_WHISPER_MODEL", "small"),
            language=_str("ASSISTANT_STT_LANGUAGE", "ru"),
            device=_str("ASSISTANT_WHISPER_DEVICE", "auto"),
            compute_type=_str("ASSISTANT_WHISPER_COMPUTE_TYPE", "auto"),
            beam_size=_int("ASSISTANT_WHISPER_BEAM_SIZE", 5),
            vad_filter=_bool("ASSISTANT_WHISPER_VAD_FILTER", True),
            cpu_threads=_int("ASSISTANT_WHISPER_CPU_THREADS", 0),
            download_root=_optional_str("ASSISTANT_WHISPER_DOWNLOAD_ROOT"),
        )
        wake = WakeConfig(
            keyword=_str("ASSISTANT_WAKE_KEYWORD", "мина"),
            window_seconds=_float("ASSISTANT_WAKE_WINDOW_SECONDS", 2.0),
            hop_seconds=_float("ASSISTANT_WAKE_HOP_SECONDS", 1.0),
            listen_rms_threshold=_float("ASSISTANT_WAKE_LISTEN_RMS", 0.008),
            listen_peak_threshold=_float("ASSISTANT_WAKE_LISTEN_PEAK", 0.02),
            listen_snr=_float("ASSISTANT_WAKE_LISTEN_SNR", 2.5),
            speech_rms_threshold=_float("ASSISTANT_WAKE_SPEECH_RMS", 0.008),
            speech_onset_seconds=_float("ASSISTANT_WAKE_SPEECH_ONSET_SECONDS", 0.15),
            min_speech_seconds=_float("ASSISTANT_WAKE_MIN_SPEECH_SECONDS", 0.5),
            silence_seconds=_float("ASSISTANT_WAKE_SILENCE_SECONDS", 1.2),
            utterance_max_seconds=_float("ASSISTANT_WAKE_UTTERANCE_MAX_SECONDS", 12.0),
            post_wake_prune_seconds=_float("ASSISTANT_WAKE_POST_WAKE_PRUNE_SECONDS", 0.35),
        )
        tts = TtsConfig(
            voice=_str("ASSISTANT_TTS_VOICE", "ru-RU-SvetlanaNeural"),
            rate=_str("ASSISTANT_TTS_RATE", "+0%"),
            sample_rate=_int("ASSISTANT_TTS_SAMPLE_RATE", 24_000),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid configuration: {error}") from error

    if audio.sample_rate != STT_SAMPLE_RATE:
        raise ConfigurationError(
            f"Invalid ASSISTANT_SAMPLE_RATE: {audio.sample_rate} (Whisper STT requires {STT_SAMPLE_RATE} Hz)"
        )

    if audio.channels != 1:
        raise ConfigurationError(f"Invalid ASSISTANT_CHANNELS: {audio.channels} (wake word requires mono)")

    if audio.blocksize < 0:
        raise ConfigurationError(f"Invalid ASSISTANT_BLOCKSIZE: {audio.blocksize}")

    if not stt.model.strip():
        raise ConfigurationError("ASSISTANT_WHISPER_MODEL must not be empty")

    if stt.language.strip().lower() != "ru":
        raise ConfigurationError(f"Unsupported ASSISTANT_STT_LANGUAGE: {stt.language!r} (only 'ru' is supported)")

    if stt.device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError(f"Invalid ASSISTANT_WHISPER_DEVICE: {stt.device!r}")

    if stt.beam_size < 1:
        raise ConfigurationError(f"Invalid ASSISTANT_WHISPER_BEAM_SIZE: {stt.beam_size}")

    if stt.cpu_threads < 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WHISPER_CPU_THREADS: {stt.cpu_threads}")

    if not wake.keyword.strip():
        raise ConfigurationError("ASSISTANT_WAKE_KEYWORD must not be empty")

    if wake.window_seconds <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_WINDOW_SECONDS: {wake.window_seconds}")

    if wake.hop_seconds <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_HOP_SECONDS: {wake.hop_seconds}")

    if wake.hop_seconds > wake.window_seconds:
        raise ConfigurationError("ASSISTANT_WAKE_HOP_SECONDS must be <= ASSISTANT_WAKE_WINDOW_SECONDS")

    if wake.listen_rms_threshold <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_LISTEN_RMS: {wake.listen_rms_threshold}")

    if wake.listen_peak_threshold <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_LISTEN_PEAK: {wake.listen_peak_threshold}")

    if wake.listen_snr <= 1.0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_LISTEN_SNR: {wake.listen_snr}")

    if wake.speech_rms_threshold <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_SPEECH_RMS: {wake.speech_rms_threshold}")

    if wake.speech_onset_seconds <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_SPEECH_ONSET_SECONDS: {wake.speech_onset_seconds}")

    if wake.min_speech_seconds <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_MIN_SPEECH_SECONDS: {wake.min_speech_seconds}")

    if wake.silence_seconds <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_SILENCE_SECONDS: {wake.silence_seconds}")

    if wake.utterance_max_seconds <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_UTTERANCE_MAX_SECONDS: {wake.utterance_max_seconds}")

    if wake.post_wake_prune_seconds < 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_POST_WAKE_PRUNE_SECONDS: {wake.post_wake_prune_seconds}")

    if not tts.voice.strip():
        raise ConfigurationError("ASSISTANT_TTS_VOICE must not be empty")

    if tts.sample_rate <= 0:
        raise ConfigurationError(f"Invalid ASSISTANT_TTS_SAMPLE_RATE: {tts.sample_rate}")

    return Config(
        app_version=_package_version(),
        audio=audio,
        stt=SttConfig(
            model=stt.model.strip(),
            language="ru",
            device=stt.device,
            compute_type=stt.compute_type.strip() or "auto",
            beam_size=stt.beam_size,
            vad_filter=stt.vad_filter,
            cpu_threads=stt.cpu_threads,
            download_root=stt.download_root,
        ),
        wake=WakeConfig(
            keyword=wake.keyword.strip(),
            window_seconds=wake.window_seconds,
            hop_seconds=wake.hop_seconds,
            listen_rms_threshold=wake.listen_rms_threshold,
            listen_peak_threshold=wake.listen_peak_threshold,
            listen_snr=wake.listen_snr,
            speech_rms_threshold=wake.speech_rms_threshold,
            speech_onset_seconds=wake.speech_onset_seconds,
            min_speech_seconds=wake.min_speech_seconds,
            silence_seconds=wake.silence_seconds,
            utterance_max_seconds=wake.utterance_max_seconds,
            post_wake_prune_seconds=wake.post_wake_prune_seconds,
        ),
        tts=TtsConfig(
            voice=tts.voice.strip(),
            rate=tts.rate.strip() or "+0%",
            sample_rate=tts.sample_rate,
        ),
    )


def _package_version() -> str:
    try:
        return version("assistant")
    except PackageNotFoundError:
        return "0.0.0"


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return None

    return int(value)


def _optional_str(name: str) -> str | None:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return None

    return value.strip()


def _str(name: str, default: str) -> str:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return value.strip()


def _int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    return float(value)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None or value.strip() == "":
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for {name}: {value!r}")
