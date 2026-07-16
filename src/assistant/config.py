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
    beam_size: int = 8
    vad_filter: bool = True
    temperature: float = 0.0
    no_speech_threshold: float = 0.5
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
    post_wake_prune_seconds: float = 0.35
    beam_size: int = 5
    vad_filter: bool = True
    no_speech_threshold: float = 0.7


@dataclass(frozen=True, slots=True)
class UtteranceConfig:
    speech_rms_threshold: float = 0.008
    speech_onset_seconds: float = 0.15
    min_speech_seconds: float = 0.5
    silence_seconds: float = 1.2
    utterance_max_seconds: float = 12.0


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
    utterance: UtteranceConfig = field(default_factory=UtteranceConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)


_DEFAULT_AUDIO = AudioConfig()
_DEFAULT_STT = SttConfig()
_DEFAULT_WAKE = WakeConfig()
_DEFAULT_UTTERANCE = UtteranceConfig()
_DEFAULT_TTS = TtsConfig()


def load_config() -> Config:
    try:
        audio = AudioConfig(
            input_device=_optional_int("ASSISTANT_INPUT_DEVICE"),
            output_device=_optional_int("ASSISTANT_OUTPUT_DEVICE"),
            sample_rate=_int("ASSISTANT_SAMPLE_RATE", _DEFAULT_AUDIO.sample_rate),
            channels=_int("ASSISTANT_CHANNELS", _DEFAULT_AUDIO.channels),
            blocksize=_int("ASSISTANT_BLOCKSIZE", _DEFAULT_AUDIO.blocksize),
        )
        language = _str("ASSISTANT_STT_LANGUAGE", _DEFAULT_STT.language)
        if language.lower() != "ru":
            raise ConfigurationError(f"Unsupported ASSISTANT_STT_LANGUAGE: {language!r} (only 'ru' is supported)")

        stt = SttConfig(
            model=_non_empty("ASSISTANT_WHISPER_MODEL", _DEFAULT_STT.model),
            language="ru",
            device=_str("ASSISTANT_WHISPER_DEVICE", _DEFAULT_STT.device),
            compute_type=_str("ASSISTANT_WHISPER_COMPUTE_TYPE", _DEFAULT_STT.compute_type) or _DEFAULT_STT.compute_type,
            beam_size=_int("ASSISTANT_WHISPER_BEAM_SIZE", _DEFAULT_STT.beam_size),
            vad_filter=_bool("ASSISTANT_WHISPER_VAD_FILTER", _DEFAULT_STT.vad_filter),
            temperature=_float("ASSISTANT_WHISPER_TEMPERATURE", _DEFAULT_STT.temperature),
            no_speech_threshold=_float("ASSISTANT_WHISPER_NO_SPEECH", _DEFAULT_STT.no_speech_threshold),
            cpu_threads=_int("ASSISTANT_WHISPER_CPU_THREADS", _DEFAULT_STT.cpu_threads),
            download_root=_optional_str("ASSISTANT_WHISPER_DOWNLOAD_ROOT"),
        )
        wake = WakeConfig(
            keyword=_non_empty("ASSISTANT_WAKE_KEYWORD", _DEFAULT_WAKE.keyword),
            window_seconds=_float("ASSISTANT_WAKE_WINDOW_SECONDS", _DEFAULT_WAKE.window_seconds),
            hop_seconds=_float("ASSISTANT_WAKE_HOP_SECONDS", _DEFAULT_WAKE.hop_seconds),
            listen_rms_threshold=_float("ASSISTANT_WAKE_LISTEN_RMS", _DEFAULT_WAKE.listen_rms_threshold),
            listen_peak_threshold=_float("ASSISTANT_WAKE_LISTEN_PEAK", _DEFAULT_WAKE.listen_peak_threshold),
            listen_snr=_float("ASSISTANT_WAKE_LISTEN_SNR", _DEFAULT_WAKE.listen_snr),
            post_wake_prune_seconds=_float(
                "ASSISTANT_WAKE_POST_WAKE_PRUNE_SECONDS", _DEFAULT_WAKE.post_wake_prune_seconds
            ),
            beam_size=_int("ASSISTANT_WAKE_BEAM_SIZE", _DEFAULT_WAKE.beam_size),
            vad_filter=_bool("ASSISTANT_WAKE_VAD_FILTER", _DEFAULT_WAKE.vad_filter),
            no_speech_threshold=_float("ASSISTANT_WAKE_NO_SPEECH", _DEFAULT_WAKE.no_speech_threshold),
        )
        utterance = UtteranceConfig(
            speech_rms_threshold=_float("ASSISTANT_UTTERANCE_SPEECH_RMS", _DEFAULT_UTTERANCE.speech_rms_threshold),
            speech_onset_seconds=_float(
                "ASSISTANT_UTTERANCE_SPEECH_ONSET_SECONDS",
                _DEFAULT_UTTERANCE.speech_onset_seconds,
            ),
            min_speech_seconds=_float("ASSISTANT_UTTERANCE_MIN_SPEECH_SECONDS", _DEFAULT_UTTERANCE.min_speech_seconds),
            silence_seconds=_float("ASSISTANT_UTTERANCE_SILENCE_SECONDS", _DEFAULT_UTTERANCE.silence_seconds),
            utterance_max_seconds=_float(
                "ASSISTANT_UTTERANCE_MAX_SECONDS",
                _DEFAULT_UTTERANCE.utterance_max_seconds,
            ),
        )
        tts = TtsConfig(
            voice=_non_empty("ASSISTANT_TTS_VOICE", _DEFAULT_TTS.voice),
            rate=_str("ASSISTANT_TTS_RATE", _DEFAULT_TTS.rate) or _DEFAULT_TTS.rate,
            sample_rate=_int("ASSISTANT_TTS_SAMPLE_RATE", _DEFAULT_TTS.sample_rate),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid configuration: {error}") from error

    _validate_audio(audio)
    _validate_stt(stt)
    _validate_wake(wake)
    _validate_utterance(utterance)
    _validate_tts(tts)

    return Config(
        app_version=_package_version(),
        audio=audio,
        stt=stt,
        wake=wake,
        utterance=utterance,
        tts=tts,
    )


def _validate_audio(audio: AudioConfig) -> None:
    if audio.sample_rate != STT_SAMPLE_RATE:
        raise ConfigurationError(
            f"Invalid ASSISTANT_SAMPLE_RATE: {audio.sample_rate} (Whisper STT requires {STT_SAMPLE_RATE} Hz)"
        )
    if audio.channels != 1:
        raise ConfigurationError(f"Invalid ASSISTANT_CHANNELS: {audio.channels} (wake word requires mono)")
    if audio.blocksize < 0:
        raise ConfigurationError(f"Invalid ASSISTANT_BLOCKSIZE: {audio.blocksize}")


def _validate_stt(stt: SttConfig) -> None:
    if stt.device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError(f"Invalid ASSISTANT_WHISPER_DEVICE: {stt.device!r}")
    _require_positive_int("ASSISTANT_WHISPER_BEAM_SIZE", stt.beam_size)
    if stt.cpu_threads < 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WHISPER_CPU_THREADS: {stt.cpu_threads}")
    _require_unit_interval("ASSISTANT_WHISPER_NO_SPEECH", stt.no_speech_threshold)
    if stt.temperature < 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WHISPER_TEMPERATURE: {stt.temperature}")


def _validate_wake(wake: WakeConfig) -> None:
    _require_positive("ASSISTANT_WAKE_WINDOW_SECONDS", wake.window_seconds)
    _require_positive("ASSISTANT_WAKE_HOP_SECONDS", wake.hop_seconds)
    if wake.hop_seconds > wake.window_seconds:
        raise ConfigurationError("ASSISTANT_WAKE_HOP_SECONDS must be <= ASSISTANT_WAKE_WINDOW_SECONDS")
    _require_positive("ASSISTANT_WAKE_LISTEN_RMS", wake.listen_rms_threshold)
    _require_positive("ASSISTANT_WAKE_LISTEN_PEAK", wake.listen_peak_threshold)
    if wake.listen_snr <= 1.0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_LISTEN_SNR: {wake.listen_snr}")
    if wake.post_wake_prune_seconds < 0:
        raise ConfigurationError(f"Invalid ASSISTANT_WAKE_POST_WAKE_PRUNE_SECONDS: {wake.post_wake_prune_seconds}")
    _require_positive_int("ASSISTANT_WAKE_BEAM_SIZE", wake.beam_size)
    _require_unit_interval("ASSISTANT_WAKE_NO_SPEECH", wake.no_speech_threshold)


def _validate_utterance(utterance: UtteranceConfig) -> None:
    _require_positive("ASSISTANT_UTTERANCE_SPEECH_RMS", utterance.speech_rms_threshold)
    _require_positive("ASSISTANT_UTTERANCE_SPEECH_ONSET_SECONDS", utterance.speech_onset_seconds)
    _require_positive("ASSISTANT_UTTERANCE_MIN_SPEECH_SECONDS", utterance.min_speech_seconds)
    _require_positive("ASSISTANT_UTTERANCE_SILENCE_SECONDS", utterance.silence_seconds)
    _require_positive("ASSISTANT_UTTERANCE_MAX_SECONDS", utterance.utterance_max_seconds)


def _validate_tts(tts: TtsConfig) -> None:
    _require_positive_int("ASSISTANT_TTS_SAMPLE_RATE", tts.sample_rate)


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigurationError(f"Invalid {name}: {value}")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ConfigurationError(f"Invalid {name}: {value}")


def _require_unit_interval(name: str, value: float) -> None:
    if not 0 < value <= 1:
        raise ConfigurationError(f"Invalid {name}: {value}")


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


def _non_empty(name: str, default: str) -> str:
    value = _str(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} must not be empty")
    return value


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
