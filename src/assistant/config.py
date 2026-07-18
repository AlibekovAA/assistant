from dataclasses import dataclass, field
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
import os

from dotenv import load_dotenv

from assistant.constants.app import APP_NAME, FALLBACK_VERSION, PACKAGE_NAME
from assistant.constants.audio import AUDIO_DEFAULT_BLOCKSIZE, AUDIO_DEFAULT_CHANNELS, STT_SAMPLE_RATE
from assistant.constants.llm import (
    GIGACHAT_DEFAULT_MAX_TOKENS,
    GIGACHAT_DEFAULT_MODEL,
    GIGACHAT_DEFAULT_SCOPE,
    GIGACHAT_DEFAULT_TEMPERATURE,
    GIGACHAT_DEFAULT_TIMEOUT_SECONDS,
    GIGACHAT_DEFAULT_VERIFY_SSL,
    GigaChatScope,
)
from assistant.constants.speech import (
    SPEECH_DEFAULT_MAX_SECONDS,
    SPEECH_DEFAULT_MIN_SECONDS,
    SPEECH_DEFAULT_ONSET_SECONDS,
    SPEECH_DEFAULT_RMS,
    SPEECH_DEFAULT_SILENCE_SECONDS,
)
from assistant.constants.tts import TTS_DEFAULT_RATE, TTS_DEFAULT_SAMPLE_RATE, TTS_DEFAULT_VOICE
from assistant.constants.wake import (
    WAKE_DEFAULT_BEAM_SIZE,
    WAKE_DEFAULT_HOP_SECONDS,
    WAKE_DEFAULT_KEYWORD,
    WAKE_DEFAULT_LISTEN_PEAK,
    WAKE_DEFAULT_LISTEN_RMS,
    WAKE_DEFAULT_LISTEN_SNR,
    WAKE_DEFAULT_NO_SPEECH,
    WAKE_DEFAULT_POST_PRUNE_SECONDS,
    WAKE_DEFAULT_VAD_FILTER,
    WAKE_DEFAULT_WINDOW_SECONDS,
)
from assistant.constants.weather import WEATHER_DEFAULT_CITY, WEATHER_DEFAULT_TIMEZONE
from assistant.constants.whisper import (
    WHISPER_DEFAULT_BEAM_SIZE,
    WHISPER_DEFAULT_COMPUTE_TYPE,
    WHISPER_DEFAULT_CPU_THREADS,
    WHISPER_DEFAULT_DEVICE,
    WHISPER_DEFAULT_LANGUAGE,
    WHISPER_DEFAULT_MODEL,
    WHISPER_DEFAULT_NO_SPEECH,
    WHISPER_DEFAULT_TEMPERATURE,
    WHISPER_DEFAULT_VAD_FILTER,
    WhisperComputeType,
    WhisperDevice,
)
from assistant.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = STT_SAMPLE_RATE
    channels: int = AUDIO_DEFAULT_CHANNELS
    blocksize: int = AUDIO_DEFAULT_BLOCKSIZE


@dataclass(frozen=True, slots=True)
class SttConfig:
    model: str = WHISPER_DEFAULT_MODEL
    language: str = WHISPER_DEFAULT_LANGUAGE
    device: WhisperDevice = WHISPER_DEFAULT_DEVICE
    compute_type: WhisperComputeType = WHISPER_DEFAULT_COMPUTE_TYPE
    beam_size: int = WHISPER_DEFAULT_BEAM_SIZE
    vad_filter: bool = WHISPER_DEFAULT_VAD_FILTER
    temperature: float = WHISPER_DEFAULT_TEMPERATURE
    no_speech_threshold: float = WHISPER_DEFAULT_NO_SPEECH
    cpu_threads: int = WHISPER_DEFAULT_CPU_THREADS
    download_root: str | None = None


@dataclass(frozen=True, slots=True)
class WakeConfig:
    keyword: str = WAKE_DEFAULT_KEYWORD
    window_seconds: float = WAKE_DEFAULT_WINDOW_SECONDS
    hop_seconds: float = WAKE_DEFAULT_HOP_SECONDS
    listen_rms_threshold: float = WAKE_DEFAULT_LISTEN_RMS
    listen_peak_threshold: float = WAKE_DEFAULT_LISTEN_PEAK
    listen_snr: float = WAKE_DEFAULT_LISTEN_SNR
    post_wake_prune_seconds: float = WAKE_DEFAULT_POST_PRUNE_SECONDS
    beam_size: int = WAKE_DEFAULT_BEAM_SIZE
    vad_filter: bool = WAKE_DEFAULT_VAD_FILTER
    no_speech_threshold: float = WAKE_DEFAULT_NO_SPEECH


@dataclass(frozen=True, slots=True)
class UtteranceConfig:
    speech_rms_threshold: float = SPEECH_DEFAULT_RMS
    speech_onset_seconds: float = SPEECH_DEFAULT_ONSET_SECONDS
    min_speech_seconds: float = SPEECH_DEFAULT_MIN_SECONDS
    silence_seconds: float = SPEECH_DEFAULT_SILENCE_SECONDS
    utterance_max_seconds: float = SPEECH_DEFAULT_MAX_SECONDS


@dataclass(frozen=True, slots=True)
class TtsConfig:
    voice: str = TTS_DEFAULT_VOICE
    rate: str = TTS_DEFAULT_RATE
    sample_rate: int = TTS_DEFAULT_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class GigaChatConfig:
    credentials: str
    scope: GigaChatScope = GIGACHAT_DEFAULT_SCOPE
    model: str = GIGACHAT_DEFAULT_MODEL
    verify_ssl_certs: bool = GIGACHAT_DEFAULT_VERIFY_SSL
    timeout_seconds: float = GIGACHAT_DEFAULT_TIMEOUT_SECONDS
    temperature: float = GIGACHAT_DEFAULT_TEMPERATURE
    max_tokens: int = GIGACHAT_DEFAULT_MAX_TOKENS


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    default_city: str = WEATHER_DEFAULT_CITY
    default_timezone: str = WEATHER_DEFAULT_TIMEZONE


@dataclass(frozen=True, slots=True)
class Config:
    app_name: str = APP_NAME
    app_version: str = FALLBACK_VERSION
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    utterance: UtteranceConfig = field(default_factory=UtteranceConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    gigachat: GigaChatConfig = field(default_factory=lambda: GigaChatConfig(credentials=""))
    tools: ToolsConfig = field(default_factory=ToolsConfig)


def load_config() -> Config:
    load_dotenv()

    audio_defaults = AudioConfig()
    stt_defaults = SttConfig()
    wake_defaults = WakeConfig()
    utterance_defaults = UtteranceConfig()
    tts_defaults = TtsConfig()
    gigachat_defaults = GigaChatConfig(credentials="")
    tools_defaults = ToolsConfig()

    try:
        audio = AudioConfig(
            input_device=_optional_int("ASSISTANT_INPUT_DEVICE"),
            output_device=_optional_int("ASSISTANT_OUTPUT_DEVICE"),
            sample_rate=_int("ASSISTANT_SAMPLE_RATE", audio_defaults.sample_rate),
            channels=_int("ASSISTANT_CHANNELS", audio_defaults.channels),
            blocksize=_int("ASSISTANT_BLOCKSIZE", audio_defaults.blocksize),
        )
        language = _str("ASSISTANT_STT_LANGUAGE", stt_defaults.language)
        if language.lower() != WHISPER_DEFAULT_LANGUAGE:
            raise ConfigurationError(
                f"Unsupported ASSISTANT_STT_LANGUAGE: {language!r} (only {WHISPER_DEFAULT_LANGUAGE!r} is supported)"
            )

        stt = SttConfig(
            model=_non_empty("ASSISTANT_WHISPER_MODEL", stt_defaults.model),
            language=WHISPER_DEFAULT_LANGUAGE,
            device=_enum("ASSISTANT_WHISPER_DEVICE", WhisperDevice, stt_defaults.device),
            compute_type=_enum(
                "ASSISTANT_WHISPER_COMPUTE_TYPE",
                WhisperComputeType,
                stt_defaults.compute_type,
            ),
            beam_size=_int("ASSISTANT_WHISPER_BEAM_SIZE", stt_defaults.beam_size),
            vad_filter=_bool("ASSISTANT_WHISPER_VAD_FILTER", stt_defaults.vad_filter),
            temperature=_float("ASSISTANT_WHISPER_TEMPERATURE", stt_defaults.temperature),
            no_speech_threshold=_float("ASSISTANT_WHISPER_NO_SPEECH", stt_defaults.no_speech_threshold),
            cpu_threads=_int("ASSISTANT_WHISPER_CPU_THREADS", stt_defaults.cpu_threads),
            download_root=_optional_str("ASSISTANT_WHISPER_DOWNLOAD_ROOT"),
        )
        wake = WakeConfig(
            keyword=_non_empty("ASSISTANT_WAKE_KEYWORD", wake_defaults.keyword),
            window_seconds=_float("ASSISTANT_WAKE_WINDOW_SECONDS", wake_defaults.window_seconds),
            hop_seconds=_float("ASSISTANT_WAKE_HOP_SECONDS", wake_defaults.hop_seconds),
            listen_rms_threshold=_float("ASSISTANT_WAKE_LISTEN_RMS", wake_defaults.listen_rms_threshold),
            listen_peak_threshold=_float("ASSISTANT_WAKE_LISTEN_PEAK", wake_defaults.listen_peak_threshold),
            listen_snr=_float("ASSISTANT_WAKE_LISTEN_SNR", wake_defaults.listen_snr),
            post_wake_prune_seconds=_float(
                "ASSISTANT_WAKE_POST_WAKE_PRUNE_SECONDS",
                wake_defaults.post_wake_prune_seconds,
            ),
            beam_size=_int("ASSISTANT_WAKE_BEAM_SIZE", wake_defaults.beam_size),
            vad_filter=_bool("ASSISTANT_WAKE_VAD_FILTER", wake_defaults.vad_filter),
            no_speech_threshold=_float("ASSISTANT_WAKE_NO_SPEECH", wake_defaults.no_speech_threshold),
        )
        utterance = UtteranceConfig(
            speech_rms_threshold=_float("ASSISTANT_UTTERANCE_SPEECH_RMS", utterance_defaults.speech_rms_threshold),
            speech_onset_seconds=_float(
                "ASSISTANT_UTTERANCE_SPEECH_ONSET_SECONDS",
                utterance_defaults.speech_onset_seconds,
            ),
            min_speech_seconds=_float(
                "ASSISTANT_UTTERANCE_MIN_SPEECH_SECONDS",
                utterance_defaults.min_speech_seconds,
            ),
            silence_seconds=_float("ASSISTANT_UTTERANCE_SILENCE_SECONDS", utterance_defaults.silence_seconds),
            utterance_max_seconds=_float(
                "ASSISTANT_UTTERANCE_MAX_SECONDS",
                utterance_defaults.utterance_max_seconds,
            ),
        )
        tts = TtsConfig(
            voice=_non_empty("ASSISTANT_TTS_VOICE", tts_defaults.voice),
            rate=_str("ASSISTANT_TTS_RATE", tts_defaults.rate) or tts_defaults.rate,
            sample_rate=_int("ASSISTANT_TTS_SAMPLE_RATE", tts_defaults.sample_rate),
        )
        gigachat = GigaChatConfig(
            credentials=_required_secret("ASSISTANT_GIGACHAT_CREDENTIALS"),
            scope=_enum("ASSISTANT_GIGACHAT_SCOPE", GigaChatScope, gigachat_defaults.scope),
            model=_non_empty("ASSISTANT_GIGACHAT_MODEL", gigachat_defaults.model),
            verify_ssl_certs=_bool(
                "ASSISTANT_GIGACHAT_VERIFY_SSL",
                gigachat_defaults.verify_ssl_certs,
            ),
            timeout_seconds=_float(
                "ASSISTANT_GIGACHAT_TIMEOUT_SECONDS",
                gigachat_defaults.timeout_seconds,
            ),
            temperature=_float("ASSISTANT_GIGACHAT_TEMPERATURE", gigachat_defaults.temperature),
            max_tokens=_int("ASSISTANT_GIGACHAT_MAX_TOKENS", gigachat_defaults.max_tokens),
        )
        tools = ToolsConfig(
            default_city=_non_empty("ASSISTANT_DEFAULT_CITY", tools_defaults.default_city),
            default_timezone=_non_empty("ASSISTANT_DEFAULT_TIMEZONE", tools_defaults.default_timezone),
        )
    except ValueError as error:
        raise ConfigurationError(f"Invalid configuration: {error}") from error

    _validate_audio(audio)
    _validate_stt(stt)
    _validate_wake(wake)
    _validate_utterance(utterance)
    _validate_tts(tts)
    _validate_gigachat(gigachat)

    return Config(
        app_version=_package_version(),
        audio=audio,
        stt=stt,
        wake=wake,
        utterance=utterance,
        tts=tts,
        gigachat=gigachat,
        tools=tools,
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


def _validate_gigachat(gigachat: GigaChatConfig) -> None:
    _require_positive("ASSISTANT_GIGACHAT_TIMEOUT_SECONDS", gigachat.timeout_seconds)
    if not 0 <= gigachat.temperature <= 2:
        raise ConfigurationError(f"Invalid ASSISTANT_GIGACHAT_TEMPERATURE: {gigachat.temperature}")
    _require_positive_int("ASSISTANT_GIGACHAT_MAX_TOKENS", gigachat.max_tokens)


def _required_secret(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigurationError(
            f"{name} is required. Set it to the GigaChat Authorization Key (base64 Client ID:Client Secret)."
        )
    return value.strip()


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
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION


def _enum[E: StrEnum](name: str, enum_type: type[E], default: E) -> E:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return enum_type(value.strip())
    except ValueError as error:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise ConfigurationError(f"Invalid {name}: {value!r} (expected one of {allowed})") from error


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
