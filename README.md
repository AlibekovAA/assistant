# Ассистент

Локальный голосовой помощник **Мина**.

## Цели проекта

* Минимальное потребление ресурсов.
* Постоянная работа в фоновом режиме.
* Низкая нагрузка на процессор и оперативную память.
* Распознавание речи.
* Озвучивание ответов.
* Активация по ключевой фразе.
* Использование Python 3.13.
* Кроссплатформенная архитектура.

## Структура проекта

```text
src/
└── assistant/
    ├── audio/
    ├── core/
    ├── stt/
    ├── tts/
    └── wake/
```

## Быстрый старт

```bash
make install
make run
```

Скажи **«мина»**, затем фразу — Мина ответит голосом:
«Привет, я голосовой помощник Мина. Вы сказали: …».

## Wake word

Детект через тот же **Whisper** (скользящее окно ~2 с, проверка каждые ~1 с).
Тишину отсекает RMS-порог, чтобы не гонять модель впустую.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `ASSISTANT_WAKE_KEYWORD` | `мина` | ключевое слово |
| `ASSISTANT_WAKE_WINDOW_SECONDS` | `2.0` | окно анализа |
| `ASSISTANT_WAKE_HOP_SECONDS` | `1.0` | шаг проверки |
| `ASSISTANT_WAKE_BEAM_SIZE` | `5` | beam size для wake-транскрипции |
| `ASSISTANT_WAKE_NO_SPEECH` | `0.7` | порог no_speech для wake |
| `ASSISTANT_WAKE_VAD_FILTER` | `true` | VAD при wake-проверке |

Пайплайн: wake → запись фразы до тишины → Whisper STT → Edge TTS → воспроизведение.

## Захват фразы

| Переменная | По умолчанию | Описание |
|---|---|---|
| `ASSISTANT_UTTERANCE_SPEECH_RMS` | `0.008` | RMS-порог речи |
| `ASSISTANT_UTTERANCE_SPEECH_ONSET_SECONDS` | `0.15` | минимальная длительность начала речи |
| `ASSISTANT_UTTERANCE_MIN_SPEECH_SECONDS` | `0.5` | минимальная длительность фразы |
| `ASSISTANT_UTTERANCE_SILENCE_SECONDS` | `1.2` | длительность тишины до завершения |
| `ASSISTANT_UTTERANCE_MAX_SECONDS` | `12.0` | максимальная длительность фразы |

## STT

* Движок: `faster-whisper`
* Язык: `ru`
* Модель: `small`
* Sample rate: `16000` Hz

| Переменная | По умолчанию | Описание |
|---|---|---|
| `ASSISTANT_WHISPER_MODEL` | `small` | модель Whisper |
| `ASSISTANT_WHISPER_BEAM_SIZE` | `8` | beam size для команд |
| `ASSISTANT_WHISPER_NO_SPEECH` | `0.5` | порог no_speech для команд |
| `ASSISTANT_WHISPER_TEMPERATURE` | `0.0` | temperature |
| `ASSISTANT_WHISPER_VAD_FILTER` | `true` | VAD при транскрипции команд |
## TTS

* Движок: `edge-tts` (нейроголос Microsoft)
* Голос по умолчанию: `ru-RU-SvetlanaNeural`

| Переменная | По умолчанию | Описание |
|---|---|---|
| `ASSISTANT_TTS_VOICE` | `ru-RU-SvetlanaNeural` | голос |
| `ASSISTANT_TTS_RATE` | `+0%` | скорость |
| `ASSISTANT_TTS_SAMPLE_RATE` | `24000` | sample rate синтеза |

## Разработка

```bash
make install
make fix
make check
uv run assistant
```
