# Ассистент

Локальный голосовой помощник **Мина**.

## Цели проекта

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
    ├── brain/
    ├── core/
    ├── overlay/
    ├── stt/
    ├── tools/
    ├── tts/
    └── wake/
```

## Быстрый старт

1. Получите Authorization Key в [GigaChat Studio](https://developers.sber.ru/portal/products/gigachat-api)
   (base64 от `Client ID:Client Secret`).
2. Положите ключ в `.env` (см. `.env.example`) или задайте переменную:

```powershell
$env:ASSISTANT_GIGACHAT_CREDENTIALS = "ваш_authorization_key"
```

3. Запуск:

```bash
make install
make run
```

## GigaChat

| Переменная | По умолчанию | Описание |
|---|---|---|
| `ASSISTANT_GIGACHAT_CREDENTIALS` | — | **обязательно**, Authorization Key |
| `ASSISTANT_GIGACHAT_SCOPE` | `GIGACHAT_API_PERS` | версия API (физлица) |
| `ASSISTANT_GIGACHAT_MODEL` | `GigaChat` | модель (Lite) |
| `ASSISTANT_GIGACHAT_VERIFY_SSL` | `false` | проверка SSL (включите `true`, если установлен сертификат НУЦ Минцифры) |
| `ASSISTANT_GIGACHAT_TIMEOUT_SECONDS` | `30` | таймаут запросов |
| `ASSISTANT_GIGACHAT_TEMPERATURE` | `0.3` | температура |
| `ASSISTANT_GIGACHAT_MAX_TOKENS` | `256` | лимит ответа |
| `ASSISTANT_DEFAULT_CITY` | `Москва` | город для погоды по умолчанию |
| `ASSISTANT_DEFAULT_TIMEZONE` | `Europe/Moscow` | часовой пояс |

Tools (без отдельных API-ключей): Open-Meteo (погода), open.er-api.com (курс валют), локальный калькулятор/время, whitelist приложений ПК, выключение ассистента («выключи себя»).

## Аватар

После wake-слова в правом нижнем углу появляется Live2D-аватар (OpenGL, sample-модель Hiyori).
Рот синхронизируется с громкостью TTS, есть авто-моргание и дыхание; после ответа окно скрывается.

Модель лежит в `src/assistant/overlay/assets/live2d/` — можно заменить на свой `.model3.json` (см. `OVERLAY_LIVE2D_MODEL`).
Sample Hiyori: [условия Live2D](https://www.live2d.com/eula/live2d-sample-model-terms_en.html).

## Wake word

Детект через тот же **Whisper** (скользящее окно ~2 с, проверка каждые ~1.5 с).
Тишину отсекает RMS-порог, чтобы не гонять модель впустую.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `ASSISTANT_WAKE_KEYWORD` | `мина` | ключевое слово |
| `ASSISTANT_WAKE_WINDOW_SECONDS` | `2.0` | окно анализа |
| `ASSISTANT_WAKE_HOP_SECONDS` | `1.5` | шаг проверки |
| `ASSISTANT_WAKE_LISTEN_RMS` | `0.012` | минимальный RMS для запуска Whisper |
| `ASSISTANT_WAKE_LISTEN_PEAK` | `0.03` | минимальный peak для запуска Whisper |
| `ASSISTANT_WAKE_LISTEN_SNR` | `3.0` | множитель SNR относительно шума |
| `ASSISTANT_WAKE_BEAM_SIZE` | `2` | beam size для wake-транскрипции |
| `ASSISTANT_WAKE_NO_SPEECH` | `0.7` | порог no_speech для wake |
| `ASSISTANT_WAKE_VAD_FILTER` | `true` | VAD при wake-проверке |

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
