from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import sounddevice as sd

from assistant.audio.exceptions import AudioDeviceError


@dataclass(frozen=True, slots=True)
class AudioDevice:
    index: int
    name: str
    hostapi: int
    hostapi_name: str
    input_channels: int
    output_channels: int
    sample_rate: int


class AudioDeviceCatalog:
    def list_input_devices(self) -> list[AudioDevice]:
        return self._list_devices(input_only=True)

    def list_output_devices(self) -> list[AudioDevice]:
        return self._list_devices(input_only=False)

    def get_default_input_device(self) -> AudioDevice | None:
        return self._find_device(_default_device_index(0), self.list_input_devices())

    def get_default_output_device(self) -> AudioDevice | None:
        return self._find_device(_default_device_index(1), self.list_output_devices())

    def get_device(self, index: int) -> AudioDevice:
        devices = {device.index: device for device in self._list_devices(input_only=None)}
        device = devices.get(index)

        if device is None:
            raise AudioDeviceError(f"Audio device not found: index={index}")

        return device

    def validate_input_device(self, index: int) -> AudioDevice:
        device = self.get_device(index)

        if device.input_channels <= 0:
            raise AudioDeviceError(f"Device is not an input device: index={index}, name={device.name!r}")

        return device

    def validate_output_device(self, index: int) -> AudioDevice:
        device = self.get_device(index)

        if device.output_channels <= 0:
            raise AudioDeviceError(f"Device is not an output device: index={index}, name={device.name!r}")

        return device

    def _list_devices(self, *, input_only: bool | None) -> list[AudioDevice]:
        hostapis = _as_sequence(sd.query_hostapis(), what="hostapis")
        devices: list[AudioDevice] = []
        seen: set[tuple[str, int]] = set()

        for index, raw in enumerate(_as_sequence(sd.query_devices(), what="devices")):
            info = _as_mapping(raw, what=f"device[{index}]")
            input_channels = _as_int(info.get("max_input_channels"), field="max_input_channels")
            output_channels = _as_int(info.get("max_output_channels"), field="max_output_channels")

            if input_only is True and input_channels <= 0:
                continue

            if input_only is False and output_channels <= 0:
                continue

            hostapi = _as_int(info.get("hostapi"), field="hostapi")
            name = _as_str(info.get("name"), field="name")
            key = (name, hostapi)

            if key in seen:
                continue

            seen.add(key)
            hostapi_info = _as_mapping(hostapis[hostapi], what=f"hostapi[{hostapi}]")

            devices.append(
                AudioDevice(
                    index=index,
                    name=name,
                    hostapi=hostapi,
                    hostapi_name=_as_str(hostapi_info.get("name"), field="hostapi.name"),
                    input_channels=input_channels,
                    output_channels=output_channels,
                    sample_rate=_as_int(info.get("default_samplerate"), field="default_samplerate"),
                )
            )

        return devices

    @staticmethod
    def _find_device(
        index: int | None,
        devices: list[AudioDevice],
    ) -> AudioDevice | None:
        if index is None:
            return None

        return next(
            (device for device in devices if device.index == index),
            None,
        )


def _default_device_index(position: int) -> int | None:
    raw: object = sd.default.device

    if not isinstance(raw, Sequence) or len(raw) <= position:
        return None

    value = raw[position]
    if value is None:
        return None

    return int(value)


def _as_sequence(value: object, *, what: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AudioDeviceError(f"Unexpected {what} payload type: {type(value)!r}")

    return value


def _as_mapping(value: object, *, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AudioDeviceError(f"Unexpected {what} payload type: {type(value)!r}")

    return value


def _as_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioDeviceError(f"Invalid device field {field!r}: {value!r}")

    return int(value)


def _as_str(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise AudioDeviceError(f"Invalid device field {field!r}: {value!r}")

    return value
