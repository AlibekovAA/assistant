from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"RingBuffer capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._end = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._size = 0
        self._end = 0

    def extend(self, samples: NDArray[np.float32]) -> None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return

        if data.size >= self._capacity:
            self._buffer[:] = data[-self._capacity :]
            self._size = self._capacity
            self._end = 0
            return

        first = min(data.size, self._capacity - self._end)
        self._buffer[self._end : self._end + first] = data[:first]

        remaining = data.size - first
        if remaining:
            self._buffer[:remaining] = data[first:]

        self._end = (self._end + data.size) % self._capacity
        self._size = min(self._capacity, self._size + data.size)

    def snapshot(self) -> NDArray[np.float32]:
        if self._size == 0:
            return np.empty(0, dtype=np.float32)

        if self._size < self._capacity:
            return np.ascontiguousarray(self._buffer[: self._size])

        start = self._end
        return np.ascontiguousarray(np.concatenate((self._buffer[start:], self._buffer[:start])))
