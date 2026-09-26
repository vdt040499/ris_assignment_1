"""Encoder interface and shared LRU cache."""

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np


class Encoder(Protocol):
    """Every encoder returns an ``(n, dim)`` float32 matrix, normalized if the space requires it."""

    name: str
    dim: int

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_images(self, images: Sequence[Any]) -> np.ndarray: ...


class LruEncoderCache:
    """Keeps at most ``maxsize`` encoders in RAM, evicting the least recently used.

    Encoder creation happens **inside** the lock. That means a request has to
    wait while a model is loading, but in exchange two concurrent requests for
    the same model won't load two copies — with a 600MB model on a 16GB machine,
    avoiding doubled memory use matters more than avoiding the wait.
    """

    def __init__(self, maxsize: int, factory: Callable[[str], Encoder]) -> None:
        """Initialize the cache.

        :param maxsize: maximum number of encoders kept in RAM at once.
        :param factory: function that builds an encoder from a space name, called on a cache miss.
        :raises ValueError: if ``maxsize`` is less than 1.
        """
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._factory = factory
        self._items: OrderedDict[str, Encoder] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Encoder:
        """Return the encoder for ``key``, building a new one via the factory if not cached.

        The entire operation (check, build, insert, evict LRU) happens inside a
        single lock, so two threads calling concurrently with the same key won't
        build the model twice — the second thread waits for the first to finish
        and then receives that same instance.
        """
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            encoder = self._factory(key)
            self._items[key] = encoder
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)
            return encoder

    def keys(self) -> list[str]:
        """Return the list of current keys, ordered from least recently used to most recently used."""
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        """Remove all encoders from the cache."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
