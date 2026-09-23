"""Giao diện encoder và cache LRU dùng chung."""

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np


class Encoder(Protocol):
    """Mọi encoder trả ma trận ``(n, dim)`` float32, đã normalize nếu space yêu cầu."""

    name: str
    dim: int

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_images(self, images: Sequence[Any]) -> np.ndarray: ...


class LruEncoderCache:
    """Giữ tối đa ``maxsize`` encoder trong RAM, loại bỏ cái ít dùng nhất.

    Việc tạo encoder xảy ra **bên trong** lock. Điều đó khiến một request phải
    chờ trong lúc model đang nạp, nhưng đổi lại hai request đồng thời cho cùng
    một model không nạp hai bản — với model 600MB trên máy 16GB thì tránh nhân
    đôi bộ nhớ quan trọng hơn tránh chờ.
    """

    def __init__(self, maxsize: int, factory: Callable[[str], Encoder]) -> None:
        """Khởi tạo cache.

        :param maxsize: số encoder tối đa được giữ trong RAM cùng lúc.
        :param factory: hàm dựng encoder từ tên space, gọi khi cache miss.
        :raises ValueError: nếu ``maxsize`` nhỏ hơn 1.
        """
        if maxsize < 1:
            raise ValueError(f"maxsize phải >= 1, nhận được {maxsize}")
        self._maxsize = maxsize
        self._factory = factory
        self._items: OrderedDict[str, Encoder] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Encoder:
        """Trả encoder cho ``key``, dựng mới qua factory nếu chưa có trong cache.

        Toàn bộ thao tác (kiểm tra, dựng, chèn, loại bỏ LRU) nằm trong một lock
        duy nhất, nên hai luồng gọi đồng thời với cùng key sẽ không dựng model
        hai lần — luồng thứ hai đợi luồng thứ nhất xong rồi nhận lại đúng
        instance đó.
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
        """Trả danh sách key hiện có, theo thứ tự từ ít dùng gần đây nhất đến mới nhất."""
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        """Xoá toàn bộ encoder khỏi cache."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
