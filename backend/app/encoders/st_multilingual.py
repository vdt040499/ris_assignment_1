"""Text tower đa ngôn ngữ, distill để nằm cùng không gian với CLIP ViT-B/32.

Chỉ encode text. Phía ảnh dùng lại collection của clip-b32 — giả định này được
test `test_multilingual_shares_the_clip_space` canh giữ.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.errors import ModeNotSupportedError
from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class MultilingualTextEncoder:
    """Text-only encoder cho model sentence-transformers CLIP đa ngôn ngữ."""

    def __init__(self, space: SpaceSpec, device: str, batch_size: int) -> None:
        """Khởi tạo encoder mà chưa nạp model.

        :param space: khai báo space từ registry.
        :param device: thiết bị chạy model, lấy từ ``Settings``.
        :param batch_size: số mẫu xử lý mỗi lượt, lấy từ ``Settings``.
        """
        self.name = space.name
        self.dim = space.dim
        self._space = space
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        """Nạp SentenceTransformer nếu chưa nạp. Idempotent."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._space.hf_id, device=self._device)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode danh sách câu (đa ngôn ngữ).

        :param texts: danh sách câu văn bản.
        :returns: ma trận ``(len(texts), dim)`` float32.
        :raises ValueError: nếu dim thực tế khác dim khai báo trong registry.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, convert_to_numpy=True
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: model trả dim {vectors.shape[1]}, registry khai {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Không hỗ trợ — space này chỉ dùng để encode text.

        :raises ModeNotSupportedError: luôn luôn, vì phía ảnh dùng lại
            collection của clip-b32 thay vì có encoder ảnh riêng.
        """
        raise ModeNotSupportedError(
            f"{self.name} chỉ encode text; phía ảnh dùng collection của clip-b32"
        )
