"""Baseline không multimodal: feature ImageNet của ResNet-50. Chỉ encode ảnh."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.errors import ModeNotSupportedError
from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class ResNetEncoder:
    """Image-only encoder dùng pooled feature 2048 chiều của ResNet-50."""

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
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        """Nạp model và image processor nếu chưa nạp. Idempotent."""
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        self._model = AutoModel.from_pretrained(self._space.hf_id).to(self._device).eval()
        self._processor = AutoImageProcessor.from_pretrained(self._space.hf_id)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Không hỗ trợ — ResNet-50 là baseline chỉ dùng cho ảnh→ảnh.

        :raises ModeNotSupportedError: luôn luôn.
        """
        raise ModeNotSupportedError(
            f"{self.name} là baseline chỉ dùng cho ảnh→ảnh, không encode text"
        )

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Lấy pooled feature 2048 chiều, dàn phẳng từ shape (n, 2048, 1, 1).

        :param images: danh sách ``PIL.Image.Image``.
        :returns: ma trận ``(len(images), dim)`` float32.
        :raises ValueError: nếu dim thực tế khác dim khai báo trong registry.
        """
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        import torch

        chunks: list[np.ndarray] = []
        for start in range(0, len(images), self._batch_size):
            batch = list(images[start : start + self._batch_size])
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            pooled = outputs.pooler_output
            chunks.append(pooled.reshape(pooled.shape[0], -1).cpu().numpy())
        vectors = np.concatenate(chunks, axis=0).astype(np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: model trả dim {vectors.shape[1]}, registry khai {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors
