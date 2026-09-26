"""Non-multimodal baseline: ImageNet features from ResNet-50. Encodes images only."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.errors import ModeNotSupportedError
from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class ResNetEncoder:
    """Image-only encoder using the 2048-dimensional pooled feature from ResNet-50."""

    def __init__(self, space: SpaceSpec, device: str, batch_size: int) -> None:
        """Initialize the encoder without loading the model yet.

        :param space: space declaration from the registry.
        :param device: device to run the model on, taken from ``Settings``.
        :param batch_size: number of samples processed per batch, taken from ``Settings``.
        """
        self.name = space.name
        self.dim = space.dim
        self._space = space
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        """Load the model and image processor if not already loaded. Idempotent."""
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        self._model = AutoModel.from_pretrained(self._space.hf_id).to(self._device).eval()
        self._processor = AutoImageProcessor.from_pretrained(self._space.hf_id)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Not supported — ResNet-50 is a baseline used only for image-to-image.

        :raises ModeNotSupportedError: always.
        """
        raise ModeNotSupportedError(
            f"{self.name} is a baseline used only for image-to-image; it does not encode text"
        )

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Extract the 2048-dimensional pooled feature, flattened from shape (n, 2048, 1, 1).

        :param images: list of ``PIL.Image.Image``.
        :returns: ``(len(images), dim)`` float32 matrix.
        :raises ValueError: if the actual dim differs from the dim declared in the registry.
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
                f"{self.name}: model returned dim {vectors.shape[1]}, registry declares {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors
