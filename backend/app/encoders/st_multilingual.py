"""Multilingual text tower, distilled to share the same space as CLIP ViT-B/32.

Encodes text only. The image side reuses the clip-b32 collection — this
assumption is guarded by the `test_multilingual_shares_the_clip_space` test.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.errors import ModeNotSupportedError
from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class MultilingualTextEncoder:
    """Text-only encoder for the multilingual CLIP sentence-transformers model."""

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

    def _ensure_loaded(self) -> None:
        """Load the SentenceTransformer if not already loaded. Idempotent."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._space.hf_id, device=self._device)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a list of sentences (multilingual).

        :param texts: list of text sentences.
        :returns: ``(len(texts), dim)`` float32 matrix.
        :raises ValueError: if the actual dim differs from the dim declared in the registry.
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
                f"{self.name}: model returned dim {vectors.shape[1]}, registry declares {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Not supported — this space is only used to encode text.

        :raises ModeNotSupportedError: always, because the image side reuses the
            clip-b32 collection instead of having its own image encoder.
        """
        raise ModeNotSupportedError(
            f"{self.name} only encodes text; the image side uses the clip-b32 collection"
        )
