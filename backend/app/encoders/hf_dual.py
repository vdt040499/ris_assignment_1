"""Encoder for the CLIP/SigLIP family loaded via transformers.

One class serves openai/clip, OpenCLIP-LAION, and SigLIP: these three models
differ in checkpoint and padding strategy, not in how they're called.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class HFDualEncoder:
    """Encodes both text and images into the same space.

    The model is lazily loaded on the first encode call, so instantiating this
    class is cheap and contract tests don't need to download hundreds of MB.
    """

    def __init__(self, space: SpaceSpec, device: str, batch_size: int) -> None:
        """Initialize the encoder without loading the model yet.

        :param space: space declaration from the registry (checkpoint, dim, padding, ...).
        :param device: device to run the model on (e.g. ``"cpu"``), taken from ``Settings``.
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
        """Load the model and processor if not already loaded. Idempotent, won't reload."""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._model = AutoModel.from_pretrained(self._space.hf_id).to(self._device).eval()
        self._processor = AutoProcessor.from_pretrained(self._space.hf_id)

    @staticmethod
    def _as_embedding_tensor(output: Any) -> Any:
        """Normalize the output of ``get_text_features``/``get_image_features``.

        On newer ``transformers`` versions (confirmed with 5.14.1), these two
        functions are wrapped by the ``@can_return_tuple`` decorator and return a
        ``BaseModelOutputWithPooling`` (the projected embedding lives in
        ``.pooler_output``) instead of returning the tensor directly, as older
        versions did and as the original code assumed. This function accepts both
        forms so it doesn't depend tightly on a specific ``transformers`` version.

        :param output: raw return value from ``get_text_features``/``get_image_features``.
        :returns: embedding tensor.
        """
        return output.pooler_output if hasattr(output, "pooler_output") else output

    def _finalize(self, vectors: np.ndarray) -> np.ndarray:
        """Check the dim and normalize if the space requires it.

        :param vectors: raw embedding matrix from the model.
        :returns: float32 matrix, L2-normalized if ``space.normalized``.
        :raises ValueError: if the actual dim differs from the dim declared in the registry —
            a dim mismatch that slips into Qdrant would become a very hard bug to trace.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: model returned dim {vectors.shape[1]}, registry declares {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a list of sentences. Sentences longer than the model's token limit are truncated.

        :param texts: list of text sentences.
        :returns: ``(len(texts), dim)`` float32 matrix.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        import torch

        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            inputs = self._processor(
                text=batch,
                padding=self._space.text_padding,
                truncation=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                features = self._as_embedding_tensor(self._model.get_text_features(**inputs))
            chunks.append(features.cpu().numpy())
        return self._finalize(np.concatenate(chunks, axis=0))

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Encode a list of PIL images already in RGB mode.

        :param images: list of ``PIL.Image.Image``.
        :returns: ``(len(images), dim)`` float32 matrix.
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
                features = self._as_embedding_tensor(self._model.get_image_features(**inputs))
            chunks.append(features.cpu().numpy())
        return self._finalize(np.concatenate(chunks, axis=0))
