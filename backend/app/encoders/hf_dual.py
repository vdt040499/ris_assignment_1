"""Encoder cho họ CLIP/SigLIP tải qua transformers.

Một lớp phục vụ cả openai/clip, OpenCLIP-LAION và SigLIP: ba model này khác
nhau ở checkpoint và chiến lược padding, không khác nhau ở cách gọi.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class HFDualEncoder:
    """Encode cả text và ảnh vào cùng một không gian.

    Model được nạp lười ở lần encode đầu tiên, nên khởi tạo lớp này rẻ và các
    test kiểm tra hợp đồng không phải tải hàng trăm MB.
    """

    def __init__(self, space: SpaceSpec, device: str, batch_size: int) -> None:
        """Khởi tạo encoder mà chưa nạp model.

        :param space: khai báo space từ registry (checkpoint, dim, padding, ...).
        :param device: thiết bị chạy model (vd ``"cpu"``), lấy từ ``Settings``.
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
        """Nạp model và processor nếu chưa nạp. Idempotent, không nạp lại."""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._model = AutoModel.from_pretrained(self._space.hf_id).to(self._device).eval()
        self._processor = AutoProcessor.from_pretrained(self._space.hf_id)

    @staticmethod
    def _as_embedding_tensor(output: Any) -> Any:
        """Chuẩn hoá đầu ra của ``get_text_features``/``get_image_features``.

        Trên ``transformers`` bản mới (đã xác nhận với 5.14.1), hai hàm này
        được decorator ``@can_return_tuple`` bọc lại và trả về một
        ``BaseModelOutputWithPooling`` (embedding đã project nằm ở
        ``.pooler_output``) thay vì trả thẳng tensor như các bản cũ hơn mà
        phần code gốc giả định. Hàm này chấp nhận cả hai dạng để không phụ
        thuộc chặt vào một phiên bản ``transformers`` cụ thể.

        :param output: giá trị trả về thô từ ``get_text_features``/``get_image_features``.
        :returns: tensor embedding.
        """
        return output.pooler_output if hasattr(output, "pooler_output") else output

    def _finalize(self, vectors: np.ndarray) -> np.ndarray:
        """Kiểm tra dim rồi normalize nếu space yêu cầu.

        :param vectors: ma trận embedding thô từ model.
        :returns: ma trận float32, đã L2-normalize nếu ``space.normalized``.
        :raises ValueError: nếu dim thực tế khác dim khai báo trong registry —
            lệch dim mà lọt vào Qdrant sẽ thành lỗi rất khó truy nguyên.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: model trả dim {vectors.shape[1]}, registry khai {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode danh sách câu. Câu dài hơn giới hạn token của model bị cắt bớt.

        :param texts: danh sách câu văn bản.
        :returns: ma trận ``(len(texts), dim)`` float32.
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
        """Encode danh sách ảnh PIL đã ở chế độ RGB.

        :param images: danh sách ``PIL.Image.Image``.
        :returns: ma trận ``(len(images), dim)`` float32.
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
