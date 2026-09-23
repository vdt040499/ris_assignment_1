"""Wrapper quanh Qdrant.

Đây là module duy nhất trong hệ import ``qdrant_client``. Mọi thành phần khác
làm việc với ``Hit`` và ``build_filter``, nên đổi vector DB về sau chỉ phải sửa
một file.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import Settings
from app.errors import VectorStoreDownError

DISTANCE_MAP = {
    "Cosine": models.Distance.COSINE,
    "Dot": models.Distance.DOT,
}

CONNECTION_ERRORS = (ResponseHandlingException, UnexpectedResponse, OSError)


@dataclass(frozen=True)
class Hit:
    """Một kết quả truy vấn: id point, điểm tương đồng, và payload kèm theo."""

    id: int
    score: float
    payload: dict


def build_filter(
    categories: Sequence[str] | None, supercategories: Sequence[str] | None
) -> models.Filter | None:
    """Dựng filter payload Qdrant từ danh sách category người dùng chọn.

    Nhiều giá trị trong cùng một field là quan hệ HOẶC (``MatchAny``); giữa hai
    field là quan hệ VÀ. Trả ``None`` khi không có filter nào, để người gọi
    truyền thẳng vào Qdrant mà không cần phân nhánh.
    """
    conditions: list[models.FieldCondition] = []
    if categories:
        conditions.append(
            models.FieldCondition(
                key="categories", match=models.MatchAny(any=list(categories))
            )
        )
    if supercategories:
        conditions.append(
            models.FieldCondition(
                key="supercategories", match=models.MatchAny(any=list(supercategories))
            )
        )
    return models.Filter(must=conditions) if conditions else None


class VectorStore:
    """Truy cập Qdrant ở chế độ server (Docker) hoặc embedded (không Docker).

    Chế độ embedded luôn chạy brute-force và bỏ qua cấu hình HNSW, nên nó chỉ
    dùng để demo và để test — không dùng để sinh số liệu cho trục ANN vs exact.
    """

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._settings = settings
        if client is not None:
            self._client = client
        elif settings.qdrant_mode == "server":
            self._client = QdrantClient(
                url=settings.qdrant_url, timeout=settings.qdrant_timeout_s
            )
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(settings.qdrant_path))

    @property
    def mode(self) -> str:
        return self._settings.qdrant_mode

    def health(self) -> str:
        """Trả ``"ok"`` nếu gọi được Qdrant, ``"down"`` nếu không."""
        try:
            self._client.get_collections()
        except CONNECTION_ERRORS:
            return "down"
        return "ok"

    def collection_exists(self, name: str) -> bool:
        try:
            return self._client.collection_exists(name)
        except CONNECTION_ERRORS as exc:
            raise VectorStoreDownError(str(exc)) from exc

    def count(self, name: str) -> int:
        """Số point trong collection; 0 nếu collection chưa tồn tại."""
        if not self.collection_exists(name):
            return 0
        return self._client.count(name, exact=True).count

    def ensure_collection(
        self, name: str, dim: int, distance: str, recreate: bool = False
    ) -> bool:
        """Tạo collection nếu chưa có. Trả True nếu lần này thật sự tạo.

        :raises ValueError: nếu tên distance không nằm trong DISTANCE_MAP.
        """
        if distance not in DISTANCE_MAP:
            raise ValueError(
                f"distance '{distance}' không hỗ trợ; hợp lệ: {sorted(DISTANCE_MAP)}"
            )
        exists = self.collection_exists(name)
        if exists and not recreate:
            return False
        if exists:
            self._client.delete_collection(name)
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=DISTANCE_MAP[distance]),
        )
        return True

    def create_payload_indexes(self, name: str, fields: Sequence[str]) -> None:
        """Tạo payload index keyword cho các field dùng để filter."""
        for field in fields:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert(
        self,
        name: str,
        ids: Sequence[int],
        vectors: np.ndarray,
        payloads: Sequence[dict],
        batch_size: int,
    ) -> int:
        """Upsert theo batch. Trả về số point đã gửi.

        :raises ValueError: nếu số id, số vector và số payload không khớp nhau.
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"lệch số lượng: {len(ids)} id, {len(vectors)} vector, "
                f"{len(payloads)} payload"
            )
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            points = [
                models.PointStruct(id=int(pid), vector=vec.tolist(), payload=payload)
                for pid, vec, payload in zip(
                    ids[start:stop], vectors[start:stop], payloads[start:stop]
                )
            ]
            self._client.upsert(collection_name=name, points=points, wait=True)
        return len(ids)

    def search(
        self,
        name: str,
        vector: np.ndarray,
        k: int,
        exact: bool = True,
        hnsw_ef: int | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[Hit]:
        """Truy vấn top-k.

        :param exact: ``True`` bỏ qua HNSW và quét toàn bộ — đây là chuẩn vàng
            cho số liệu trong báo cáo. ``False`` dùng HNSW với ``hnsw_ef``.
        :raises VectorStoreDownError: nếu không gọi được Qdrant.
        """
        params = models.SearchParams(exact=exact, hnsw_ef=None if exact else hnsw_ef)
        try:
            response = self._client.query_points(
                collection_name=name,
                query=vector.tolist(),
                limit=k,
                query_filter=query_filter,
                search_params=params,
                with_payload=True,
            )
        except CONNECTION_ERRORS as exc:
            raise VectorStoreDownError(
                f"Không truy vấn được Qdrant ({self._settings.qdrant_url}). "
                "Chạy: docker compose up -d qdrant"
            ) from exc
        return [
            Hit(id=int(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]
