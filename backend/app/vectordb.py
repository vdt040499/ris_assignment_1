"""Wrapper quanh Qdrant.

Đây là module duy nhất trong hệ import ``qdrant_client``. Mọi thành phần khác
làm việc với ``Hit`` và ``build_filter``, nên đổi vector DB về sau chỉ phải sửa
một file.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import httpx
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


def _is_down(exc: Exception) -> bool:
    """``True`` nếu ``exc`` thật sự là Qdrant không phản hồi được.

    ``ResponseHandlingException`` và ``OSError`` luôn là lỗi transport (không
    tới được server). ``UnexpectedResponse`` thì mơ hồ hơn: nó bọc **mọi**
    status code không phải 2xx, kể cả 4xx — tức Qdrant *đã* trả lời nhưng từ
    chối chính request đó (vd ``limit`` <= 0), không phải downtime. Coi 4xx là
    "down" khiến finding I5 xảy ra: ``k<=0`` bị báo nhầm thành "Qdrant không
    chạy" (503) trong khi Qdrant vẫn sống và chỉ đang làm đúng việc của nó là
    từ chối một tham số sai. Chỉ 5xx (hoặc status không xác định được) mới
    thật sự là "down".
    """
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is None or exc.status_code >= 500
    return True


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
            # qdrant-client tự tắt HTTP keep-alive khi host là localhost/127.0.0.1
            # (xem QdrantRemote.__init__), khiến mỗi lần .search() mở một kết nối
            # TCP mới. Trên Windows, hàng chục nghìn query tuần tự của CLI evaluate
            # có thể cạn cổng ephemeral (WinError 10048) vì TIME_WAIT dồn nhanh hơn
            # tốc độ hệ điều hành thu hồi cổng. Truyền limits tường minh để ghi đè
            # mặc định đó và tái sử dụng kết nối.
            #
            # Đo thực tế trên máy chạy dự án này: gọi qua tên "localhost" tốn
            # ~100ms/query (kể cả sau khi bật lại keep-alive ở trên), trong khi
            # cùng request gửi thẳng tới "127.0.0.1" chỉ còn ~50ms — chênh lệch
            # đến từ cách Windows phân giải "localhost" cho từng kết nối, không
            # liên quan gì tới Qdrant hay tới thuật toán tìm kiếm (exact và ANN
            # đo được thời gian như nhau, loại trừ khả năng do compute). Thay
            # "localhost" bằng "127.0.0.1" chỉ ở URL thực sự dùng để kết nối —
            # settings.qdrant_url (hiển thị trong thông báo lỗi, v.v.) giữ nguyên.
            connect_url = settings.qdrant_url.replace("localhost", "127.0.0.1")
            self._client = QdrantClient(
                url=connect_url,
                timeout=settings.qdrant_timeout_s,
                limits=httpx.Limits(
                    max_connections=settings.qdrant_pool_size,
                    max_keepalive_connections=settings.qdrant_pool_size,
                ),
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
        except CONNECTION_ERRORS as exc:
            if not _is_down(exc):
                raise
            return "down"
        return "ok"

    def collection_exists(self, name: str) -> bool:
        try:
            return self._client.collection_exists(name)
        except CONNECTION_ERRORS as exc:
            if not _is_down(exc):
                raise
            raise VectorStoreDownError(str(exc)) from exc

    def count(self, name: str) -> int:
        """Số point trong collection; 0 nếu collection chưa tồn tại."""
        if not self.collection_exists(name):
            return 0
        return self._client.count(name, exact=True).count

    def indexing_status(self, name: str) -> tuple[int, int]:
        """Trả ``(points_count, indexed_vectors_count)`` của collection.

        Dùng để phát hiện đúng lỗi ở finding C1: nếu ``indexed_vectors_count``
        không bằng ``points_count`` thì HNSW chưa build xong (hoặc chưa build
        gì cả), nên mọi truy vấn "ANN" thực chất đang full-scan.

        :raises VectorStoreDownError: nếu không gọi được Qdrant.
        """
        try:
            info = self._client.get_collection(name)
        except CONNECTION_ERRORS as exc:
            if not _is_down(exc):
                raise
            raise VectorStoreDownError(str(exc)) from exc
        return info.points_count or 0, info.indexed_vectors_count or 0

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
            # Qdrant chỉ build HNSW khi một segment vượt `indexing_threshold` (mặc
            # định 20000 KB). 5.000 vector x 512 float32 (~10MB) chia trên nhiều
            # segment không bao giờ chạm ngưỡng đó, nên HNSW không bao giờ được
            # build và mọi truy vấn — kể cả truy vấn "ANN" — thực chất là full-scan
            # (exact đội lốt ANN, phát hiện ở finding C1 của final review).
            #
            # LƯU Ý QUAN TRỌNG (khác với hướng dẫn ban đầu của finding C1):
            # `indexing_threshold=0` KHÔNG có nghĩa "luôn index" — theo đúng
            # docstring của `OptimizersConfigDiff.indexing_threshold` trong
            # qdrant-client ("To disable vector indexing, set to 0"), giá trị 0
            # là sentinel **tắt hẳn** indexing, tác dụng ngược hoàn toàn với ý
            # định. Xác nhận thực nghiệm trên server thật: đặt 0 rồi restart
            # container, chờ hơn 10 phút, `indexed_vectors_count` vẫn y nguyên 0
            # và `/telemetry` báo `optimizations.count = 0` cho mọi collection —
            # tức optimizer không hề chạy, đúng như "disabled". Dùng một ngưỡng
            # dương rất nhỏ (1 KB) thay vì 0: theo đúng đơn vị của field này
            # ("1kB = 1 vector kích thước 256"), 1 KB đã nhỏ hơn một vector 512
            # chiều, nên mọi segment có dữ liệu đều vượt ngưỡng ngay lập tức —
            # hiệu quả tương đương "luôn index" mà không rơi vào sentinel tắt.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
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
            if not _is_down(exc):
                # 4xx: Qdrant sống và đã trả lời, chỉ từ chối chính request này
                # (vd k<=0 lọt qua tầng FastAPI, xem finding I5) — không phải
                # downtime, nên không được báo "docker compose up -d qdrant".
                raise
            raise VectorStoreDownError(
                f"Không truy vấn được Qdrant ({self._settings.qdrant_url}). "
                "Chạy: docker compose up -d qdrant"
            ) from exc
        return [
            Hit(id=int(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]
