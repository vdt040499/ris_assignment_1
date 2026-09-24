"""Điều phối tìm kiếm.

Module này không biết CLIP hay SigLIP là gì: nó hỏi registry space nào dùng
backend nào, gọi encoder tương ứng, rồi trả về một hình dạng response duy nhất
cho cả hai chiều truy vấn.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.corpus import Corpus
from app.encoders import get_encoder
from app.encoders.bm25 import Bm25Retriever
from app.errors import (
    BadRequestError,
    IndexNotBuiltError,
    ModeNotSupportedError,
    ValidationRangeError,
)
from app.imageio import load_corpus_image
from app.registry import (
    BACKEND_BM25,
    MODE_IMAGE2IMAGE,
    MODE_IMAGE2TEXT,
    MODE_TEXT2IMAGE,
    SPACES,
    SpaceSpec,
    collection_name,
    get_space,
)
from app.schemas import (
    Filters,
    SearchResponse,
    SearchResultItem,
    SpaceInfo,
    TextSearchRequest,
)
from app.vectordb import Hit, VectorStore, build_filter

THUMB_URL_PREFIX = "/thumbs"


@dataclass
class SearchService:
    """Một instance dùng chung cho cả process API.

    :param encoder_factory: điểm tiêm phụ thuộc — test truyền encoder giả vào
        đây để kiểm tra điều phối mà không tải model thật.
    """

    settings: Settings
    store: VectorStore
    corpus: Corpus
    encoder_factory: Callable[[str], Any] = get_encoder
    _bm25: Bm25Retriever | None = field(default=None, init=False, repr=False)

    def _resolve_k(self, k: int | None) -> int:
        """Áp mặc định và trần cấu hình cho top-k.

        :raises ValidationRangeError: nếu k vượt ``MAX_TOP_K``.
        """
        resolved = self.settings.top_k_default if k is None else k
        if resolved > self.settings.max_top_k:
            raise ValidationRangeError(
                f"k={resolved} vượt giới hạn MAX_TOP_K={self.settings.max_top_k}"
            )
        return resolved

    @staticmethod
    def _require_mode(space: SpaceSpec, mode: str) -> None:
        """:raises ModeNotSupportedError: nếu space không làm được chiều này."""
        if mode not in space.modes:
            raise ModeNotSupportedError(
                f"Space '{space.name}' không hỗ trợ '{mode}'. "
                f"Hỗ trợ: {', '.join(space.modes)}"
            )

    def _allowed_ids(self, filters: Filters) -> set[int] | None:
        """Tập image_id thoả filter, tính từ corpus.

        Chỉ dùng cho backend BM25 — nó không có payload filter như Qdrant, nên
        việc lọc phải làm ở phía ta.
        """
        if not filters.categories and not filters.supercategories:
            return None
        wanted_cats = set(filters.categories)
        wanted_supers = set(filters.supercategories)
        return {
            record.image_id
            for record in self.corpus.records
            if (not wanted_cats or wanted_cats & set(record.categories))
            and (not wanted_supers or wanted_supers & set(record.supercategories))
        }

    def _bm25_retriever(self) -> Bm25Retriever:
        """Nạp lười index BM25.

        :raises IndexNotBuiltError: nếu chưa build, kèm đúng lệnh cần chạy.
        """
        if self._bm25 is None:
            try:
                self._bm25 = Bm25Retriever.load(self.settings.bm25_path, self.corpus)
            except FileNotFoundError as exc:
                raise IndexNotBuiltError(
                    "Chưa build index BM25. Chạy: python tasks.py build --space bm25-cap"
                ) from exc
        return self._bm25

    def _require_index(self, space: SpaceSpec, target: str) -> tuple[str, int]:
        """Trả về ``(tên collection, số point)``.

        :raises IndexNotBuiltError: nếu collection rỗng hoặc chưa tồn tại.
        """
        name = collection_name(space, self.settings, target=target)
        count = self.store.count(name)
        if count == 0:
            raise IndexNotBuiltError(
                f"Space '{space.name}' (target={target}) chưa có index. "
                f"Chạy: python tasks.py build --space {space.name}"
            )
        return name, count

    def _items(self, hits: list[Hit], target: str) -> list[SearchResultItem]:
        """Đổi Hit thành item response. Một hình dạng duy nhất cho mọi chiều."""
        items: list[SearchResultItem] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload
            items.append(
                SearchResultItem(
                    image_id=int(payload["image_id"]),
                    file_name=payload["file_name"],
                    thumb_url=f"{THUMB_URL_PREFIX}/{payload['file_name']}",
                    score=hit.score,
                    captions=payload.get("captions", []),
                    categories=payload.get("categories", []),
                    rank=rank,
                    matched_caption=payload.get("caption") if target == "caption" else None,
                    caption_index=(
                        payload.get("caption_index") if target == "caption" else None
                    ),
                )
            )
        return items

    def search_text(self, request: TextSearchRequest) -> SearchResponse:
        """Tìm bằng câu chữ. Trả top-k ảnh, hoặc top-k caption nếu target='caption'."""
        started = time.perf_counter()
        space = get_space(request.space)
        mode = MODE_IMAGE2TEXT if request.target == "caption" else MODE_TEXT2IMAGE
        self._require_mode(space, mode)
        k = self._resolve_k(request.k)
        text = (
            request.prompt_template.format(request.query)
            if request.prompt_template
            else request.query
        )

        if space.backend == BACKEND_BM25:
            encode_ms = 0.0
            search_started = time.perf_counter()
            hits = self._bm25_retriever().search(
                text, k, allowed_ids=self._allowed_ids(request.filters)
            )
            search_ms = (time.perf_counter() - search_started) * 1000
            total = len(self.corpus)
        else:
            name, total = self._require_index(space, request.target)
            encode_started = time.perf_counter()
            vector = self.encoder_factory(space.name).encode_texts([text])[0]
            encode_ms = (time.perf_counter() - encode_started) * 1000
            search_started = time.perf_counter()
            hits = self.store.search(
                name,
                vector,
                k=k,
                exact=request.exact,
                hnsw_ef=request.hnsw_ef or self.settings.hnsw_ef_default,
                query_filter=build_filter(
                    request.filters.categories, request.filters.supercategories
                ),
            )
            search_ms = (time.perf_counter() - search_started) * 1000

        return SearchResponse(
            results=self._items(hits, request.target),
            latency_ms=(time.perf_counter() - started) * 1000,
            encode_ms=encode_ms,
            search_ms=search_ms,
            space=space.name,
            exact=request.exact,
            total_searched=total,
            query_echo=text,
        )

    def search_image(
        self,
        *,
        space: str,
        k: int | None,
        exact: bool,
        hnsw_ef: int | None,
        filters: Filters,
        image: Any = None,
        image_id: int | None = None,
    ) -> SearchResponse:
        """Tìm bằng ảnh: ảnh upload, hoặc một ảnh đã có trong corpus.

        Khi tìm bằng ``image_id``, ảnh query bị loại khỏi kết quả — nó luôn tự
        khớp với chính mình ở hạng 1, vừa chiếm một suất vô nghĩa trên UI vừa
        làm phồng P@k khi đo.

        :raises BadRequestError: nếu không có đúng một trong ``image``/``image_id``,
            hoặc ``image_id`` không có trong corpus.
        """
        started = time.perf_counter()
        if (image is None) == (image_id is None):
            raise BadRequestError("Cần đúng một trong hai: file ảnh hoặc image_id")

        spec = get_space(space)
        self._require_mode(spec, MODE_IMAGE2IMAGE)
        k = self._resolve_k(k)

        if image_id is not None:
            record = self.corpus.by_image_id.get(image_id)
            if record is None:
                raise BadRequestError(f"image_id {image_id} không có trong corpus")
            image = load_corpus_image(
                self.settings.images_dir / record.file_name, self.settings
            )

        name, total = self._require_index(spec, "image")
        encode_started = time.perf_counter()
        vector = self.encoder_factory(spec.name).encode_images([image])[0]
        encode_ms = (time.perf_counter() - encode_started) * 1000

        fetch = k + 1 if image_id is not None else k
        search_started = time.perf_counter()
        hits = self.store.search(
            name,
            vector,
            k=fetch,
            exact=exact,
            hnsw_ef=hnsw_ef or self.settings.hnsw_ef_default,
            query_filter=build_filter(filters.categories, filters.supercategories),
        )
        search_ms = (time.perf_counter() - search_started) * 1000
        if image_id is not None:
            hits = [h for h in hits if int(h.payload["image_id"]) != image_id][:k]

        return SearchResponse(
            results=self._items(hits, "image"),
            latency_ms=(time.perf_counter() - started) * 1000,
            encode_ms=encode_ms,
            search_ms=search_ms,
            space=spec.name,
            exact=exact,
            total_searched=total,
            query_echo=f"image_id={image_id}" if image_id is not None else "upload",
        )

    def space_infos(self) -> list[SpaceInfo]:
        """Liệt kê mọi space kèm trạng thái index, để frontend dựng dropdown."""
        infos: list[SpaceInfo] = []
        for space in SPACES.values():
            if space.backend == BACKEND_BM25:
                n_points = len(self.corpus) if self.settings.bm25_path.exists() else 0
            else:
                n_points = self.store.count(
                    collection_name(space, self.settings, target="image")
                )
            infos.append(
                SpaceInfo(
                    name=space.name,
                    hf_id=space.hf_id,
                    dim=space.dim,
                    backend=space.backend,
                    modes=list(space.modes),
                    languages=list(space.languages),
                    ready=n_points > 0,
                    n_points=n_points,
                    note=space.note,
                )
            )
        return infos
