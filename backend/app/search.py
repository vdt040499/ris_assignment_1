"""Search orchestration.

This module doesn't know what CLIP or SigLIP are: it asks the registry which
backend a space uses, calls the corresponding encoder, then returns a single
unified response shape for both query directions.
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
    """A single instance shared across the whole API process.

    :param encoder_factory: dependency-injection point — tests pass a fake
        encoder here to test the orchestration without loading a real model.
    """

    settings: Settings
    store: VectorStore
    corpus: Corpus
    encoder_factory: Callable[[str], Any] = get_encoder
    _bm25: Bm25Retriever | None = field(default=None, init=False, repr=False)

    def _resolve_k(self, k: int | None) -> int:
        """Apply the configured default and cap for top-k.

        :raises ValidationRangeError: if k <= 0 or exceeds ``MAX_TOP_K``.
        """
        resolved = self.settings.top_k_default if k is None else k
        # The HTTP layer (`Form(..., ge=1)` for /search/image, `Field(ge=1)`
        # for TextSearchRequest) already blocks k<=0 before it gets here, but
        # this service is also called directly (CLI, tests) without going
        # through that layer — re-check here so k<=0 always becomes a 422
        # (ValidationRangeError), never slipping down to Qdrant and being
        # misread as "Qdrant is down" (finding I5).
        if resolved <= 0:
            raise ValidationRangeError(f"k={resolved} must be >= 1")
        if resolved > self.settings.max_top_k:
            raise ValidationRangeError(
                f"k={resolved} exceeds the MAX_TOP_K={self.settings.max_top_k} limit"
            )
        return resolved

    @staticmethod
    def _require_mode(space: SpaceSpec, mode: str) -> None:
        """:raises ModeNotSupportedError: if the space can't handle this direction."""
        if mode not in space.modes:
            raise ModeNotSupportedError(
                f"Space '{space.name}' does not support '{mode}'. "
                f"Supported: {', '.join(space.modes)}"
            )

    def _allowed_ids(self, filters: Filters) -> set[int] | None:
        """Set of image_ids satisfying the filter, computed from the corpus.

        Only used for the BM25 backend — it doesn't have a payload filter like
        Qdrant, so the filtering has to be done on our side.
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
        """Lazily load the BM25 index.

        :raises IndexNotBuiltError: if it hasn't been built yet, including the exact command to run.
        """
        if self._bm25 is None:
            try:
                self._bm25 = Bm25Retriever.load(self.settings.bm25_path, self.corpus)
            except FileNotFoundError as exc:
                raise IndexNotBuiltError(
                    "BM25 index not built yet. Run: python tasks.py build --space bm25-cap"
                ) from exc
        return self._bm25

    def _require_index(self, space: SpaceSpec, target: str) -> tuple[str, int]:
        """Returns ``(collection name, point count)``.

        :raises IndexNotBuiltError: if the collection is empty or doesn't exist yet.
        """
        name = collection_name(space, self.settings, target=target)
        count = self.store.count(name)
        if count == 0:
            raise IndexNotBuiltError(
                f"Space '{space.name}' (target={target}) has no index yet. "
                f"Run: python tasks.py build --space {space.name}"
            )
        return name, count

    def _items(self, hits: list[Hit], target: str) -> list[SearchResultItem]:
        """Convert a Hit into a response item. A single shape for every direction."""
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
        """Search by text. Returns top-k images, or top-k captions if target='caption'."""
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
        """Search by image: an uploaded image, or an image already in the corpus.

        When searching by ``image_id``, the query image is excluded from the
        results — it would always match itself at rank 1, which both wastes a
        meaningless slot in the UI and inflates P@k when measuring.

        :raises BadRequestError: if there isn't exactly one of ``image``/``image_id``,
            or ``image_id`` isn't in the corpus.
        """
        started = time.perf_counter()
        if (image is None) == (image_id is None):
            raise BadRequestError("Need exactly one of: an image file or an image_id")

        spec = get_space(space)
        self._require_mode(spec, MODE_IMAGE2IMAGE)
        k = self._resolve_k(k)

        if image_id is not None:
            record = self.corpus.by_image_id.get(image_id)
            if record is None:
                raise BadRequestError(f"image_id {image_id} is not in the corpus")
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
        """List every space along with its index status, for the frontend to build a dropdown."""
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
