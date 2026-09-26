"""Wrapper around Qdrant.

This is the only module in the system that imports ``qdrant_client``. Every
other component works with ``Hit`` and ``build_filter``, so swapping the
vector DB later only requires editing this one file.
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
    """``True`` if ``exc`` genuinely means Qdrant is unreachable.

    ``ResponseHandlingException`` and ``OSError`` are always transport errors
    (the server could not be reached at all). ``UnexpectedResponse`` is more
    ambiguous: it wraps **every** non-2xx status code, including 4xx — meaning
    Qdrant *did* respond but rejected that specific request (e.g. ``limit``
    <= 0), which is not downtime. Treating 4xx as "down" is what caused
    finding I5: ``k<=0`` was misreported as "Qdrant is down" (503) even
    though Qdrant was alive and simply doing its job of rejecting a bad
    parameter. Only 5xx (or a status that can't be determined) truly counts
    as "down".
    """
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is None or exc.status_code >= 500
    return True


@dataclass(frozen=True)
class Hit:
    """A single query result: point id, similarity score, and its payload."""

    id: int
    score: float
    payload: dict


def build_filter(
    categories: Sequence[str] | None, supercategories: Sequence[str] | None
) -> models.Filter | None:
    """Build a Qdrant payload filter from the user-selected category lists.

    Multiple values within the same field are an OR relation (``MatchAny``);
    between the two fields it's an AND relation. Returns ``None`` when there
    is no filter at all, so the caller can pass it straight to Qdrant without
    branching.
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
    """Access Qdrant in server mode (Docker) or embedded mode (no Docker).

    Embedded mode always runs brute-force and ignores the HNSW configuration,
    so it's only for demos and tests — not for producing numbers along the
    ANN-vs-exact axis.
    """

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._settings = settings
        if client is not None:
            self._client = client
        elif settings.qdrant_mode == "server":
            # qdrant-client automatically disables HTTP keep-alive when the host is
            # localhost/127.0.0.1 (see QdrantRemote.__init__), which makes every
            # .search() call open a new TCP connection. On Windows, the tens of
            # thousands of sequential queries from the CLI evaluate command can
            # exhaust the ephemeral port range (WinError 10048) because TIME_WAIT
            # piles up faster than the OS reclaims ports. Pass explicit limits to
            # override that default and reuse connections.
            #
            # Measured in practice on the machine running this project: calling via
            # the name "localhost" costs ~100ms/query (even after re-enabling
            # keep-alive above), while the same request sent directly to
            # "127.0.0.1" takes only ~50ms — the gap comes from how Windows
            # resolves "localhost" on every connection, and has nothing to do with
            # Qdrant or the search algorithm (exact and ANN measure the same
            # timing, ruling out compute as the cause). Replace "localhost" with
            # "127.0.0.1" only in the URL actually used to connect —
            # settings.qdrant_url (shown in error messages, etc.) stays unchanged.
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
        """Returns ``"ok"`` if Qdrant can be reached, ``"down"`` otherwise."""
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
        """Number of points in the collection; 0 if the collection doesn't exist yet."""
        if not self.collection_exists(name):
            return 0
        return self._client.count(name, exact=True).count

    def indexing_status(self, name: str) -> tuple[int, int]:
        """Returns ``(points_count, indexed_vectors_count)`` for the collection.

        Used to properly detect the bug in finding C1: if
        ``indexed_vectors_count`` doesn't equal ``points_count``, HNSW hasn't
        finished building (or hasn't built at all), so every "ANN" query is
        actually a full scan.

        :raises VectorStoreDownError: if Qdrant can't be reached.
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
        """Create the collection if it doesn't exist. Returns True if it was actually created this time.

        :raises ValueError: if the distance name isn't in DISTANCE_MAP.
        """
        if distance not in DISTANCE_MAP:
            raise ValueError(
                f"distance '{distance}' is not supported; valid: {sorted(DISTANCE_MAP)}"
            )
        exists = self.collection_exists(name)
        if exists and not recreate:
            return False
        if exists:
            self._client.delete_collection(name)
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=DISTANCE_MAP[distance]),
            # Qdrant only builds HNSW once a segment exceeds `indexing_threshold`
            # (default 20000 KB). 5,000 vectors x 512 float32 (~10MB) split across
            # multiple segments never reaches that threshold, so HNSW is never
            # built and every query — including "ANN" queries — is actually a
            # full scan (exact search masquerading as ANN, discovered in finding
            # C1 of the final review).
            #
            # IMPORTANT NOTE (differs from finding C1's original guidance):
            # `indexing_threshold=0` does NOT mean "always index" — per the exact
            # docstring of `OptimizersConfigDiff.indexing_threshold` in
            # qdrant-client ("To disable vector indexing, set to 0"), the value 0
            # is a sentinel that **fully disables** indexing, the complete
            # opposite of the intent. Confirmed experimentally on a real server:
            # setting it to 0 and restarting the container, waiting over 10
            # minutes, `indexed_vectors_count` stays at exactly 0 and
            # `/telemetry` reports `optimizations.count = 0` for every
            # collection — meaning the optimizer never runs at all, exactly as
            # "disabled". Use a very small positive threshold (1 KB) instead of
            # 0: per this field's actual unit ("1kB = 1 vector of size 256"),
            # 1 KB is already smaller than a single 512-dimensional vector, so
            # every segment with data crosses the threshold immediately —
            # equivalent in effect to "always index" without hitting the
            # disable sentinel.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
        )
        return True

    def create_payload_indexes(self, name: str, fields: Sequence[str]) -> None:
        """Create keyword payload indexes for the fields used to filter."""
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
        """Upsert in batches. Returns the number of points sent.

        :raises ValueError: if the number of ids, vectors, and payloads don't match.
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"count mismatch: {len(ids)} ids, {len(vectors)} vectors, "
                f"{len(payloads)} payloads"
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
        """Query the top-k results.

        :param exact: ``True`` skips HNSW and scans everything — this is the
            gold standard for the numbers in the report. ``False`` uses HNSW
            with ``hnsw_ef``.
        :raises VectorStoreDownError: if Qdrant can't be reached.
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
                # 4xx: Qdrant is alive and responded, it just rejected this
                # specific request (e.g. k<=0 slipping past the FastAPI layer,
                # see finding I5) — not downtime, so it must not be reported
                # as "docker compose up -d qdrant".
                raise
            raise VectorStoreDownError(
                f"Could not query Qdrant ({self._settings.qdrant_url}). "
                "Run: docker compose up -d qdrant"
            ) from exc
        return [
            Hit(id=int(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]
