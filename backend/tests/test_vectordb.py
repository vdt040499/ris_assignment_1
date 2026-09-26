import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import Settings
from app.errors import VectorStoreDownError
from app.vectordb import Hit, VectorStore, build_filter

COLLECTION = "test_images"
DIM = 4


@pytest.fixture
def store():
    settings = Settings(_env_file=None, qdrant_mode="embedded")
    return VectorStore(settings, client=QdrantClient(location=":memory:"))


@pytest.fixture
def populated(store):
    store.ensure_collection(COLLECTION, dim=DIM, distance="Cosine")
    vectors = np.eye(3, DIM, dtype=np.float32)
    payloads = [
        {"image_id": 1, "categories": ["dog"], "supercategories": ["animal"]},
        {"image_id": 2, "categories": ["zebra"], "supercategories": ["animal"]},
        {"image_id": 3, "categories": [], "supercategories": []},
    ]
    store.upsert(COLLECTION, ids=[1, 2, 3], vectors=vectors, payloads=payloads,
                 batch_size=2)
    store.create_payload_indexes(COLLECTION, ["categories", "supercategories"])
    return store


def test_no_filters_means_no_filter_object():
    assert build_filter(None, None) is None
    assert build_filter([], []) is None


def test_category_filter_matches_any_of_the_given_values():
    flt = build_filter(["dog", "cat"], None)
    assert flt is not None
    assert len(flt.must) == 1
    assert flt.must[0].key == "categories"


def test_both_filters_produce_two_conditions():
    flt = build_filter(["dog"], ["animal"])
    assert {c.key for c in flt.must} == {"categories", "supercategories"}


def test_ensure_collection_is_idempotent(store):
    assert store.ensure_collection(COLLECTION, dim=DIM, distance="Cosine") is True
    assert store.ensure_collection(COLLECTION, dim=DIM, distance="Cosine") is False
    assert store.collection_exists(COLLECTION) is True


def test_unknown_distance_is_rejected(store):
    with pytest.raises(ValueError):
        store.ensure_collection("bad", dim=DIM, distance="Manhattan")


def test_upsert_counts_every_point_across_batches(populated):
    assert populated.count(COLLECTION) == 3


def test_search_returns_hits_ordered_by_score(populated):
    hits = populated.search(COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=3)
    assert [h.id for h in hits][0] == 1
    assert all(isinstance(h, Hit) for h in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_search_carries_payload_back(populated):
    hits = populated.search(COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=1)
    assert hits[0].payload["image_id"] == 1


def test_search_honours_category_filter(populated):
    hits = populated.search(
        COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=3,
        query_filter=build_filter(["zebra"], None),
    )
    assert [h.id for h in hits] == [2]


def test_search_with_filter_matching_nothing_returns_empty(populated):
    hits = populated.search(
        COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=3,
        query_filter=build_filter(["unicorn"], None),
    )
    assert hits == []


def test_approximate_search_is_accepted(populated):
    hits = populated.search(COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=2,
                            exact=False, hnsw_ef=64)
    assert len(hits) <= 2


def test_count_on_missing_collection_is_zero(store):
    assert store.count("never_created") == 0


def test_health_reports_ok_for_a_live_client(store):
    assert store.health() == "ok"


def test_mode_reflects_settings(store):
    assert store.mode == "embedded"


class _RaisingClient:
    """Bare-bones fake client: every method called raises the given ``exc``.

    Used to verify finding I5 without needing a real Qdrant to actually
    return 4xx/5xx errors: ``UnexpectedResponse`` carries a directly
    simulated ``status_code``.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def get_collections(self):
        raise self._exc

    def collection_exists(self, name):
        raise self._exc

    def get_collection(self, name):
        raise self._exc

    def query_points(self, **kwargs):
        raise self._exc


def _unexpected(status_code):
    return UnexpectedResponse(
        status_code=status_code, reason_phrase="", content=b"{}", headers={}
    )


@pytest.mark.parametrize(
    "exc",
    [
        OSError("no route to host"),
        ResponseHandlingException("boom"),
        _unexpected(500),
        _unexpected(503),
        _unexpected(None),
    ],
)
def test_transport_and_5xx_failures_are_reported_as_qdrant_down(exc):
    """Real transport errors and 5xx are both Qdrant being "down" — a 503 at the HTTP layer."""
    settings = Settings(_env_file=None, qdrant_mode="server")
    store = VectorStore(settings, client=_RaisingClient(exc))
    assert store.health() == "down"
    with pytest.raises(VectorStoreDownError):
        store.collection_exists("any")
    with pytest.raises(VectorStoreDownError):
        store.indexing_status("any")
    with pytest.raises(VectorStoreDownError):
        store.search("any", np.zeros(4, dtype=np.float32), k=1)


def test_health_returns_down_string_on_5xx_without_raising():
    settings = Settings(_env_file=None, qdrant_mode="server")
    store = VectorStore(settings, client=_RaisingClient(_unexpected(500)))
    assert store.health() == "down"


@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_4xx_unexpected_response_is_not_mistaken_for_down(status_code):
    """Finding I5: Qdrant is alive and responding (4xx = it rejected the
    request), not downtime — this must not become a ``VectorStoreDownError``
    (a 503 would misleadingly suggest running "docker compose up -d qdrant"
    while Qdrant is actually running fine).
    """
    settings = Settings(_env_file=None, qdrant_mode="server")
    exc = _unexpected(status_code)
    store = VectorStore(settings, client=_RaisingClient(exc))
    with pytest.raises(UnexpectedResponse):
        store.search("any", np.zeros(4, dtype=np.float32), k=1)
    with pytest.raises(UnexpectedResponse):
        store.collection_exists("any")
    with pytest.raises(UnexpectedResponse):
        store.indexing_status("any")
