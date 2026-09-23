import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.config import Settings
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
