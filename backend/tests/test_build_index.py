import json

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.vectordb import VectorStore
from cli.build_index import build_caption_collection, build_space, image_vectors
from app.registry import get_space


@pytest.fixture
def store(settings):
    return VectorStore(settings, client=QdrantClient(location=":memory:"))


@pytest.fixture
def factory(encoder):
    return lambda name, _settings=None: encoder


@pytest.fixture(autouse=True)
def _real_image_files(mini_images_dir):
    """`build_index._load_image` đọc file thật từ `settings.images_dir`.

    `corpus` (từ conftest.py) chỉ dựng corpus.jsonl từ annotation, không tự
    sinh file ảnh. `mini_images_dir` (cũng từ conftest.py) ghi 3 file JPEG
    thật vào cùng `tmp_path/val2017` mà `settings.images_dir` trỏ tới — autouse
    fixture này chỉ đảm bảo nó luôn được gọi, không cần mọi test khai báo lại.
    """
    return mini_images_dir


def test_build_space_upserts_every_image(settings, store, corpus, factory):
    meta = build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    assert meta["n_points"] == 3
    assert store.count("coco_clip_b32") == 3


def test_index_meta_records_what_the_report_needs(settings, store, corpus, factory):
    meta = build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    assert set(meta) >= {"space", "hf_id", "dim", "distance", "normalized",
                         "n_points", "encode_seconds", "batch_size", "built_at",
                         "git_commit"}
    saved = json.loads((settings.index_meta_dir / "clip-b32.json").read_text("utf-8"))
    assert saved["space"] == "clip-b32"


def test_second_build_is_skipped(settings, store, corpus, factory):
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    assert build_space("clip-b32", settings, store, corpus, encoder_factory=factory) is None


def test_force_rebuilds(settings, store, corpus, factory):
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    meta = build_space("clip-b32", settings, store, corpus, force=True,
                       encoder_factory=factory)
    assert meta["n_points"] == 3


def test_space_that_reuses_another_collection_is_skipped(settings, store, corpus, factory):
    assert build_space("mclip-b32", settings, store, corpus, encoder_factory=factory) is None
    assert not (settings.index_meta_dir / "mclip-b32.json").exists()


def test_bm25_space_writes_its_pickle(settings, store, corpus, factory):
    meta = build_space("bm25-cap", settings, store, corpus, encoder_factory=factory)
    assert settings.bm25_path.exists()
    assert meta["n_points"] == 3


def test_caption_collection_holds_one_point_per_caption(settings, store, corpus, factory):
    assert build_caption_collection(settings, store, corpus, factory) == 4
    assert store.count("coco_cap_clip_b32") == 4


def test_image_vectors_are_cached_on_disk(settings, corpus, factory):
    first = image_vectors(get_space("clip-b32"), settings, corpus, factory)
    assert (settings.cache_dir / "imgvec_clip-b32.npy").exists()
    np.testing.assert_allclose(
        first, image_vectors(get_space("clip-b32"), settings, corpus, factory)
    )


def test_normalized_space_derives_from_the_raw_cache(settings, corpus, factory):
    raw = image_vectors(get_space("clip-b32-raw"), settings, corpus, factory)
    normalized = image_vectors(get_space("clip-b32"), settings, corpus, factory)
    assert raw.shape == normalized.shape
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1),
                               np.ones(len(normalized)), atol=1e-5)


def test_filterable_payload_survives_the_upsert(settings, store, corpus, factory):
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    from app.vectordb import build_filter

    hits = store.search("coco_clip_b32", np.eye(1, 4, dtype=np.float32)[0], k=3,
                        query_filter=build_filter(["zebra"], None))
    assert [h.id for h in hits] == [2]
