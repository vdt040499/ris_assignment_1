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
    """`build_index._load_image` reads real files from `settings.images_dir`.

    `corpus` (from conftest.py) only builds corpus.jsonl from the
    annotations and doesn't generate image files itself. `mini_images_dir`
    (also from conftest.py) writes 3 real JPEG files into the same
    `tmp_path/val2017` that `settings.images_dir` points to — this autouse
    fixture just ensures it always gets called, so tests don't have to
    redeclare it.
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


def test_image_vectors_are_cached_on_disk(settings, corpus, factory, encoder):
    first = image_vectors(get_space("clip-b32"), settings, corpus, factory)
    assert (settings.cache_dir / "imgvec_clip-b32.npy").exists()
    np.testing.assert_allclose(
        first, image_vectors(get_space("clip-b32"), settings, corpus, factory)
    )
    # Prove the second call is a cache hit rather than another encode call
    # that coincidentally produces the same result (FakeEncoder is
    # deterministic, so the two are indistinguishable by comparing vector
    # values alone).
    assert encoder.image_encode_calls == 1


def test_normalized_space_derives_from_the_raw_cache(settings, corpus, factory, encoder):
    raw = image_vectors(get_space("clip-b32-raw"), settings, corpus, factory)
    normalized = image_vectors(get_space("clip-b32"), settings, corpus, factory)
    assert raw.shape == normalized.shape
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1),
                               np.ones(len(normalized)), atol=1e-5)
    # This is the real proof of the DERIVED_FROM mechanism: if the derive
    # branch were removed, reversed, or BUILD_ORDER got sorted in the wrong
    # order, clip-b32 would fall through to a real second encode instead of
    # reloading clip-b32-raw's cache. Since FakeEncoder is deterministic, two
    # encode calls would still produce the same vector and the two
    # assertions above would still pass — only counting the number of
    # encode_images calls can catch this regression.
    assert encoder.image_encode_calls == 1


def test_filterable_payload_survives_the_upsert(settings, store, corpus, factory):
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    from app.vectordb import build_filter

    hits = store.search("coco_clip_b32", np.eye(1, 4, dtype=np.float32)[0], k=3,
                        query_filter=build_filter(["zebra"], None))
    assert [h.id for h in hits] == [2]
