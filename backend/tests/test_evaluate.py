import json

import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.vectordb import VectorStore
from cli.build_index import build_caption_collection, build_space
from cli.evaluate import (
    EvalContext,
    eval_ann_sweep,
    eval_image2image,
    eval_language,
    eval_short_queries,
    eval_text2image,
    write_table,
)


@pytest.fixture
def factory(encoder):
    return lambda name, _settings=None: encoder


@pytest.fixture(autouse=True)
def _real_image_files(mini_images_dir):
    """`build_index._load_image` reads real files from `settings.images_dir`.

    Same reason as the fixture of the same name in `test_build_index.py`:
    `ctx` calls `build_space`, which needs real images on disk to encode, but
    `corpus` (from conftest.py) only builds corpus.jsonl and does not
    generate image files itself. Autouse here so every test using `ctx`
    doesn't have to declare `mini_images_dir` itself.
    """
    return mini_images_dir


@pytest.fixture
def ctx(settings, corpus, factory):
    store = VectorStore(settings, client=QdrantClient(location=":memory:"))
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    build_space("clip-b32-raw", settings, store, corpus, encoder_factory=factory)
    build_space("bm25-cap", settings, store, corpus, encoder_factory=factory)
    build_caption_collection(settings, store, corpus, factory)
    return EvalContext(settings, store, corpus, encoder_factory=factory)


@pytest.fixture
def caption_queries(corpus):
    return [(text, image_id) for image_id, _, text in corpus.caption_pairs()]


def test_text2image_finds_the_source_image_of_every_caption(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1, 3))
    assert row["R@1"] == pytest.approx(1.0)
    assert row["n_queries"] == 4
    assert row["space"] == "clip-b32"


def test_text2image_reports_mrr_and_latency(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1,))
    assert 0.0 <= row["MRR@10"] <= 1.0
    assert row["search_ms_p50"] >= 0
    assert row["search_ms_p95"] >= row["search_ms_p50"]


def test_text2image_is_deterministic(ctx, caption_queries):
    first = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1, 3))
    second = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1, 3))
    assert first["R@1"] == second["R@1"]
    assert first["MRR@10"] == second["MRR@10"]


def test_bm25_baseline_is_evaluated_through_the_same_function(ctx, caption_queries):
    row = eval_text2image(ctx, "bm25-cap", caption_queries, ks=(1,))
    assert row["space"] == "bm25-cap"
    assert 0.0 <= row["R@1"] <= 1.0


def test_unnormalized_space_is_evaluated_on_its_own_collection(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32-raw", caption_queries, ks=(1,))
    assert row["space"] == "clip-b32-raw"


def test_prompt_template_is_recorded_in_the_row(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1,),
                          prompt_template="a photo of {}", cache_key="tmpl")
    assert row["prompt_template"] == "a photo of {}"


def test_image2image_uses_the_category_proxy(ctx):
    row = eval_image2image(ctx, "clip-b32", image_ids=[1, 2, 3], k=2)
    assert row["n_queries"] == 3
    assert 0.0 <= row["P@2"] <= 1.0
    assert 0.0 <= row["mAP@2"] <= 1.0


def test_image2image_reports_self_as_its_own_nearest_neighbor(ctx):
    """self_hits is counted on the RAW ranking (before self-filtering), so it
    depends on the actual data rather than being a constant determined by
    the code's structure.

    FakeEncoder encodes image 1 (80x60) as a one-hot vector at index 0,
    clearly distinct from images 2 and 3 — so when image 1's own vector is
    used as the query, image 1 is always its own absolute nearest neighbor
    (cosine=1.0) in the raw ranking, and self_hits must equal exactly 1 (not
    the default 0).
    """
    row = eval_image2image(ctx, "clip-b32", image_ids=[1], k=2)
    assert row["self_hits"] == 1


def test_image2image_excludes_the_query_image_from_scoring(ctx):
    """Verify exclusion through its actual effect on P@k, not through
    self_hits (self_hits is counted on the raw ranking — see the test above
    — so it can't prove whether the ``others`` filtering step actually runs:
    a self_hits counted on the already-filtered ``others`` would always come
    out 0 regardless of whether the exclusion code is correct).

    The mini corpus assigns image 1 the categories ``{"couch", "dog"}``,
    which don't overlap with image 2 (``{"zebra"}``) or image 3 (empty). If
    the step that excludes the query image from ``others`` were removed (or
    broken), image 1 would match its own category with itself (100% match)
    and end up in ``others`` — pushing P@2 above 0. P@2 == 0.0 is therefore
    only correct when the exclusion is actually working.
    """
    row = eval_image2image(ctx, "clip-b32", image_ids=[1], k=2)
    assert row["P@2"] == pytest.approx(0.0)


def test_short_queries_are_scored_by_category_hit(ctx, settings):
    (settings.data_dir / "queryset_short.json").write_text(
        json.dumps([{"query": "a dog", "gold_category": "dog", "n_gold": 1}]),
        encoding="utf-8",
    )
    row = eval_short_queries(ctx, "clip-b32", k=1)
    assert row["n_queries"] == 1
    assert 0.0 <= row["P@1"] <= 1.0


def test_ann_sweep_refuses_to_run_in_embedded_mode(ctx, caption_queries):
    with pytest.raises(RuntimeError):
        eval_ann_sweep(ctx, "clip-b32", caption_queries, efs=(16,), k=2)


def test_ann_sweep_reports_overlap_against_exact(settings, corpus, factory,
                                                 caption_queries, monkeypatch):
    server_like = settings.model_copy(update={"qdrant_mode": "server"})
    store = VectorStore(server_like, client=QdrantClient(location=":memory:"))
    build_space("clip-b32", server_like, store, corpus, encoder_factory=factory)
    # QdrantClient(location=":memory:") is Qdrant's *local* mode: it never
    # builds a real HNSW graph and always reports indexed_vectors_count=0
    # (see qdrant_client/local/local_collection.py), regardless of
    # `qdrant_mode="server"` above. That's fine for exercising the overlap
    # computation itself, but it would trip the new C1 guard below, which is
    # specifically there to catch this exact "collection exists but HNSW
    # never got built" condition on a *real* server. Fake a fully-indexed
    # collection so this test still isolates what it means to test.
    monkeypatch.setattr(store, "indexing_status", lambda name: (4, 4))
    context = EvalContext(server_like, store, corpus, encoder_factory=factory)
    rows = eval_ann_sweep(context, "clip-b32", caption_queries, efs=(16, 64), k=2)
    assert [r["hnsw_ef"] for r in rows] == [16, 64]
    assert all(0.0 <= r["overlap@2"] <= 1.0 for r in rows)


def test_ann_sweep_refuses_when_hnsw_never_got_built(settings, corpus, factory,
                                                      caption_queries, monkeypatch):
    """Guard for C1: a collection can be in "server" mode and still have
    never built HNSW, if it never crossed Qdrant's `indexing_threshold`
    (the exact bug the final review found live). `indexed_vectors_count !=
    points_count` must refuse loudly instead of silently measuring exact vs
    exact and calling it ANN vs exact.
    """
    server_like = settings.model_copy(update={"qdrant_mode": "server"})
    store = VectorStore(server_like, client=QdrantClient(location=":memory:"))
    build_space("clip-b32", server_like, store, corpus, encoder_factory=factory)
    monkeypatch.setattr(store, "indexing_status", lambda name: (4, 0))
    context = EvalContext(server_like, store, corpus, encoder_factory=factory)
    with pytest.raises(RuntimeError, match="HNSW"):
        eval_ann_sweep(context, "clip-b32", caption_queries, efs=(16,), k=2)


def test_language_axis_scores_both_spaces_on_both_languages(ctx, settings):
    (settings.data_dir / "queryset_vi.json").write_text(
        json.dumps([
            {"image_id": 1, "caption_index": 0, "en": "a brown dog on a red sofa",
             "vi": "một con chó nâu trên ghế sofa đỏ"},
            {"image_id": 3, "caption_index": 0, "en": "a slice of pepperoni pizza",
             "vi": "một miếng pizza pepperoni"},
        ]),
        encoding="utf-8",
    )
    rows = eval_language(ctx, ["clip-b32"], k=(1,))
    assert {r["language"] for r in rows} == {"en", "vi"}
    assert all(r["n_queries"] == 2 for r in rows)


def test_write_table_emits_both_csv_and_markdown(settings):
    rows = [{"space": "clip-b32", "R@1": 0.3}, {"space": "laion-b32", "R@1": 0.35}]
    write_table(rows, "axis1_model_t2i", settings, title="Axis 1")
    csv_text = (settings.results_dir / "axis1_model_t2i.csv").read_text("utf-8")
    md_text = (settings.results_dir / "axis1_model_t2i.md").read_text("utf-8")
    assert csv_text.splitlines()[0] == "space,R@1"
    assert "| space | R@1 |" in md_text
    assert "Axis 1" in md_text


def test_write_table_with_no_rows_is_not_an_error(settings):
    write_table([], "empty_axis", settings, title="Empty")
    assert (settings.results_dir / "empty_axis.md").exists()
