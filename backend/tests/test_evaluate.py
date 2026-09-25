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
    """`build_index._load_image` đọc file thật từ `settings.images_dir`.

    Cùng lý do với fixture cùng tên ở `test_build_index.py`: `ctx` gọi
    `build_space`, vốn cần ảnh thật trên đĩa để encode, nhưng `corpus` (từ
    conftest.py) chỉ dựng corpus.jsonl chứ không tự sinh file ảnh. Autouse ở
    đây để mọi test dùng `ctx` không phải tự khai báo `mini_images_dir`.
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
    """self_hits đếm trên ranking THÔ (trước lọc self), nên phụ thuộc dữ liệu
    thật chứ không phải một hằng số theo cấu trúc code.

    FakeEncoder mã hoá ảnh 1 (80x60) thành vector one-hot chỉ số 0, khác hẳn
    ảnh 2 và 3 — nên khi lấy chính vector ảnh 1 làm query, ảnh 1 luôn là láng
    giềng gần nhất tuyệt đối (cosine=1.0) của chính nó trong ranking thô, và
    self_hits phải bằng đúng 1 (không phải 0 mặc định).
    """
    row = eval_image2image(ctx, "clip-b32", image_ids=[1], k=2)
    assert row["self_hits"] == 1


def test_image2image_excludes_the_query_image_from_scoring(ctx):
    """Kiểm exclusion bằng hiệu ứng thật của nó lên P@k, không phải bằng
    self_hits (self_hits đếm trên ranking thô — xem test phía trên — nên
    không thể chứng minh bước lọc ``others`` có chạy hay không: một self_hits
    đếm trên chính ``others`` đã lọc luôn ra 0 bất kể code loại trừ có đúng
    hay không).

    Corpus mini gán ảnh 1 categories ``{"couch", "dog"}``, không trùng ảnh 2
    (``{"zebra"}``) hay ảnh 3 (rỗng). Nếu bước loại trừ ảnh query khỏi
    ``others`` bị gỡ bỏ (hoặc hỏng), ảnh 1 sẽ tự khớp category với chính nó
    (trùng 100%) và lọt vào ``others`` — kéo P@2 lên trên 0. P@2 == 0.0 do đó
    chỉ đúng khi việc loại trừ thật sự hoạt động.
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
    write_table(rows, "axis1_model_t2i", settings, title="Trục 1")
    csv_text = (settings.results_dir / "axis1_model_t2i.csv").read_text("utf-8")
    md_text = (settings.results_dir / "axis1_model_t2i.md").read_text("utf-8")
    assert csv_text.splitlines()[0] == "space,R@1"
    assert "| space | R@1 |" in md_text
    assert "Trục 1" in md_text


def test_write_table_with_no_rows_is_not_an_error(settings):
    write_table([], "empty_axis", settings, title="Rỗng")
    assert (settings.results_dir / "empty_axis.md").exists()
