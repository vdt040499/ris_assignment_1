import json
from pathlib import Path

import pytest

from cli.make_querysets import (
    build_short_queries,
    sample_i2i_queries,
    sample_vi_queries,
)

VI_QUERYSET = Path("data/queryset_vi.json")


def test_vi_sampling_is_deterministic(corpus):
    assert sample_vi_queries(corpus, 2, seed=7) == sample_vi_queries(corpus, 2, seed=7)


def test_vi_sampling_changes_with_the_seed(corpus):
    a = sample_vi_queries(corpus, 3, seed=1)
    b = sample_vi_queries(corpus, 3, seed=2)
    assert a != b or len(corpus) < 3


def test_vi_entries_carry_the_english_source_and_an_empty_slot(corpus):
    entry = sample_vi_queries(corpus, 1, seed=7)[0]
    assert entry["en"]
    assert entry["vi"] == ""
    assert entry["caption_index"] >= 0


def test_vi_sampling_never_repeats_a_caption(corpus):
    entries = sample_vi_queries(corpus, 4, seed=7)
    keys = {(e["image_id"], e["caption_index"]) for e in entries}
    assert len(keys) == len(entries)


def test_vi_sampling_caps_at_the_corpus_size(corpus):
    assert len(sample_vi_queries(corpus, 10_000, seed=7)) <= len(corpus.caption_pairs())


def test_i2i_sampling_is_deterministic_and_unique(corpus):
    ids = sample_i2i_queries(corpus, 3, seed=7)
    assert ids == sample_i2i_queries(corpus, 3, seed=7)
    assert len(set(ids)) == len(ids)


def test_short_queries_cover_every_category_present(corpus):
    queries = build_short_queries(corpus)
    assert {q["gold_category"] for q in queries} == {"dog", "couch", "zebra"}


def test_short_queries_report_how_many_images_are_relevant(corpus):
    by_cat = {q["gold_category"]: q for q in build_short_queries(corpus)}
    assert by_cat["dog"]["n_gold"] == 1
    assert by_cat["dog"]["query"] == "a dog"


def test_short_queries_skip_categories_with_no_images(corpus):
    assert all(q["n_gold"] > 0 for q in build_short_queries(corpus))


@pytest.mark.skipif(not VI_QUERYSET.exists(), reason="bộ tiếng Việt chưa sinh")
def test_vietnamese_queryset_is_fully_translated():
    entries = json.loads(VI_QUERYSET.read_text(encoding="utf-8"))
    assert len(entries) >= 200
    assert all(e["vi"].strip() for e in entries), "còn entry chưa dịch"
    assert all(e["vi"].strip() != e["en"].strip() for e in entries), "có entry chưa dịch thật"
