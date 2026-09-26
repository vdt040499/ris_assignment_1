import pytest

from app.corpus import Corpus, CorpusRecord
from app.encoders.bm25 import Bm25Retriever, tokenize


def record(image_id, captions, categories=()):
    return CorpusRecord(
        image_id=image_id,
        file_name=f"{image_id:012d}.jpg",
        width=10,
        height=10,
        captions=list(captions),
        categories=list(categories),
    )


@pytest.fixture
def corpus():
    return Corpus(
        records=[
            record(1, ["a brown dog on a red sofa", "a sleeping puppy indoors"]),
            record(2, ["two zebras grazing in a field"]),
            record(3, ["a slice of pepperoni pizza"]),
        ]
    )


@pytest.fixture
def retriever(corpus):
    return Bm25Retriever.build(corpus)


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("A Brown Dog, on a SOFA!") == ["a", "brown", "dog", "on", "a", "sofa"]


def test_tokenize_keeps_digits():
    assert tokenize("flight 370") == ["flight", "370"]


def test_keyword_match_ranks_first(retriever):
    hits = retriever.search("zebras field", k=3)
    assert hits[0].id == 2


def test_score_of_an_image_is_the_best_of_its_captions(retriever):
    hits = retriever.search("sleeping puppy", k=3)
    assert hits[0].id == 1


def test_only_images_with_a_match_come_back(retriever):
    hits = retriever.search("pizza", k=3)
    assert [h.id for h in hits] == [3]


def test_query_without_usable_tokens_returns_nothing(retriever):
    assert retriever.search("!!! ???", k=5) == []


def test_k_limits_the_result_count(retriever):
    assert len(retriever.search("a", k=2)) <= 2


def test_hits_carry_the_corpus_payload(retriever):
    hits = retriever.search("pizza", k=1)
    assert hits[0].payload["file_name"] == "000000000003.jpg"
    assert hits[0].payload["captions"] == ["a slice of pepperoni pizza"]


def test_allowed_ids_restricts_the_candidate_set(retriever):
    hits = retriever.search("a", k=5, allowed_ids={3})
    assert {h.id for h in hits} <= {3}


def test_save_and_load_preserve_ranking(retriever, corpus, tmp_path):
    path = tmp_path / "bm25.pkl"
    retriever.save(path)
    reloaded = Bm25Retriever.load(path, corpus)
    assert reloaded.search("zebras", k=1)[0].id == retriever.search("zebras", k=1)[0].id


def test_load_missing_file_raises(tmp_path, corpus):
    with pytest.raises(FileNotFoundError):
        Bm25Retriever.load(tmp_path / "nope.pkl", corpus)
