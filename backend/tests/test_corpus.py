import pytest

from app.corpus import Corpus, CorpusRecord, load_corpus, record_payload, write_corpus
from app.errors import CorpusError


def make_record(image_id=1, captions=("a dog",)):
    return CorpusRecord(
        image_id=image_id,
        file_name=f"{image_id:012d}.jpg",
        width=80,
        height=60,
        captions=list(captions),
        categories=["dog"],
        supercategories=["animal"],
    )


def test_record_rejects_empty_caption_list():
    with pytest.raises(ValueError):
        CorpusRecord(image_id=1, file_name="a.jpg", width=1, height=1, captions=[])


def test_record_strips_blank_captions():
    rec = CorpusRecord(
        image_id=1, file_name="a.jpg", width=1, height=1,
        captions=["  a dog  ", "   ", ""],
    )
    assert rec.captions == ["a dog"]


def test_corpus_indexes_by_image_id():
    corpus = Corpus(records=[make_record(1), make_record(2)])
    assert len(corpus) == 2
    assert corpus.by_image_id[2].image_id == 2
    assert corpus.image_ids() == [1, 2]


def test_caption_pairs_flattens_with_indices():
    corpus = Corpus(records=[make_record(1, ("a", "b")), make_record(2, ("c",))])
    assert corpus.caption_pairs() == [(1, 0, "a"), (1, 1, "b"), (2, 0, "c")]


def test_record_payload_carries_metadata_for_filtering():
    payload = record_payload(make_record(7))
    assert payload["image_id"] == 7
    assert payload["categories"] == ["dog"]
    assert payload["supercategories"] == ["animal"]
    assert payload["file_name"] == "000000000007.jpg"
    assert payload["captions"] == ["a dog"]


def test_write_then_load_roundtrips(tmp_path):
    path = tmp_path / "corpus.jsonl"
    written = write_corpus([make_record(1), make_record(2)], path)
    assert written == 2
    corpus = load_corpus(path)
    assert [r.image_id for r in corpus.records] == [1, 2]


def test_load_missing_file_raises_corpus_error(tmp_path):
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "nope.jsonl")


def test_load_empty_file_raises_corpus_error(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(CorpusError):
        load_corpus(path)


def test_load_bad_line_names_the_line_number(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"image_id": 1}\n', encoding="utf-8")
    with pytest.raises(CorpusError) as exc:
        load_corpus(path)
    assert "line 1" in str(exc.value)
