from PIL import Image

from app.corpus import load_corpus
from cli.ingest import build_corpus, make_thumbnails


def test_build_corpus_writes_one_record_per_image(mini_annotations_dir, tmp_path):
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(mini_annotations_dir, out)
    assert stats.n_images == 3
    assert stats.n_captions == 4
    corpus = load_corpus(out)
    assert corpus.image_ids() == [1, 2, 3]


def test_records_are_sorted_by_image_id(mini_annotations_dir, tmp_path):
    out = tmp_path / "corpus.jsonl"
    build_corpus(mini_annotations_dir, out)
    ids = load_corpus(out).image_ids()
    assert ids == sorted(ids)


def test_categories_are_deduplicated_and_sorted(mini_annotations_dir, tmp_path):
    out = tmp_path / "corpus.jsonl"
    build_corpus(mini_annotations_dir, out)
    rec = load_corpus(out).by_image_id[1]
    assert rec.categories == ["couch", "dog"]
    assert rec.supercategories == ["animal", "furniture"]


def test_image_without_annotation_gets_empty_lists(mini_annotations_dir, tmp_path):
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(mini_annotations_dir, out)
    rec = load_corpus(out).by_image_id[3]
    assert rec.categories == []
    assert rec.supercategories == []
    assert stats.n_without_category == 1


def test_caption_count_stats_are_measured_not_assumed(mini_annotations_dir, tmp_path):
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(mini_annotations_dir, out)
    assert stats.min_captions == 1
    assert stats.max_captions == 2


def test_thumbnails_cap_the_long_side(mini_annotations_dir, mini_images_dir, tmp_path):
    out = tmp_path / "corpus.jsonl"
    build_corpus(mini_annotations_dir, out)
    corpus = load_corpus(out)
    thumbs = tmp_path / "thumbs"
    made = make_thumbnails(corpus, mini_images_dir, thumbs, size=32)
    assert made == 3
    with Image.open(thumbs / "000000000002.jpg") as img:
        assert max(img.size) == 32


def test_thumbnails_skip_work_already_done(mini_annotations_dir, mini_images_dir, tmp_path):
    out = tmp_path / "corpus.jsonl"
    build_corpus(mini_annotations_dir, out)
    corpus = load_corpus(out)
    thumbs = tmp_path / "thumbs"
    assert make_thumbnails(corpus, mini_images_dir, thumbs, size=32) == 3
    assert make_thumbnails(corpus, mini_images_dir, thumbs, size=32) == 0
