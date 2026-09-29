"""A single test that runs the whole chain on a 3-image corpus: ingest → build → search → eval."""

import pytest
from qdrant_client import QdrantClient

from app.corpus import load_corpus
from app.schemas import TextSearchRequest
from app.search import SearchService
from app.vectordb import VectorStore
from cli.build_index import build_caption_collection, build_space
from cli.evaluate import EvalContext, eval_text2image
from cli.ingest import build_corpus, make_thumbnails


def test_whole_pipeline_from_annotations_to_metrics(
    settings, mini_annotations_dir, mini_images_dir, encoder
):
    factory = lambda name, _settings=None: encoder  # noqa: E731

    stats = build_corpus(mini_annotations_dir, settings.corpus_path)
    assert stats.n_images == 3
    corpus = load_corpus(settings.corpus_path)
    assert make_thumbnails(corpus, mini_images_dir, settings.thumbs_dir,
                           settings.thumb_size) == 3

    store = VectorStore(settings, client=QdrantClient(location=":memory:"))
    assert build_space("clip-b32", settings, store, corpus,
                       encoder_factory=factory)["n_points"] == 3
    assert build_caption_collection(settings, store, corpus, factory) == 4

    service = SearchService(settings, store, corpus,
                            encoder_factory=lambda name: encoder)
    response = service.search_text(
        TextSearchRequest(query="a brown dog", space="clip-b32", k=1)
    )
    assert response.results[0].image_id == 1

    queries = [(text, image_id) for image_id, _, text in corpus.caption_pairs()]
    row = eval_text2image(
        EvalContext(settings, store, corpus, encoder_factory=factory),
        "clip-b32", queries, ks=(1,),
    )
    assert row["R@1"] == pytest.approx(1.0)
