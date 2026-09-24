import numpy as np
import pytest
from PIL import Image
from qdrant_client import QdrantClient

from app.config import Settings
from app.corpus import caption_payload, caption_point_id, load_corpus, record_payload
from app.encoders.bm25 import Bm25Retriever
from app.errors import (
    BadRequestError,
    IndexNotBuiltError,
    ModeNotSupportedError,
    UnknownSpaceError,
    ValidationRangeError,
)
from app.schemas import Filters, TextSearchRequest
from app.search import SearchService
from app.vectordb import VectorStore
from cli.ingest import build_corpus

DIM = 4
TEXT_KEYWORDS = {"dog": 0, "puppy": 0, "couch": 0, "sofa": 0, "zebra": 1, "pizza": 2}
SIZE_TO_INDEX = {(80, 60): 0, (60, 90): 1, (40, 40): 2}


class FakeEncoder:
    """Encoder tất định 4 chiều: đủ để kiểm tra điều phối mà không tải model."""

    name = "fake"
    dim = DIM

    def __init__(self):
        self.seen_texts: list[str] = []

    def encode_texts(self, texts):
        self.seen_texts.extend(texts)
        basis = np.eye(DIM, dtype=np.float32)
        return np.stack([
            basis[next((i for kw, i in TEXT_KEYWORDS.items() if kw in t.lower()), 3)]
            for t in texts
        ])

    def encode_images(self, images):
        basis = np.eye(DIM, dtype=np.float32)
        return np.stack([basis[SIZE_TO_INDEX.get(img.size, 3)] for img in images])


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, qdrant_mode="embedded",
                    top_k_default=2, max_top_k=5)


@pytest.fixture
def corpus(mini_annotations_dir, settings):
    build_corpus(mini_annotations_dir, settings.corpus_path)
    return load_corpus(settings.corpus_path)


@pytest.fixture
def service(settings, corpus, encoder, mini_images_dir):
    store = VectorStore(settings, client=QdrantClient(location=":memory:"))
    images = [
        Image.open(mini_images_dir / r.file_name).convert("RGB") for r in corpus.records
    ]
    vectors = encoder.encode_images(images)

    store.ensure_collection("coco_clip_b32", dim=DIM, distance="Cosine")
    store.upsert("coco_clip_b32", ids=corpus.image_ids(), vectors=vectors,
                 payloads=[record_payload(r) for r in corpus.records], batch_size=8)
    store.create_payload_indexes("coco_clip_b32", ["categories", "supercategories"])

    caption_ids, caption_vectors, caption_payloads = [], [], []
    for record in corpus.records:
        for idx in range(len(record.captions)):
            caption_ids.append(caption_point_id(record.image_id, idx))
            caption_vectors.append(encoder.encode_texts([record.captions[idx]])[0])
            caption_payloads.append(caption_payload(record, idx))
    store.ensure_collection("coco_cap_clip_b32", dim=DIM, distance="Cosine")
    store.upsert("coco_cap_clip_b32", ids=caption_ids,
                 vectors=np.stack(caption_vectors), payloads=caption_payloads,
                 batch_size=8)

    Bm25Retriever.build(corpus).save(settings.bm25_path)
    return SearchService(settings, store, corpus, encoder_factory=lambda name: encoder)


def text_request(**kwargs):
    base = {"query": "a brown dog", "space": "clip-b32"}
    base.update(kwargs)
    return TextSearchRequest(**base)


def test_text_search_ranks_the_matching_image_first(service):
    response = service.search_text(text_request())
    assert response.results[0].image_id == 1
    assert response.results[0].rank == 1


def test_ranks_are_consecutive_from_one(service):
    response = service.search_text(text_request(k=3))
    assert [item.rank for item in response.results] == [1, 2, 3]


def test_scores_come_back_in_descending_order(service):
    scores = [i.score for i in service.search_text(text_request(k=3)).results]
    assert scores == sorted(scores, reverse=True)


def test_thumb_url_points_at_the_thumbnail_route(service):
    item = service.search_text(text_request()).results[0]
    assert item.thumb_url == f"/thumbs/{item.file_name}"


def test_k_falls_back_to_the_configured_default(service, settings):
    assert len(service.search_text(text_request()).results) == settings.top_k_default


def test_k_above_the_configured_ceiling_is_refused(service):
    with pytest.raises(ValidationRangeError):
        service.search_text(text_request(k=99))


def test_unknown_space_is_reported_as_such(service):
    with pytest.raises(UnknownSpaceError):
        service.search_text(text_request(space="clip-b99"))


def test_text_query_against_an_image_only_space_is_refused(service):
    with pytest.raises(ModeNotSupportedError):
        service.search_text(text_request(space="resnet50"))


def test_space_without_an_index_says_so(service):
    with pytest.raises(IndexNotBuiltError):
        service.search_text(text_request(space="siglip-b16"))


def test_prompt_template_reaches_the_encoder(service, encoder):
    service.search_text(text_request(prompt_template="a photo of {}"))
    assert encoder.seen_texts[-1] == "a photo of a brown dog"


def test_category_filter_narrows_the_candidates(service):
    response = service.search_text(
        text_request(k=3, filters=Filters(categories=["zebra"]))
    )
    assert [i.image_id for i in response.results] == [2]


def test_caption_target_returns_the_matched_caption(service):
    response = service.search_text(text_request(query="pizza", target="caption"))
    assert response.results[0].matched_caption == "a slice of pepperoni pizza"
    assert response.results[0].caption_index == 0


def test_latency_is_split_into_encode_and_search(service):
    response = service.search_text(text_request())
    assert response.encode_ms >= 0
    assert response.search_ms >= 0
    assert response.latency_ms >= response.search_ms


def test_total_searched_reports_the_collection_size(service):
    assert service.search_text(text_request()).total_searched == 3


def test_bm25_space_answers_text_queries(service):
    response = service.search_text(text_request(query="zebras grazing", space="bm25-cap"))
    assert response.results[0].image_id == 2


def test_bm25_space_honours_category_filter(service):
    response = service.search_text(
        text_request(query="a", space="bm25-cap", k=5,
                     filters=Filters(categories=["zebra"]))
    )
    assert {i.image_id for i in response.results} <= {2}


def test_image_search_by_corpus_id_excludes_the_query_itself(service):
    response = service.search_image(space="clip-b32", k=2, exact=True, hnsw_ef=None,
                                    filters=Filters(), image_id=1)
    assert 1 not in [i.image_id for i in response.results]


def test_image_search_needs_exactly_one_input(service):
    with pytest.raises(BadRequestError):
        service.search_image(space="clip-b32", k=2, exact=True, hnsw_ef=None,
                             filters=Filters())


def test_unknown_image_id_is_a_bad_request(service):
    with pytest.raises(BadRequestError):
        service.search_image(space="clip-b32", k=2, exact=True, hnsw_ef=None,
                             filters=Filters(), image_id=999999)


def test_space_infos_flags_which_indexes_are_ready(service):
    infos = {info.name: info for info in service.space_infos()}
    assert infos["clip-b32"].ready is True
    assert infos["clip-b32"].n_points == 3
    assert infos["siglip-b16"].ready is False
