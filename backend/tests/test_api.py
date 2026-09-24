import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.errors import VectorStoreDownError
from app.main import create_app
from app.search import SearchService


@pytest.fixture
def client(settings, service):
    return TestClient(create_app(settings=settings, service=service))


@pytest.fixture
def down_client(settings, service):
    class DownStore:
        mode = settings.qdrant_mode

        def health(self):
            return "down"

        def count(self, name):
            raise VectorStoreDownError("Qdrant không phản hồi")

        def search(self, *args, **kwargs):
            raise VectorStoreDownError("Qdrant không phản hồi")

    broken = SearchService(settings, DownStore(), service.corpus,
                           encoder_factory=service.encoder_factory)
    return TestClient(create_app(settings=settings, service=broken))


def png_bytes(size=(20, 20), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    return buffer.getvalue()


def post_text(client, **kwargs):
    body = {"query": "a brown dog", "space": "clip-b32"}
    body.update(kwargs)
    return client.post("/search/text", json=body)


def test_health_reports_ready_spaces(client):
    body = client.get("/health").json()
    assert body["qdrant"] == "ok"
    assert "clip-b32" in body["spaces_ready"]
    assert "siglip-b16" in body["spaces_missing"]


def test_spaces_lists_every_registered_space(client):
    body = client.get("/spaces").json()
    assert len(body) == 8
    by_name = {s["name"]: s for s in body}
    assert by_name["clip-b32"]["ready"] is True
    assert by_name["mclip-b32"]["languages"] == ["vi", "en", "multi"]


def test_examples_include_vietnamese_queries(client):
    body = client.get("/examples").json()
    assert body
    assert any(item["language"] == "vi" for item in body)


def test_text_search_returns_the_documented_shape(client):
    body = post_text(client).json()
    assert body["space"] == "clip-b32"
    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["thumb_url"].startswith("/thumbs/")
    assert set(body) >= {"results", "latency_ms", "encode_ms", "search_ms",
                         "space", "exact", "total_searched", "query_echo"}


def test_k_larger_than_the_corpus_returns_fewer_results(client):
    response = post_text(client, k=5)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3


def test_filter_matching_nothing_returns_an_empty_list(client):
    response = post_text(client, k=3, filters={"categories": ["unicorn"]})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_blank_query_is_422(client):
    assert post_text(client, query="   ").status_code == 422


def test_k_zero_is_422(client):
    assert post_text(client, k=0).status_code == 422


def test_k_above_the_ceiling_is_422(client):
    assert post_text(client, k=99).status_code == 422


def test_unknown_space_is_404(client):
    assert post_text(client, space="clip-b99").status_code == 404


def test_text_query_on_an_image_only_space_is_400(client):
    assert post_text(client, space="resnet50").status_code == 400


def test_space_without_index_is_409_and_names_the_command(client):
    response = post_text(client, space="siglip-b16")
    assert response.status_code == 409
    assert "tasks.py build" in response.json()["detail"]


def test_qdrant_down_is_503_with_the_fix_in_the_message(down_client):
    response = post_text(down_client)
    assert response.status_code == 503


def test_health_still_answers_when_qdrant_is_down(down_client):
    body = down_client.get("/health").json()
    assert body["qdrant"] == "down"
    assert body["status"] == "degraded"


def test_image_upload_search_works(client):
    response = client.post(
        "/search/image",
        data={"space": "clip-b32", "k": "2"},
        files={"file": ("q.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_image_search_by_corpus_id_works(client):
    response = client.post("/search/image", data={"space": "clip-b32", "image_id": "1"})
    assert response.status_code == 200
    assert 1 not in [r["image_id"] for r in response.json()["results"]]


def test_image_search_without_file_or_id_is_400(client):
    assert client.post("/search/image", data={"space": "clip-b32"}).status_code == 400


def test_image_search_with_both_file_and_id_is_400(client):
    response = client.post(
        "/search/image",
        data={"space": "clip-b32", "image_id": "1"},
        files={"file": ("q.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 400


def test_disallowed_image_type_is_400(client):
    response = client.post(
        "/search/image",
        data={"space": "clip-b32"},
        files={"file": ("q.gif", png_bytes(), "image/gif")},
    )
    assert response.status_code == 400


def test_upload_above_the_size_limit_is_413(client, settings):
    payload = b"x" * (settings.max_upload_bytes + 1)
    response = client.post(
        "/search/image",
        data={"space": "clip-b32"},
        files={"file": ("big.png", payload, "image/png")},
    )
    assert response.status_code == 413


def test_malformed_filters_json_is_422(client):
    response = client.post(
        "/search/image",
        data={"space": "clip-b32", "image_id": "1", "filters_json": "{not json"},
    )
    assert response.status_code == 422


def test_thumbnail_route_serves_a_real_file(client, service):
    name = service.corpus.records[0].file_name
    response = client.get(f"/thumbs/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_missing_thumbnail_is_404(client):
    assert client.get("/thumbs/000000999999.jpg").status_code == 404


def test_path_traversal_is_refused(client):
    assert client.get("/images/..%2F..%2Fcorpus.jsonl").status_code == 404


def test_nested_path_traversal_is_refused(client):
    assert client.get("/images/subdir/../../corpus.jsonl").status_code == 404
