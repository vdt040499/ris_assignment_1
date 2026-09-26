import pytest
from pydantic import ValidationError

from app.schemas import Filters, SearchResponse, SearchResultItem, TextSearchRequest


def test_minimal_request_fills_sane_defaults():
    req = TextSearchRequest(query="a dog", space="clip-b32")
    assert req.k is None
    assert req.exact is True
    assert req.target == "image"
    assert req.filters == Filters()
    assert req.prompt_template is None


def test_blank_query_is_rejected():
    with pytest.raises(ValidationError):
        TextSearchRequest(query="   ", space="clip-b32")


def test_query_is_trimmed():
    assert TextSearchRequest(query="  a dog  ", space="clip-b32").query == "a dog"


def test_k_below_one_is_rejected():
    with pytest.raises(ValidationError):
        TextSearchRequest(query="a dog", space="clip-b32", k=0)


def test_unknown_target_is_rejected():
    with pytest.raises(ValidationError):
        TextSearchRequest(query="a dog", space="clip-b32", target="video")


def test_prompt_template_must_contain_a_placeholder():
    with pytest.raises(ValidationError):
        TextSearchRequest(query="a dog", space="clip-b32", prompt_template="a photo of")


def test_valid_prompt_template_is_accepted():
    req = TextSearchRequest(query="a dog", space="clip-b32", prompt_template="a photo of {}")
    assert req.prompt_template == "a photo of {}"


def test_response_shape_is_the_same_for_both_directions():
    item = SearchResultItem(
        image_id=1, file_name="a.jpg", thumb_url="/thumbs/a.jpg", score=0.5,
        captions=["a dog"], categories=["dog"], rank=1,
    )
    response = SearchResponse(
        results=[item], latency_ms=1.0, encode_ms=0.4, search_ms=0.6,
        space="clip-b32", exact=True, total_searched=5, query_echo="a dog",
    )
    assert response.results[0].matched_caption is None
    assert response.results[0].caption_index is None
