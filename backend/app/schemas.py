"""Hợp đồng request/response của API.

Response của cả chiều text→ảnh và ảnh→ảnh dùng chung một hình dạng, nên
frontend chỉ cần một component grid duy nhất.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Filters(BaseModel):
    """Filter metadata. Nhiều giá trị trong một field là HOẶC, giữa hai field là VÀ."""

    categories: list[str] = []
    supercategories: list[str] = []


class TextSearchRequest(BaseModel):
    """Request body for semantic search: text or image query into embedding space."""

    query: str
    space: str
    k: int | None = Field(default=None, ge=1)
    exact: bool = True
    hnsw_ef: int | None = Field(default=None, ge=1)
    target: Literal["image", "caption"] = "image"
    filters: Filters = Filters()
    prompt_template: str | None = None

    @field_validator("query")
    @classmethod
    def _require_text(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("query không được rỗng")
        return trimmed

    @field_validator("prompt_template")
    @classmethod
    def _require_placeholder(cls, value: str | None) -> str | None:
        if value is not None and "{}" not in value:
            raise ValueError("prompt_template phải chứa '{}' để chèn query vào")
        return value


class SearchResultItem(BaseModel):
    """One search result, used identically for both text→image and image→image.

    For text→image search (target="image"), matched_caption and caption_index are
    None — the match is at image level. For image→caption search (target="caption"),
    these fields identify which specific caption matched.
    """

    image_id: int
    file_name: str
    thumb_url: str
    score: float
    captions: list[str]
    categories: list[str]
    rank: int
    matched_caption: str | None = None
    caption_index: int | None = None


class SearchResponse(BaseModel):
    """Response body for search queries, with unified shape for all directions."""

    results: list[SearchResultItem]
    latency_ms: float
    encode_ms: float
    search_ms: float
    space: str
    exact: bool
    total_searched: int
    query_echo: str


class SpaceInfo(BaseModel):
    """Metadata about an embedding space."""

    name: str
    hf_id: str | None
    dim: int | None
    backend: str
    modes: list[str]
    languages: list[str]
    ready: bool
    n_points: int
    note: str


class HealthResponse(BaseModel):
    """Server health status and space readiness."""

    status: str
    qdrant: str
    qdrant_mode: str
    spaces_ready: list[str]
    spaces_missing: list[str]


class ExampleQuery(BaseModel):
    """Example query for API documentation and frontend UI."""

    label: str
    query: str
    space: str
    language: str
