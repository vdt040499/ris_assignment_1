"""System-wide configuration. Every runtime parameter of the system lives here, not scattered in code."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration parameters, loaded in this order: environment variables > .env > defaults."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    qdrant_mode: Literal["server", "embedded"] = "server"
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout_s: float = 10.0
    #: Number of HTTP connections to keep alive to the Qdrant server. qdrant-client
    #: disables keep-alive by default when the host is localhost/127.0.0.1
    #: (treating that as a latency optimization), but that causes every
    #: .search() call to open a new TCP connection and close it right away —
    #: running tens of thousands of sequential queries (like CLI evaluate)
    #: piles up ports in TIME_WAIT and can exhaust the ephemeral port range on
    #: Windows (WinError 10048). VectorStore overrides this with httpx.Limits
    #: using this value to keep and reuse connections.
    qdrant_pool_size: int = 10

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    collection_prefix: str = "coco"

    top_k_default: int = 20
    max_top_k: int = 100
    max_upload_mb: int = 10
    max_image_pixels: int = 4_000_000
    allowed_image_types: str = "image/jpeg,image/png,image/webp"

    model_cache_size: int = 2
    device: str = "cpu"
    batch_size: int = 32
    thumb_size: int = 256
    hnsw_ef_default: int = 128

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    random_seed: int = 42
    queryset_size: int = 200

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "val2017"

    @property
    def annotations_dir(self) -> Path:
        return self.data_dir / "annotations"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def index_meta_dir(self) -> Path:
        return self.data_dir / "index_meta"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25_index.pkl"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant_local"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_image_types_set(self) -> frozenset[str]:
        return frozenset(t.strip() for t in self.allowed_image_types.split(",") if t.strip())

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Returns the shared Settings instance, cached so .env isn't parsed repeatedly.

    Tests must call ``get_settings.cache_clear()`` after changing environment variables.
    """
    return Settings()
