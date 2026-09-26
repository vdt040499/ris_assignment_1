# Multimodal Semantic Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây hệ tìm kiếm ngữ nghĩa trên 5.000 ảnh COCO val2017, truy vấn bằng text tự do hoặc bằng ảnh, kèm bộ số liệu ablation 5 không gian nhúng + 2 baseline.

**Architecture:** Bốn thành phần tách biệt: `ingest` sinh `corpus.jsonl` (nguồn sự thật duy nhất), `build_index` encode và upsert vector vào Qdrant, `search-api` (FastAPI) encode query rồi truy vấn Qdrant, `eval` CLI đọc cùng index đó để sinh số liệu. Registry các "embedding space" là interface duy nhất giữa phần model và phần hệ thống — thêm model = thêm một entry, không sửa API/frontend/eval.

**Tech Stack:** Python 3.11, FastAPI, Qdrant (Docker), transformers 5.14, sentence-transformers 5.3, rank-bm25, numpy, Pillow, pytest; frontend Vite + React + TypeScript + Tailwind.

**Spec:** `docs/superpowers/specs/2026-09-23-multimodal-search-design.md`

## Global Constraints

- Máy **không có GPU**. `DEVICE=cpu` cho mọi encode. Không viết code đòi CUDA.
- **Không hardcode giá trị.** Mọi tham số (top-k, giới hạn upload, batch size, URL Qdrant, tên collection prefix, thumb size, hnsw_ef) đọc từ `backend/app/config.py` (pydantic-settings, nạp `.env`). Hằng số duy nhất được viết trong code là các giá trị thuộc bản chất model (dim, hf_id) và chúng sống trong `registry.py`.
- **Tái sử dụng trước khi viết mới.** Trước mỗi task, đọc các module đã có (`config.py`, `registry.py`, `corpus.py`, `vectordb.py`, `metrics.py`) và dùng lại; không viết lại hàm normalize, loader corpus, hay builder filter lần thứ hai.
- **JSDoc/docstring cho hàm phức tạp**: hàm có logic không hiển nhiên, nhiều tham số, hoặc giá trị trả về không rõ ràng phải có docstring nêu tham số, giá trị trả về, và lỗi có thể raise.
- **Frontend: không inline style.** Chỉ Tailwind. Theo đúng style/design đã có trong `frontend/src` khi thêm component mới.
- **Commit message: không có trailer `Co-Authored-By: Claude`.** Dùng tiền tố `feat:` / `test:` / `docs:` / `chore:`.
- Prefix collection Qdrant: `coco` (config `COLLECTION_PREFIX`), tên collection = `{prefix}_{suffix}`.
- Dim cố định theo model: `clip-b32` 512, `clip-b16` 512, `laion-b32` 512, `siglip-b16` 768, `mclip-b32` 512, `resnet50` 2048.
- Mã lỗi HTTP theo §3.6 của spec: 400 ảnh sai định dạng / chiều truy vấn không hỗ trợ / thiếu-thừa file+image_id, 404 space không tồn tại, 409 space chưa build index, 413 ảnh quá lớn, 422 validation, 503 Qdrant down.
- `pytest` mặc định **bỏ qua** test `integration` (tải model thật). CI mặc định chạy suite nhanh.

## Review Focus

Năm lớp đầu vào mà spec hàm ý nhưng không task nào tự nhiên test tới. Mỗi dòng đã được gắn một test vào task sở hữu đoạn code đó.

1. **Ảnh upload không phải RGB** (grayscale, RGBA có alpha, CMYK) hoặc có EXIF orientation → phải chuyển RGB và áp `exif_transpose` trước khi encode, không raise. *Test ở Task 8.*
2. **Ảnh upload rất lớn** (vd 6000×4000, dưới giới hạn byte nhưng khổng lồ về pixel) → phải downscale trước khi encode, không ngốn hết RAM. *Test ở Task 8.*
3. **Query text dài hơn 77 token** của CLIP → phải `truncation=True`, trả kết quả chứ không raise. *Test ở Task 6.*
4. **`k` lớn hơn số point trong collection** → trả về ít hơn `k` kết quả, không lỗi. *Test ở Task 10.*
5. **Filter khớp 0 point** → `results` rỗng với HTTP 200, không 500. *Test ở Task 10.*

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `docker-compose.yml` | Service Qdrant, tag image lấy từ biến môi trường |
| `.env.example` | Mọi tham số cấu hình kèm giá trị mặc định |
| `.gitignore` | Loại `data/` trừ ba query set, `node_modules`, `.env` |
| `requirements.txt` | Dependency Python có pin khoảng phiên bản |
| `pyproject.toml` | Cấu hình pytest (`pythonpath`, marker `integration`) |
| `tasks.py` | CLI gốc: `ingest`, `build`, `eval`, `serve`, `querysets` |
| `backend/app/config.py` | `Settings` + `get_settings()` + các đường dẫn dẫn xuất |
| `backend/app/errors.py` | Exception miền: `UnknownSpaceError`, `IndexNotBuiltError`, … |
| `backend/app/registry.py` | `SpaceSpec`, `SPACES`, `get_space`, `collection_name` |
| `backend/app/corpus.py` | `CorpusRecord`, `Corpus`, `load_corpus`, validate schema |
| `backend/app/vecutil.py` | `l2_normalize` (một chỗ duy nhất) |
| `backend/app/metrics.py` | `recall_at_k`, `mrr_at_k`, `precision_at_k`, `map_at_k`, `overlap_at_k` |
| `backend/app/vectordb.py` | `VectorStore` (server + embedded), `Hit`, `build_filter` |
| `backend/app/encoders/base.py` | Protocol `Encoder`, `LruEncoderCache` |
| `backend/app/encoders/hf_dual.py` | `HFDualEncoder` dùng cho CLIP, OpenCLIP-LAION, SigLIP |
| `backend/app/encoders/st_multilingual.py` | `MultilingualTextEncoder` (chỉ text) |
| `backend/app/encoders/resnet.py` | `ResNetEncoder` (chỉ ảnh) |
| `backend/app/encoders/bm25.py` | `Bm25Retriever` (build / load / search) |
| `backend/app/encoders/__init__.py` | `get_encoder(space_name)` + cache LRU theo config |
| `backend/app/imageio.py` | `load_upload_image` (RGB, EXIF, downscale, giới hạn) |
| `backend/app/schemas.py` | Pydantic request/response |
| `backend/app/search.py` | `SearchService`: điều phối encode → truy vấn → format |
| `backend/app/main.py` | `create_app()`, routes, exception handler, serve ảnh |
| `backend/cli/ingest.py` | Tải COCO, `build_corpus`, `make_thumbnails` |
| `backend/cli/build_index.py` | Build từng space, ghi `index_meta`, idempotent |
| `backend/cli/make_querysets.py` | Sinh ba query set với seed cố định |
| `backend/cli/evaluate.py` | Năm trục ablation → `results/*.csv` + `*.md` |
| `backend/tests/` | Unit, contract, integration, smoke |
| `frontend/src/api/client.ts` | Gọi API, kiểu TypeScript khớp `schemas.py` |
| `frontend/src/hooks/useSearch.ts` | State truy vấn, loading, lỗi |
| `frontend/src/components/*.tsx` | SearchBar, ModelSelect, AdvancedPanel, ExampleChips, ResultGrid, ResultCard, DetailModal, CompareView |
| `frontend/src/App.tsx` | Ghép ba khu: query, kết quả, so sánh |
| `docs/report.md` | Báo cáo cuối, số đọc từ `results/` |

---

### Task 1: Scaffold, cấu hình, exception miền

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.env.example`, `.gitignore`, `docker-compose.yml`
- Create: `backend/app/__init__.py`, `backend/app/config.py`, `backend/app/errors.py`
- Create: `backend/tests/__init__.py`, `backend/tests/test_config.py`

**Interfaces:**
- Consumes: không có (task đầu).
- Produces: `get_settings() -> Settings`; `Settings` với các field `qdrant_mode, qdrant_url, qdrant_timeout_s, data_dir, results_dir, collection_prefix, top_k_default, max_top_k, max_upload_mb, max_image_pixels, allowed_image_types, cors_origins, model_cache_size, device, batch_size, thumb_size, hnsw_ef_default, api_host, api_port, random_seed, queryset_size` và property `images_dir, annotations_dir, thumbs_dir, corpus_path, cache_dir, index_meta_dir, bm25_path, qdrant_path, max_upload_bytes, allowed_image_types_set, cors_origins_list`; các exception trong `backend/app/errors.py`: `SearchError` (gốc), `UnknownSpaceError`, `IndexNotBuiltError`, `ModeNotSupportedError`, `VectorStoreDownError`, `BadImageError`, `ImageTooLargeError`, `BadRequestError`, `CorpusError`.

- [ ] **Step 1: Tạo cây thư mục và file dependency**

```bash
mkdir -p backend/app/encoders backend/cli backend/tests/fixtures frontend results docs
touch backend/app/__init__.py backend/app/encoders/__init__.py backend/cli/__init__.py backend/tests/__init__.py
```

`requirements.txt`:

```
fastapi>=0.124,<1
uvicorn[standard]>=0.38,<1
pydantic>=2.9,<3
pydantic-settings>=2.5,<3
python-multipart>=0.0.9,<1
qdrant-client>=1.12,<2
transformers>=5.14,<6
sentence-transformers>=5.3,<6
torch>=2.8,<3
pillow>=11,<12
numpy>=2.2,<3
rank-bm25>=0.2.2,<1
requests>=2.32,<3
tqdm>=4.66,<5
pytest>=8,<10
httpx>=0.27,<1
```

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["backend", "."]
testpaths = ["backend/tests"]
markers = ["integration: cần model thật hoặc Qdrant server đang chạy"]
addopts = "-m 'not integration' -q"
```

`.gitignore`:

```
data/*
!data/queryset_vi.json
!data/queryset_i2i.json
!data/queryset_short.json
.env
__pycache__/
*.pyc
.pytest_cache/
frontend/node_modules/
frontend/dist/
```

`docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:${QDRANT_IMAGE_TAG:-v1.12.4}
    container_name: coco_search_qdrant
    ports:
      - "${QDRANT_HTTP_PORT:-6333}:6333"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
    restart: unless-stopped
```

`.env.example`:

```
QDRANT_IMAGE_TAG=v1.12.4
QDRANT_HTTP_PORT=6333
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
QDRANT_TIMEOUT_S=10
DATA_DIR=data
RESULTS_DIR=results
COLLECTION_PREFIX=coco
TOP_K_DEFAULT=20
MAX_TOP_K=100
MAX_UPLOAD_MB=10
MAX_IMAGE_PIXELS=4000000
ALLOWED_IMAGE_TYPES=image/jpeg,image/png,image/webp
MODEL_CACHE_SIZE=2
DEVICE=cpu
BATCH_SIZE=32
THUMB_SIZE=256
HNSW_EF_DEFAULT=128
API_HOST=127.0.0.1
API_PORT=8000
CORS_ORIGINS=http://localhost:5173
RANDOM_SEED=42
QUERYSET_SIZE=200
```

- [ ] **Step 2: Cài dependency**

Run: `pip install -r requirements.txt`
Expected: cài xong không lỗi. Môi trường này đã có sẵn phần lớn package; thực tế chỉ `rank-bm25` là mới.

- [ ] **Step 3: Viết test thất bại cho config**

`backend/tests/test_config.py`:

```python
from pathlib import Path

from app.config import Settings, get_settings


def test_defaults_match_env_example():
    s = Settings(_env_file=None)
    assert s.top_k_default == 20
    assert s.max_top_k == 100
    assert s.device == "cpu"
    assert s.collection_prefix == "coco"
    assert s.qdrant_mode == "server"


def test_derived_paths_hang_off_data_dir():
    s = Settings(_env_file=None, data_dir=Path("mydata"))
    assert s.corpus_path == Path("mydata/corpus.jsonl")
    assert s.images_dir == Path("mydata/val2017")
    assert s.annotations_dir == Path("mydata/annotations")
    assert s.thumbs_dir == Path("mydata/thumbs")
    assert s.cache_dir == Path("mydata/cache")
    assert s.index_meta_dir == Path("mydata/index_meta")
    assert s.bm25_path == Path("mydata/bm25_index.pkl")
    assert s.qdrant_path == Path("mydata/qdrant_local")


def test_csv_fields_parse_into_collections():
    s = Settings(_env_file=None, allowed_image_types="image/jpeg, image/png",
                 cors_origins="http://a.test,http://b.test")
    assert s.allowed_image_types_set == frozenset({"image/jpeg", "image/png"})
    assert s.cors_origins_list == ["http://a.test", "http://b.test"]


def test_upload_limit_converts_mb_to_bytes():
    s = Settings(_env_file=None, max_upload_mb=3)
    assert s.max_upload_bytes == 3 * 1024 * 1024


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("TOP_K_DEFAULT", "7")
    get_settings.cache_clear()
    assert get_settings().top_k_default == 7
    get_settings.cache_clear()
```

- [ ] **Step 4: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 5: Viết `backend/app/config.py`**

```python
"""Cấu hình toàn hệ. Mọi tham số chạy của hệ sống ở đây, không rải trong code."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tham số cấu hình, nạp theo thứ tự: biến môi trường > .env > mặc định."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    qdrant_mode: Literal["server", "embedded"] = "server"
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout_s: float = 10.0

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
    """Trả về Settings dùng chung, cache lại để không parse .env nhiều lần.

    Test phải gọi ``get_settings.cache_clear()`` sau khi đổi biến môi trường.
    """
    return Settings()
```

- [ ] **Step 6: Viết `backend/app/errors.py`**

```python
"""Exception miền. Tầng HTTP map các lớp này sang status code ở app/main.py."""


class SearchError(Exception):
    """Gốc của mọi lỗi miền trong hệ."""


class UnknownSpaceError(SearchError):
    """Tên embedding space không có trong registry → HTTP 404."""


class IndexNotBuiltError(SearchError):
    """Space hợp lệ nhưng chưa build index → HTTP 409."""


class ModeNotSupportedError(SearchError):
    """Space không hỗ trợ chiều truy vấn được yêu cầu → HTTP 400."""


class VectorStoreDownError(SearchError):
    """Không kết nối được Qdrant → HTTP 503."""


class BadImageError(SearchError):
    """Ảnh upload sai định dạng hoặc không giải mã được → HTTP 400."""


class ImageTooLargeError(SearchError):
    """Ảnh upload vượt giới hạn byte → HTTP 413."""


class BadRequestError(SearchError):
    """Request sai logic (vd thiếu cả file và image_id) → HTTP 400."""


class CorpusError(SearchError):
    """corpus.jsonl thiếu, rỗng, hoặc sai schema."""
```

- [ ] **Step 7: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_config.py -v`
Expected: PASS, 5 test.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pyproject.toml .env.example .gitignore docker-compose.yml backend/
git commit -m "feat: scaffold project, cấu hình pydantic-settings và exception miền"
```

---

### Task 2: Registry các embedding space

**Files:**
- Create: `backend/app/registry.py`
- Create: `backend/tests/test_registry.py`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.errors.UnknownSpaceError`.
- Produces: `SpaceSpec` (dataclass frozen với field `name, hf_id, dim, distance, normalized, backend, collection_suffix, modes, languages, encoder_key, text_padding, builds_index, note`); `SPACES: dict[str, SpaceSpec]`; `get_space(name: str) -> SpaceSpec`; `list_space_names() -> list[str]`; `collection_name(space: SpaceSpec, settings: Settings, target: str = "image") -> str`; hằng số `MODE_TEXT2IMAGE = "text2image"`, `MODE_IMAGE2IMAGE = "image2image"`, `MODE_IMAGE2TEXT = "image2text"`; `BACKEND_QDRANT = "qdrant"`, `BACKEND_BM25 = "bm25"`.

- [ ] **Step 1: Viết test thất bại**

`backend/tests/test_registry.py`:

```python
import pytest

from app.config import Settings
from app.errors import UnknownSpaceError
from app.registry import (
    BACKEND_BM25,
    BACKEND_QDRANT,
    MODE_IMAGE2IMAGE,
    MODE_TEXT2IMAGE,
    SPACES,
    collection_name,
    get_space,
    list_space_names,
)

KNOWN_ENCODERS = {"hf_dual", "st_multilingual", "resnet", "bm25"}


@pytest.fixture
def settings():
    return Settings(_env_file=None)


def test_all_eight_spaces_declared():
    assert set(list_space_names()) == {
        "clip-b32", "clip-b16", "laion-b32", "siglip-b16",
        "mclip-b32", "clip-b32-raw", "resnet50", "bm25-cap",
    }


def test_dict_key_matches_space_name():
    for key, space in SPACES.items():
        assert key == space.name


def test_every_space_uses_a_known_encoder():
    for space in SPACES.values():
        assert space.encoder_key in KNOWN_ENCODERS


def test_dims_match_spec():
    assert get_space("clip-b32").dim == 512
    assert get_space("clip-b16").dim == 512
    assert get_space("laion-b32").dim == 512
    assert get_space("siglip-b16").dim == 768
    assert get_space("mclip-b32").dim == 512
    assert get_space("resnet50").dim == 2048
    assert get_space("bm25-cap").dim is None


def test_unknown_space_raises():
    with pytest.raises(UnknownSpaceError):
        get_space("clip-b99")


def test_multilingual_reuses_baseline_image_collection(settings):
    assert collection_name(get_space("mclip-b32"), settings) == collection_name(
        get_space("clip-b32"), settings
    )
    assert get_space("mclip-b32").builds_index is False


def test_raw_space_is_unnormalized_dot(settings):
    raw = get_space("clip-b32-raw")
    assert raw.distance == "Dot"
    assert raw.normalized is False
    assert raw.builds_index is True
    assert collection_name(raw, settings) == "coco_clip_b32_raw"


def test_caption_target_gets_its_own_collection(settings):
    assert collection_name(get_space("clip-b32"), settings, target="caption") == (
        "coco_cap_clip_b32"
    )


def test_bm25_space_has_no_qdrant_collection(settings):
    space = get_space("bm25-cap")
    assert space.backend == BACKEND_BM25
    with pytest.raises(ValueError):
        collection_name(space, settings)


def test_siglip_needs_max_length_padding():
    assert get_space("siglip-b16").text_padding == "max_length"
    assert get_space("clip-b32").text_padding == "longest"


def test_modes_reflect_what_each_space_can_do():
    assert MODE_TEXT2IMAGE in get_space("clip-b32").modes
    assert get_space("resnet50").modes == (MODE_IMAGE2IMAGE,)
    assert get_space("mclip-b32").modes == (MODE_TEXT2IMAGE,)
    assert get_space("bm25-cap").modes == (MODE_TEXT2IMAGE,)


def test_qdrant_spaces_all_declare_dim_and_distance():
    for space in SPACES.values():
        if space.backend == BACKEND_QDRANT:
            assert space.dim is not None and space.distance is not None


def test_vietnamese_supported_only_by_multilingual():
    assert "vi" in get_space("mclip-b32").languages
    assert "vi" not in get_space("clip-b32").languages
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.registry'`

- [ ] **Step 3: Viết `backend/app/registry.py`**

```python
"""Registry các embedding space.

Đây là interface duy nhất giữa phần model và phần hệ thống: API, frontend và
eval chỉ biết tới SpaceSpec, không biết CLIP hay SigLIP là gì. Thêm một model
nghĩa là thêm một entry ở đây và không sửa gì khác.
"""

from dataclasses import dataclass

from app.config import Settings
from app.errors import UnknownSpaceError

MODE_TEXT2IMAGE = "text2image"
MODE_IMAGE2IMAGE = "image2image"
MODE_IMAGE2TEXT = "image2text"

BACKEND_QDRANT = "qdrant"
BACKEND_BM25 = "bm25"

DISTANCE_COSINE = "Cosine"
DISTANCE_DOT = "Dot"


@dataclass(frozen=True)
class SpaceSpec:
    """Khai báo một không gian nhúng.

    :param collection_suffix: hậu tố tên collection Qdrant; ``None`` nếu space
        không lưu trong Qdrant (vd BM25).
    :param encoder_key: chọn lớp encoder nào dựng space này.
    :param text_padding: chiến lược padding của tokenizer; SigLIP đòi
        ``max_length``, CLIP dùng ``longest``.
    :param builds_index: ``False`` nghĩa là space dùng lại collection ảnh của
        một space khác và bước build phải bỏ qua nó.
    """

    name: str
    hf_id: str | None
    dim: int | None
    distance: str | None
    normalized: bool
    backend: str
    collection_suffix: str | None
    modes: tuple[str, ...]
    languages: tuple[str, ...]
    encoder_key: str
    text_padding: str
    builds_index: bool
    note: str


SPACES: dict[str, SpaceSpec] = {
    "clip-b32": SpaceSpec(
        name="clip-b32",
        hf_id="openai/clip-vit-base-patch32",
        dim=512,
        distance=DISTANCE_COSINE,
        normalized=True,
        backend=BACKEND_QDRANT,
        collection_suffix="clip_b32",
        modes=(MODE_TEXT2IMAGE, MODE_IMAGE2IMAGE, MODE_IMAGE2TEXT),
        languages=("en",),
        encoder_key="hf_dual",
        text_padding="longest",
        builds_index=True,
        note="baseline",
    ),
    "clip-b16": SpaceSpec(
        name="clip-b16",
        hf_id="openai/clip-vit-base-patch16",
        dim=512,
        distance=DISTANCE_COSINE,
        normalized=True,
        backend=BACKEND_QDRANT,
        collection_suffix="clip_b16",
        modes=(MODE_TEXT2IMAGE, MODE_IMAGE2IMAGE),
        languages=("en",),
        encoder_key="hf_dual",
        text_padding="longest",
        builds_index=True,
        note="đổi kích thước patch so với baseline",
    ),
    "laion-b32": SpaceSpec(
        name="laion-b32",
        hf_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        dim=512,
        distance=DISTANCE_COSINE,
        normalized=True,
        backend=BACKEND_QDRANT,
        collection_suffix="laion_b32",
        modes=(MODE_TEXT2IMAGE, MODE_IMAGE2IMAGE),
        languages=("en",),
        encoder_key="hf_dual",
        text_padding="longest",
        builds_index=True,
        note="cùng kiến trúc baseline, đổi dữ liệu huấn luyện sang LAION-2B",
    ),
    "siglip-b16": SpaceSpec(
        name="siglip-b16",
        hf_id="google/siglip-base-patch16-224",
        dim=768,
        distance=DISTANCE_COSINE,
        normalized=True,
        backend=BACKEND_QDRANT,
        collection_suffix="siglip_b16",
        modes=(MODE_TEXT2IMAGE, MODE_IMAGE2IMAGE),
        languages=("en",),
        encoder_key="hf_dual",
        text_padding="max_length",
        builds_index=True,
        note="đổi hàm loss sang sigmoid",
    ),
    "mclip-b32": SpaceSpec(
        name="mclip-b32",
        hf_id="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        dim=512,
        distance=DISTANCE_COSINE,
        normalized=True,
        backend=BACKEND_QDRANT,
        collection_suffix="clip_b32",
        modes=(MODE_TEXT2IMAGE,),
        languages=("vi", "en", "multi"),
        encoder_key="st_multilingual",
        text_padding="longest",
        builds_index=False,
        note="text tower đa ngôn ngữ, dùng lại collection ảnh của clip-b32",
    ),
    "clip-b32-raw": SpaceSpec(
        name="clip-b32-raw",
        hf_id="openai/clip-vit-base-patch32",
        dim=512,
        distance=DISTANCE_DOT,
        normalized=False,
        backend=BACKEND_QDRANT,
        collection_suffix="clip_b32_raw",
        modes=(MODE_TEXT2IMAGE, MODE_IMAGE2IMAGE),
        languages=("en",),
        encoder_key="hf_dual",
        text_padding="longest",
        builds_index=True,
        note="nhóm đối chứng cho ablation normalize",
    ),
    "resnet50": SpaceSpec(
        name="resnet50",
        hf_id="microsoft/resnet-50",
        dim=2048,
        distance=DISTANCE_COSINE,
        normalized=True,
        backend=BACKEND_QDRANT,
        collection_suffix="resnet50",
        modes=(MODE_IMAGE2IMAGE,),
        languages=(),
        encoder_key="resnet",
        text_padding="longest",
        builds_index=True,
        note="baseline không multimodal",
    ),
    "bm25-cap": SpaceSpec(
        name="bm25-cap",
        hf_id=None,
        dim=None,
        distance=None,
        normalized=False,
        backend=BACKEND_BM25,
        collection_suffix=None,
        modes=(MODE_TEXT2IMAGE,),
        languages=("en",),
        encoder_key="bm25",
        text_padding="longest",
        builds_index=True,
        note="baseline không ngữ nghĩa, khớp từ khoá trên caption",
    ),
}


def get_space(name: str) -> SpaceSpec:
    """Tra một space theo tên.

    :raises UnknownSpaceError: nếu tên không có trong registry, kèm danh sách
        tên hợp lệ để người gọi sửa được ngay.
    """
    try:
        return SPACES[name]
    except KeyError as exc:
        raise UnknownSpaceError(
            f"Space '{name}' không tồn tại. Hợp lệ: {', '.join(sorted(SPACES))}"
        ) from exc


def list_space_names() -> list[str]:
    return sorted(SPACES)


def collection_name(space: SpaceSpec, settings: Settings, target: str = "image") -> str:
    """Tên collection Qdrant của một space.

    :param target: ``"image"`` cho collection ảnh, ``"caption"`` cho collection
        caption (dùng ở chiều ảnh→text).
    :raises ValueError: nếu space không lưu trong Qdrant, hoặc target không hợp lệ.
    """
    if space.backend != BACKEND_QDRANT or space.collection_suffix is None:
        raise ValueError(f"Space '{space.name}' không dùng Qdrant")
    if target not in ("image", "caption"):
        raise ValueError(f"target không hợp lệ: {target}")
    suffix = space.collection_suffix if target == "image" else f"cap_{space.collection_suffix}"
    return f"{settings.collection_prefix}_{suffix}"
```

- [ ] **Step 4: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_registry.py -v`
Expected: PASS, 13 test.

- [ ] **Step 5: Commit**

```bash
git add backend/app/registry.py backend/tests/test_registry.py
git commit -m "feat: registry 8 embedding space làm interface duy nhất giữa model và hệ thống"
```

---

### Task 3: Module corpus và CLI ingest

**Files:**
- Create: `backend/app/corpus.py`, `backend/cli/ingest.py`
- Create: `backend/tests/conftest.py`, `backend/tests/test_corpus.py`, `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.errors.CorpusError`.
- Produces: `CorpusRecord` (pydantic: `image_id: int`, `file_name: str`, `width: int`, `height: int`, `captions: list[str]`, `categories: list[str]`, `supercategories: list[str]`); `Corpus` (dataclass: `.records: list[CorpusRecord]`, `.by_image_id: dict[int, CorpusRecord]`, `.caption_pairs() -> list[tuple[int, int, str]]`, `.image_ids() -> list[int]`, `__len__`); `load_corpus(path: Path) -> Corpus`; `write_corpus(records: Iterable[CorpusRecord], path: Path) -> int`; `record_payload(record: CorpusRecord) -> dict`; và từ `backend/cli/ingest.py`: `IngestStats` (dataclass: `n_images, n_captions, n_without_category, min_captions, max_captions`), `build_corpus(annotations_dir: Path, out_path: Path) -> IngestStats`, `make_thumbnails(corpus: Corpus, images_dir: Path, thumbs_dir: Path, size: int) -> int`, `download_if_missing(url: str, dest: Path) -> bool`, `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Viết fixture COCO thu nhỏ**

`backend/tests/conftest.py`:

```python
import json
from pathlib import Path

import pytest
from PIL import Image

MINI_IMAGES = [
    {"id": 1, "file_name": "000000000001.jpg", "width": 80, "height": 60},
    {"id": 2, "file_name": "000000000002.jpg", "width": 60, "height": 90},
    {"id": 3, "file_name": "000000000003.jpg", "width": 40, "height": 40},
]


@pytest.fixture
def mini_annotations_dir(tmp_path: Path) -> Path:
    """Sinh cặp file annotation COCO thu nhỏ: 3 ảnh, ảnh #3 không có category."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    captions = {
        "images": MINI_IMAGES,
        "annotations": [
            {"image_id": 1, "id": 10, "caption": "a dog on a red sofa"},
            {"image_id": 1, "id": 11, "caption": "a brown dog sleeping"},
            {"image_id": 2, "id": 12, "caption": "two zebras in a field"},
            {"image_id": 3, "id": 13, "caption": "a slice of pizza"},
        ],
    }
    instances = {
        "images": MINI_IMAGES,
        "categories": [
            {"id": 18, "name": "dog", "supercategory": "animal"},
            {"id": 24, "name": "zebra", "supercategory": "animal"},
            {"id": 63, "name": "couch", "supercategory": "furniture"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 18},
            {"image_id": 1, "category_id": 63},
            {"image_id": 1, "category_id": 18},
            {"image_id": 2, "category_id": 24},
        ],
    }
    (ann_dir / "captions_val2017.json").write_text(json.dumps(captions), encoding="utf-8")
    (ann_dir / "instances_val2017.json").write_text(json.dumps(instances), encoding="utf-8")
    return ann_dir


@pytest.fixture
def mini_images_dir(tmp_path: Path) -> Path:
    """Sinh 3 file JPEG thật đúng kích thước khai báo trong annotation."""
    img_dir = tmp_path / "val2017"
    img_dir.mkdir()
    for spec in MINI_IMAGES:
        Image.new("RGB", (spec["width"], spec["height"]), color=(120, 60, 30)).save(
            img_dir / spec["file_name"], format="JPEG"
        )
    return img_dir
```

- [ ] **Step 2: Viết test thất bại cho corpus**

`backend/tests/test_corpus.py`:

```python
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
    assert "dòng 1" in str(exc.value)
```

- [ ] **Step 3: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.corpus'`

- [ ] **Step 4: Viết `backend/app/corpus.py`**

```python
"""Đọc/ghi corpus.jsonl — nguồn sự thật duy nhất của hệ.

Sau bước ingest, không thành phần nào đọc lại file COCO JSON gốc; tất cả đi
qua module này.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

from app.errors import CorpusError


class CorpusRecord(BaseModel):
    """Một ảnh trong corpus kèm caption và metadata category của nó."""

    image_id: int
    file_name: str
    width: int
    height: int
    captions: list[str]
    categories: list[str] = []
    supercategories: list[str] = []

    @field_validator("captions")
    @classmethod
    def _clean_captions(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("record phải có ít nhất một caption không rỗng")
        return cleaned


@dataclass
class Corpus:
    """Toàn bộ corpus nạp trong RAM, kèm chỉ mục tra nhanh theo image_id."""

    records: list[CorpusRecord]
    by_image_id: dict[int, CorpusRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_image_id = {r.image_id: r for r in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def image_ids(self) -> list[int]:
        return [r.image_id for r in self.records]

    def caption_pairs(self) -> list[tuple[int, int, str]]:
        """Dàn phẳng mọi caption thành ``(image_id, caption_index, text)``.

        Dùng cho cả việc build collection caption và việc sinh query set
        text→ảnh, nên hai bên luôn đánh cùng một chỉ số cho cùng một caption.
        """
        return [
            (rec.image_id, idx, text)
            for rec in self.records
            for idx, text in enumerate(rec.captions)
        ]


def record_payload(record: CorpusRecord) -> dict:
    """Payload gắn kèm mỗi point Qdrant. Giữ đúng các field cần để filter và hiển thị."""
    return {
        "image_id": record.image_id,
        "file_name": record.file_name,
        "width": record.width,
        "height": record.height,
        "captions": record.captions,
        "categories": record.categories,
        "supercategories": record.supercategories,
    }


def write_corpus(records: Iterable[CorpusRecord], path: Path) -> int:
    """Ghi corpus ra JSONL. Trả về số record đã ghi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
            count += 1
    return count


def load_corpus(path: Path) -> Corpus:
    """Nạp corpus.jsonl và validate từng dòng.

    :raises CorpusError: file không tồn tại, rỗng, hoặc có dòng sai schema —
        thông báo nêu rõ số dòng để sửa được ngay.
    """
    if not path.exists():
        raise CorpusError(f"Không thấy {path}. Chạy: python tasks.py ingest")
    records: list[CorpusRecord] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(CorpusRecord.model_validate_json(line))
            except ValidationError as exc:
                raise CorpusError(f"{path} sai schema ở dòng {lineno}: {exc}") from exc
    if not records:
        raise CorpusError(f"{path} rỗng. Chạy lại: python tasks.py ingest")
    return Corpus(records=records)
```

- [ ] **Step 5: Chạy test corpus để chắc nó pass**

Run: `python -m pytest backend/tests/test_corpus.py -v`
Expected: PASS, 9 test.

- [ ] **Step 6: Viết test thất bại cho ingest**

`backend/tests/test_ingest.py`:

```python
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
```

- [ ] **Step 7: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli.ingest'`

- [ ] **Step 8: Viết `backend/cli/ingest.py`**

```python
"""Tải COCO val2017 và sinh corpus.jsonl + thumbnail.

Chạy một lần. Mọi bước đều resume được: file đã tải không tải lại, thumbnail
đã có không sinh lại.
"""

import argparse
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageOps

from app.config import Settings, get_settings
from app.corpus import Corpus, CorpusRecord, load_corpus, write_corpus

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DOWNLOAD_CHUNK_BYTES = 1 << 20
CAPTIONS_FILE = "captions_val2017.json"
INSTANCES_FILE = "instances_val2017.json"


@dataclass
class IngestStats:
    """Số liệu đếm được từ dữ liệu thật, để báo cáo không phải giả định."""

    n_images: int
    n_captions: int
    n_without_category: int
    min_captions: int
    max_captions: int


def download_if_missing(url: str, dest: Path) -> bool:
    """Tải ``url`` về ``dest`` nếu chưa có. Trả True nếu lần này thật sự tải.

    Ghi vào file ``.part`` rồi mới đổi tên, nên một lần tải bị ngắt không để
    lại file hỏng trông như đã xong. Zip tải xong được kiểm tra tính toàn vẹn.
    """
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                fh.write(chunk)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(part) as zf:
            if zf.testzip() is not None:
                part.unlink()
                raise RuntimeError(f"Zip tải về bị hỏng: {url}")
    part.rename(dest)
    return True


def extract_zip(zip_path: Path, dest_dir: Path, marker: Path) -> bool:
    """Giải nén nếu ``marker`` chưa tồn tại. Trả True nếu lần này thật sự giải nén."""
    if marker.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return True


def build_corpus(annotations_dir: Path, out_path: Path) -> IngestStats:
    """Gộp caption và category của COCO thành corpus.jsonl.

    Ảnh không có annotation category vẫn được giữ với mảng rỗng — bỏ chúng đi
    sẽ làm lệch mẫu số của mọi metric sau này. Ảnh không có caption nào thì bị
    loại, vì nó không dùng được cho cả eval lẫn baseline BM25.
    """
    captions_raw = json.loads((annotations_dir / CAPTIONS_FILE).read_text(encoding="utf-8"))
    instances_raw = json.loads((annotations_dir / INSTANCES_FILE).read_text(encoding="utf-8"))

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in captions_raw["annotations"]:
        captions_by_image[ann["image_id"]].append(ann["caption"].strip())

    category_by_id = {c["id"]: c for c in instances_raw["categories"]}
    cats_by_image: dict[int, set[str]] = defaultdict(set)
    supercats_by_image: dict[int, set[str]] = defaultdict(set)
    for ann in instances_raw["annotations"]:
        cat = category_by_id.get(ann["category_id"])
        if cat is None:
            continue
        cats_by_image[ann["image_id"]].add(cat["name"])
        supercats_by_image[ann["image_id"]].add(cat["supercategory"])

    records: list[CorpusRecord] = []
    for img in sorted(captions_raw["images"], key=lambda i: i["id"]):
        image_id = img["id"]
        captions = [c for c in captions_by_image.get(image_id, []) if c]
        if not captions:
            continue
        records.append(
            CorpusRecord(
                image_id=image_id,
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
                captions=captions,
                categories=sorted(cats_by_image.get(image_id, set())),
                supercategories=sorted(supercats_by_image.get(image_id, set())),
            )
        )

    write_corpus(records, out_path)
    caption_counts = [len(r.captions) for r in records]
    return IngestStats(
        n_images=len(records),
        n_captions=sum(caption_counts),
        n_without_category=sum(1 for r in records if not r.categories),
        min_captions=min(caption_counts),
        max_captions=max(caption_counts),
    )


def make_thumbnails(corpus: Corpus, images_dir: Path, thumbs_dir: Path, size: int) -> int:
    """Sinh thumbnail cạnh dài ``size`` px. Trả về số thumbnail mới tạo lần này."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for record in corpus.records:
        target = thumbs_dir / record.file_name
        if target.exists():
            continue
        with Image.open(images_dir / record.file_name) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((size, size))
            img.save(target, format="JPEG", quality=85)
        made += 1
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tải COCO val2017 và sinh corpus.jsonl")
    parser.add_argument("--skip-download", action="store_true",
                        help="Bỏ qua bước tải, dùng file đã có trong DATA_DIR")
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        images_zip = settings.data_dir / "val2017.zip"
        ann_zip = settings.data_dir / "annotations_trainval2017.zip"
        download_if_missing(COCO_IMAGES_URL, images_zip)
        download_if_missing(COCO_ANNOTATIONS_URL, ann_zip)
        extract_zip(images_zip, settings.data_dir, marker=settings.images_dir)
        extract_zip(ann_zip, settings.data_dir,
                    marker=settings.annotations_dir / CAPTIONS_FILE)

    stats = build_corpus(settings.annotations_dir, settings.corpus_path)
    corpus = load_corpus(settings.corpus_path)
    made = make_thumbnails(corpus, settings.images_dir, settings.thumbs_dir,
                           settings.thumb_size)

    print(f"corpus: {stats.n_images} ảnh, {stats.n_captions} caption "
          f"(mỗi ảnh {stats.min_captions}-{stats.max_captions})")
    print(f"ảnh không có category: {stats.n_without_category}")
    print(f"thumbnail mới sinh: {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 9: Chạy test ingest để chắc nó pass**

Run: `python -m pytest backend/tests/test_ingest.py -v`
Expected: PASS, 7 test.

- [ ] **Step 10: Commit**

```bash
git add backend/app/corpus.py backend/cli/ingest.py backend/tests/conftest.py backend/tests/test_corpus.py backend/tests/test_ingest.py
git commit -m "feat: corpus.jsonl làm nguồn sự thật duy nhất và CLI ingest resume được"
```

---

### Task 4: Metric retrieval

**Files:**
- Create: `backend/app/metrics.py`, `backend/app/vecutil.py`
- Create: `backend/tests/test_metrics.py`, `backend/tests/test_vecutil.py`

**Interfaces:**
- Consumes: `numpy`.
- Produces: `l2_normalize(matrix: np.ndarray) -> np.ndarray` (trong `app/vecutil.py`); và trong `app/metrics.py`: `recall_at_k(rankings: Sequence[Sequence[int]], gold: Sequence[int], ks: Sequence[int]) -> dict[int, float]`, `mrr_at_k(rankings, gold, k: int) -> float`, `precision_at_k(ranked_labels: Sequence[Sequence[set[str]]], query_labels: Sequence[set[str]], k: int) -> float`, `map_at_k(ranked_labels, query_labels, k: int) -> float`, `overlap_at_k(rankings_a: Sequence[Sequence[int]], rankings_b: Sequence[Sequence[int]], k: int) -> float`.

- [ ] **Step 1: Viết test thất bại cho `l2_normalize`**

`backend/tests/test_vecutil.py`:

```python
import numpy as np
import pytest

from app.vecutil import l2_normalize


def test_rows_become_unit_length():
    matrix = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    out = l2_normalize(matrix)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)


def test_direction_is_preserved():
    matrix = np.array([[3.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(l2_normalize(matrix), [[0.6, 0.8]], atol=1e-6)


def test_zero_row_stays_zero_instead_of_nan():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = l2_normalize(matrix)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[0], [0.0, 0.0])


def test_output_is_float32_regardless_of_input_dtype():
    assert l2_normalize(np.array([[1.0, 1.0]], dtype=np.float64)).dtype == np.float32


def test_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        l2_normalize(np.array([1.0, 2.0], dtype=np.float32))
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_vecutil.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.vecutil'`

- [ ] **Step 3: Viết `backend/app/vecutil.py`**

```python
"""Tiện ích vector. Mọi chỗ cần L2-normalize đều gọi hàm ở đây, không tự viết lại."""

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Chuẩn hoá L2 theo từng hàng, trả về float32.

    Hàng có norm 0 được giữ nguyên là vector 0 thay vì thành NaN — một vector 0
    lọt vào index sẽ làm mọi truy vấn sau đó trả NaN và rất khó truy nguyên.

    :raises ValueError: nếu đầu vào không phải ma trận 2 chiều.
    """
    if matrix.ndim != 2:
        raise ValueError(f"cần ma trận 2 chiều, nhận được ndim={matrix.ndim}")
    out = matrix.astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, norms, out=out, where=norms > 0)
    return out
```

- [ ] **Step 4: Viết test thất bại cho metric**

`backend/tests/test_metrics.py`:

```python
import pytest

from app.metrics import map_at_k, mrr_at_k, overlap_at_k, precision_at_k, recall_at_k


def test_recall_counts_a_hit_anywhere_in_top_k():
    rankings = [[9, 8, 1], [2, 7, 7], [5, 5, 5]]
    gold = [1, 2, 99]
    out = recall_at_k(rankings, gold, ks=(1, 3))
    assert out[1] == pytest.approx(1 / 3)
    assert out[3] == pytest.approx(2 / 3)


def test_recall_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        recall_at_k([[1]], [1, 2], ks=(1,))


def test_mrr_uses_reciprocal_of_first_hit_position():
    rankings = [[5, 1], [2, 9], [7, 7]]
    gold = [1, 2, 3]
    assert mrr_at_k(rankings, gold, k=2) == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_mrr_ignores_hits_beyond_k():
    assert mrr_at_k([[9, 9, 1]], [1], k=2) == pytest.approx(0.0)


def test_precision_at_k_counts_label_overlap():
    ranked_labels = [[{"dog"}, {"cat"}, {"dog", "sofa"}, set()]]
    query_labels = [{"dog"}]
    assert precision_at_k(ranked_labels, query_labels, k=4) == pytest.approx(0.5)


def test_precision_at_k_with_no_query_labels_is_zero():
    assert precision_at_k([[{"dog"}]], [set()], k=1) == pytest.approx(0.0)


def test_map_at_k_weights_early_hits_more():
    early = map_at_k([[{"dog"}, set(), set()]], [{"dog"}], k=3)
    late = map_at_k([[set(), set(), {"dog"}]], [{"dog"}], k=3)
    assert early > late
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(1 / 3)


def test_map_at_k_is_zero_when_nothing_relevant_retrieved():
    assert map_at_k([[set(), set()]], [{"dog"}], k=2) == pytest.approx(0.0)


def test_overlap_measures_agreement_between_two_rankings():
    a = [[1, 2, 3, 4]]
    b = [[3, 4, 5, 6]]
    assert overlap_at_k(a, b, k=4) == pytest.approx(0.5)


def test_overlap_is_one_for_identical_rankings():
    assert overlap_at_k([[1, 2]], [[2, 1]], k=2) == pytest.approx(1.0)
```

- [ ] **Step 5: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.metrics'`

- [ ] **Step 6: Viết `backend/app/metrics.py`**

```python
"""Metric retrieval. Hàm thuần, không biết gì về Qdrant hay model.

Quy ước chung: ``rankings[i]`` là danh sách image_id đã xếp hạng cho query
thứ ``i``, phần tử đầu là kết quả hạng 1.
"""

from collections.abc import Sequence


def _check_lengths(rankings: Sequence, gold: Sequence) -> None:
    if len(rankings) != len(gold):
        raise ValueError(
            f"số ranking ({len(rankings)}) khác số nhãn đúng ({len(gold)})"
        )


def recall_at_k(
    rankings: Sequence[Sequence[int]], gold: Sequence[int], ks: Sequence[int]
) -> dict[int, float]:
    """Recall@k cho bài toán một-nhãn-đúng (mỗi caption có đúng một ảnh gốc).

    :return: dict ``{k: tỉ lệ query có ảnh gốc nằm trong top-k}``.
    :raises ValueError: nếu số ranking khác số nhãn đúng.
    """
    _check_lengths(rankings, gold)
    result: dict[int, float] = {}
    total = len(gold) or 1
    for k in ks:
        hits = sum(1 for ranked, want in zip(rankings, gold) if want in ranked[:k])
        result[k] = hits / total
    return result


def mrr_at_k(rankings: Sequence[Sequence[int]], gold: Sequence[int], k: int) -> float:
    """Mean Reciprocal Rank, tính trong top-k; query không có hit đóng góp 0."""
    _check_lengths(rankings, gold)
    total = 0.0
    for ranked, want in zip(rankings, gold):
        for position, item in enumerate(ranked[:k], start=1):
            if item == want:
                total += 1.0 / position
                break
    return total / (len(gold) or 1)


def precision_at_k(
    ranked_labels: Sequence[Sequence[set[str]]],
    query_labels: Sequence[set[str]],
    k: int,
) -> float:
    """P@k cho ảnh→ảnh: một kết quả tính là liên quan nếu chia sẻ ≥1 nhãn với query.

    Đây là proxy, không phải ground-truth thật; nó thiên vị ảnh nhiều object.
    Query không có nhãn nào đóng góp 0.
    """
    _check_lengths(ranked_labels, query_labels)
    total = 0.0
    for ranked, want in zip(ranked_labels, query_labels):
        top = ranked[:k]
        if not top:
            continue
        relevant = sum(1 for labels in top if want and labels & want)
        total += relevant / len(top)
    return total / (len(query_labels) or 1)


def map_at_k(
    ranked_labels: Sequence[Sequence[set[str]]],
    query_labels: Sequence[set[str]],
    k: int,
) -> float:
    """mAP@k với độ liên quan = có chia sẻ nhãn.

    AP@k ở đây chuẩn hoá theo **số item liên quan tìm được trong top-k**, không
    theo tổng số item liên quan trong corpus (con số đó vô nghĩa với proxy
    nhãn, vì hàng nghìn ảnh cùng chứa 'person'). Định nghĩa này phải được ghi
    đúng như vậy trong báo cáo.
    """
    _check_lengths(ranked_labels, query_labels)
    total = 0.0
    for ranked, want in zip(ranked_labels, query_labels):
        hits = 0
        precision_sum = 0.0
        for position, labels in enumerate(ranked[:k], start=1):
            if want and labels & want:
                hits += 1
                precision_sum += hits / position
        if hits:
            total += precision_sum / hits
    return total / (len(query_labels) or 1)


def overlap_at_k(
    rankings_a: Sequence[Sequence[int]], rankings_b: Sequence[Sequence[int]], k: int
) -> float:
    """Tỉ lệ trùng nhau giữa hai top-k, bình quân trên các query.

    Dùng cho trục ANN vs exact: ``rankings_b`` là kết quả exact làm chuẩn vàng.
    """
    _check_lengths(rankings_a, rankings_b)
    total = 0.0
    for first, second in zip(rankings_a, rankings_b):
        top_a, top_b = set(first[:k]), set(second[:k])
        denominator = max(len(top_a), len(top_b)) or 1
        total += len(top_a & top_b) / denominator
    return total / (len(rankings_a) or 1)
```

- [ ] **Step 7: Chạy cả hai file test để chắc nó pass**

Run: `python -m pytest backend/tests/test_vecutil.py backend/tests/test_metrics.py -v`
Expected: PASS, 15 test.

- [ ] **Step 8: Commit**

```bash
git add backend/app/metrics.py backend/app/vecutil.py backend/tests/test_metrics.py backend/tests/test_vecutil.py
git commit -m "feat: metric retrieval và l2_normalize dùng chung"
```

---

### Task 5: Wrapper Qdrant và builder filter

**Files:**
- Create: `backend/app/vectordb.py`
- Create: `backend/tests/test_vectordb.py`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.errors.VectorStoreDownError`, `qdrant_client`.
- Produces: `Hit` (dataclass frozen: `id: int`, `score: float`, `payload: dict`); `build_filter(categories: Sequence[str] | None, supercategories: Sequence[str] | None) -> models.Filter | None`; `VectorStore` với `__init__(settings: Settings, client: QdrantClient | None = None)`, `.mode -> str`, `.health() -> str`, `.collection_exists(name: str) -> bool`, `.count(name: str) -> int`, `.ensure_collection(name: str, dim: int, distance: str, recreate: bool = False) -> bool`, `.create_payload_indexes(name: str, fields: Sequence[str]) -> None`, `.upsert(name: str, ids: Sequence[int], vectors: np.ndarray, payloads: Sequence[dict], batch_size: int) -> int`, `.search(name: str, vector: np.ndarray, k: int, exact: bool = True, hnsw_ef: int | None = None, query_filter: models.Filter | None = None) -> list[Hit]`.

- [ ] **Step 1: Viết test thất bại**

`backend/tests/test_vectordb.py`:

```python
import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.vectordb import Hit, VectorStore, build_filter

COLLECTION = "test_images"
DIM = 4


@pytest.fixture
def store():
    settings = Settings(_env_file=None, qdrant_mode="embedded")
    return VectorStore(settings, client=QdrantClient(location=":memory:"))


@pytest.fixture
def populated(store):
    store.ensure_collection(COLLECTION, dim=DIM, distance="Cosine")
    vectors = np.eye(3, DIM, dtype=np.float32)
    payloads = [
        {"image_id": 1, "categories": ["dog"], "supercategories": ["animal"]},
        {"image_id": 2, "categories": ["zebra"], "supercategories": ["animal"]},
        {"image_id": 3, "categories": [], "supercategories": []},
    ]
    store.upsert(COLLECTION, ids=[1, 2, 3], vectors=vectors, payloads=payloads,
                 batch_size=2)
    store.create_payload_indexes(COLLECTION, ["categories", "supercategories"])
    return store


def test_no_filters_means_no_filter_object():
    assert build_filter(None, None) is None
    assert build_filter([], []) is None


def test_category_filter_matches_any_of_the_given_values():
    flt = build_filter(["dog", "cat"], None)
    assert flt is not None
    assert len(flt.must) == 1
    assert flt.must[0].key == "categories"


def test_both_filters_produce_two_conditions():
    flt = build_filter(["dog"], ["animal"])
    assert {c.key for c in flt.must} == {"categories", "supercategories"}


def test_ensure_collection_is_idempotent(store):
    assert store.ensure_collection(COLLECTION, dim=DIM, distance="Cosine") is True
    assert store.ensure_collection(COLLECTION, dim=DIM, distance="Cosine") is False
    assert store.collection_exists(COLLECTION) is True


def test_unknown_distance_is_rejected(store):
    with pytest.raises(ValueError):
        store.ensure_collection("bad", dim=DIM, distance="Manhattan")


def test_upsert_counts_every_point_across_batches(populated):
    assert populated.count(COLLECTION) == 3


def test_search_returns_hits_ordered_by_score(populated):
    hits = populated.search(COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=3)
    assert [h.id for h in hits][0] == 1
    assert all(isinstance(h, Hit) for h in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_search_carries_payload_back(populated):
    hits = populated.search(COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=1)
    assert hits[0].payload["image_id"] == 1


def test_search_honours_category_filter(populated):
    hits = populated.search(
        COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=3,
        query_filter=build_filter(["zebra"], None),
    )
    assert [h.id for h in hits] == [2]


def test_search_with_filter_matching_nothing_returns_empty(populated):
    hits = populated.search(
        COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=3,
        query_filter=build_filter(["unicorn"], None),
    )
    assert hits == []


def test_approximate_search_is_accepted(populated):
    hits = populated.search(COLLECTION, np.eye(1, DIM, dtype=np.float32)[0], k=2,
                            exact=False, hnsw_ef=64)
    assert len(hits) <= 2


def test_count_on_missing_collection_is_zero(store):
    assert store.count("never_created") == 0


def test_health_reports_ok_for_a_live_client(store):
    assert store.health() == "ok"


def test_mode_reflects_settings(store):
    assert store.mode == "embedded"
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_vectordb.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.vectordb'`

- [ ] **Step 3: Viết `backend/app/vectordb.py`**

```python
"""Wrapper quanh Qdrant.

Đây là module duy nhất trong hệ import ``qdrant_client``. Mọi thành phần khác
làm việc với ``Hit`` và ``build_filter``, nên đổi vector DB về sau chỉ phải sửa
một file.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import Settings
from app.errors import VectorStoreDownError

DISTANCE_MAP = {
    "Cosine": models.Distance.COSINE,
    "Dot": models.Distance.DOT,
}

CONNECTION_ERRORS = (ResponseHandlingException, UnexpectedResponse, OSError)


@dataclass(frozen=True)
class Hit:
    """Một kết quả truy vấn: id point, điểm tương đồng, và payload kèm theo."""

    id: int
    score: float
    payload: dict


def build_filter(
    categories: Sequence[str] | None, supercategories: Sequence[str] | None
) -> models.Filter | None:
    """Dựng filter payload Qdrant từ danh sách category người dùng chọn.

    Nhiều giá trị trong cùng một field là quan hệ HOẶC (``MatchAny``); giữa hai
    field là quan hệ VÀ. Trả ``None`` khi không có filter nào, để người gọi
    truyền thẳng vào Qdrant mà không cần phân nhánh.
    """
    conditions: list[models.FieldCondition] = []
    if categories:
        conditions.append(
            models.FieldCondition(
                key="categories", match=models.MatchAny(any=list(categories))
            )
        )
    if supercategories:
        conditions.append(
            models.FieldCondition(
                key="supercategories", match=models.MatchAny(any=list(supercategories))
            )
        )
    return models.Filter(must=conditions) if conditions else None


class VectorStore:
    """Truy cập Qdrant ở chế độ server (Docker) hoặc embedded (không Docker).

    Chế độ embedded luôn chạy brute-force và bỏ qua cấu hình HNSW, nên nó chỉ
    dùng để demo và để test — không dùng để sinh số liệu cho trục ANN vs exact.
    """

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self._settings = settings
        if client is not None:
            self._client = client
        elif settings.qdrant_mode == "server":
            self._client = QdrantClient(
                url=settings.qdrant_url, timeout=settings.qdrant_timeout_s
            )
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(settings.qdrant_path))

    @property
    def mode(self) -> str:
        return self._settings.qdrant_mode

    def health(self) -> str:
        """Trả ``"ok"`` nếu gọi được Qdrant, ``"down"`` nếu không."""
        try:
            self._client.get_collections()
        except CONNECTION_ERRORS:
            return "down"
        return "ok"

    def collection_exists(self, name: str) -> bool:
        try:
            return self._client.collection_exists(name)
        except CONNECTION_ERRORS as exc:
            raise VectorStoreDownError(str(exc)) from exc

    def count(self, name: str) -> int:
        """Số point trong collection; 0 nếu collection chưa tồn tại."""
        if not self.collection_exists(name):
            return 0
        return self._client.count(name, exact=True).count

    def ensure_collection(
        self, name: str, dim: int, distance: str, recreate: bool = False
    ) -> bool:
        """Tạo collection nếu chưa có. Trả True nếu lần này thật sự tạo.

        :raises ValueError: nếu tên distance không nằm trong DISTANCE_MAP.
        """
        if distance not in DISTANCE_MAP:
            raise ValueError(
                f"distance '{distance}' không hỗ trợ; hợp lệ: {sorted(DISTANCE_MAP)}"
            )
        exists = self.collection_exists(name)
        if exists and not recreate:
            return False
        if exists:
            self._client.delete_collection(name)
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=DISTANCE_MAP[distance]),
        )
        return True

    def create_payload_indexes(self, name: str, fields: Sequence[str]) -> None:
        """Tạo payload index keyword cho các field dùng để filter."""
        for field in fields:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert(
        self,
        name: str,
        ids: Sequence[int],
        vectors: np.ndarray,
        payloads: Sequence[dict],
        batch_size: int,
    ) -> int:
        """Upsert theo batch. Trả về số point đã gửi.

        :raises ValueError: nếu số id, số vector và số payload không khớp nhau.
        """
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"lệch số lượng: {len(ids)} id, {len(vectors)} vector, "
                f"{len(payloads)} payload"
            )
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            points = [
                models.PointStruct(id=int(pid), vector=vec.tolist(), payload=payload)
                for pid, vec, payload in zip(
                    ids[start:stop], vectors[start:stop], payloads[start:stop]
                )
            ]
            self._client.upsert(collection_name=name, points=points, wait=True)
        return len(ids)

    def search(
        self,
        name: str,
        vector: np.ndarray,
        k: int,
        exact: bool = True,
        hnsw_ef: int | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[Hit]:
        """Truy vấn top-k.

        :param exact: ``True`` bỏ qua HNSW và quét toàn bộ — đây là chuẩn vàng
            cho số liệu trong báo cáo. ``False`` dùng HNSW với ``hnsw_ef``.
        :raises VectorStoreDownError: nếu không gọi được Qdrant.
        """
        params = models.SearchParams(exact=exact, hnsw_ef=None if exact else hnsw_ef)
        try:
            response = self._client.query_points(
                collection_name=name,
                query=vector.tolist(),
                limit=k,
                query_filter=query_filter,
                search_params=params,
                with_payload=True,
            )
        except CONNECTION_ERRORS as exc:
            raise VectorStoreDownError(
                f"Không truy vấn được Qdrant ({self._settings.qdrant_url}). "
                "Chạy: docker compose up -d qdrant"
            ) from exc
        return [
            Hit(id=int(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]
```

- [ ] **Step 4: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_vectordb.py -v`
Expected: PASS, 14 test.

- [ ] **Step 5: Commit**

```bash
git add backend/app/vectordb.py backend/tests/test_vectordb.py
git commit -m "feat: wrapper Qdrant với chế độ exact/HNSW và filter metadata"
```

---

### Task 6: Encoder và cache LRU

**Files:**
- Create: `backend/app/encoders/base.py`, `backend/app/encoders/hf_dual.py`, `backend/app/encoders/st_multilingual.py`, `backend/app/encoders/resnet.py`
- Modify: `backend/app/encoders/__init__.py` (đang rỗng từ Task 1)
- Create: `backend/tests/test_encoders.py`, `backend/tests/test_encoders_integration.py`

**Interfaces:**
- Consumes: `app.registry.SpaceSpec`/`get_space`, `app.config.Settings`/`get_settings`, `app.vecutil.l2_normalize`, `app.errors.ModeNotSupportedError`.
- Produces: `Encoder` protocol (`.name: str`, `.dim: int`, `.encode_texts(texts: Sequence[str]) -> np.ndarray`, `.encode_images(images: Sequence[PIL.Image.Image]) -> np.ndarray`, cả hai trả ma trận `(n, dim)` float32); `LruEncoderCache(maxsize: int, factory: Callable[[str], Encoder])` với `.get(key) -> Encoder`, `.keys() -> list[str]`, `.clear()`, `__len__`; `HFDualEncoder(space, device, batch_size)`; `MultilingualTextEncoder(space, device, batch_size)`; `ResNetEncoder(space, device, batch_size)`; `build_encoder(space_name: str, settings: Settings | None = None) -> Encoder`; `get_encoder(space_name: str, settings: Settings | None = None) -> Encoder`; `reset_encoder_cache() -> None`.
- **Hợp đồng quan trọng:** encoder tự áp L2-normalize khi `space.normalized` là True. Nhờ vậy build index và search không thể xử lý khác nhau — chỉ có một chỗ quyết định.

- [ ] **Step 1: Viết test thất bại (không tải model)**

`backend/tests/test_encoders.py`:

```python
import threading

import numpy as np
import pytest

from app.config import Settings
from app.encoders import build_encoder, get_encoder, reset_encoder_cache
from app.encoders.base import LruEncoderCache
from app.errors import ModeNotSupportedError, UnknownSpaceError
from app.registry import get_space


class FakeEncoder:
    def __init__(self, name):
        self.name = name
        self.dim = 4

    def encode_texts(self, texts):
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def encode_images(self, images):
        return np.zeros((len(images), self.dim), dtype=np.float32)


@pytest.fixture
def settings():
    return Settings(_env_file=None, model_cache_size=2)


@pytest.fixture(autouse=True)
def clean_cache():
    reset_encoder_cache()
    yield
    reset_encoder_cache()


def test_cache_returns_the_same_instance_for_one_key():
    cache = LruEncoderCache(2, FakeEncoder)
    assert cache.get("a") is cache.get("a")
    assert len(cache) == 1


def test_cache_evicts_the_least_recently_used():
    cache = LruEncoderCache(2, FakeEncoder)
    first = cache.get("a")
    cache.get("b")
    cache.get("c")
    assert cache.keys() == ["b", "c"]
    assert cache.get("a") is not first


def test_reading_a_key_makes_it_recent():
    cache = LruEncoderCache(2, FakeEncoder)
    a = cache.get("a")
    cache.get("b")
    assert cache.get("a") is a
    cache.get("c")
    assert cache.keys() == ["a", "c"]


def test_maxsize_below_one_is_rejected():
    with pytest.raises(ValueError):
        LruEncoderCache(0, FakeEncoder)


def test_concurrent_gets_build_the_encoder_once():
    calls: list[str] = []

    def counting_factory(name):
        calls.append(name)
        return FakeEncoder(name)

    cache = LruEncoderCache(2, counting_factory)
    threads = [threading.Thread(target=cache.get, args=("a",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["a"]


def test_bm25_space_has_no_embedding_encoder(settings):
    with pytest.raises(ValueError):
        build_encoder("bm25-cap", settings)


def test_unknown_space_raises(settings):
    with pytest.raises(UnknownSpaceError):
        build_encoder("clip-b99", settings)


def test_resnet_refuses_text_before_loading_any_model(settings):
    encoder = build_encoder("resnet50", settings)
    with pytest.raises(ModeNotSupportedError):
        encoder.encode_texts(["a dog"])


def test_multilingual_refuses_images_before_loading_any_model(settings):
    encoder = build_encoder("mclip-b32", settings)
    with pytest.raises(ModeNotSupportedError):
        encoder.encode_images([object()])


def test_empty_input_returns_empty_matrix_without_loading(settings):
    encoder = build_encoder("clip-b32", settings)
    assert encoder.encode_texts([]).shape == (0, get_space("clip-b32").dim)
    assert encoder.encode_images([]).shape == (0, get_space("clip-b32").dim)


def test_get_encoder_uses_the_shared_cache(settings):
    assert get_encoder("clip-b32", settings) is get_encoder("clip-b32", settings)
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.encoders.base'`

- [ ] **Step 3: Viết `backend/app/encoders/base.py`**

```python
"""Giao diện encoder và cache LRU dùng chung."""

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np


class Encoder(Protocol):
    """Mọi encoder trả ma trận ``(n, dim)`` float32, đã normalize nếu space yêu cầu."""

    name: str
    dim: int

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_images(self, images: Sequence[Any]) -> np.ndarray: ...


class LruEncoderCache:
    """Giữ tối đa ``maxsize`` encoder trong RAM, loại bỏ cái ít dùng nhất.

    Việc tạo encoder xảy ra **bên trong** lock. Điều đó khiến một request phải
    chờ trong lúc model đang nạp, nhưng đổi lại hai request đồng thời cho cùng
    một model không nạp hai bản — với model 600MB trên máy 16GB thì tránh nhân
    đôi bộ nhớ quan trọng hơn tránh chờ.
    """

    def __init__(self, maxsize: int, factory: Callable[[str], Encoder]) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize phải >= 1, nhận được {maxsize}")
        self._maxsize = maxsize
        self._factory = factory
        self._items: OrderedDict[str, Encoder] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Encoder:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            encoder = self._factory(key)
            self._items[key] = encoder
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)
            return encoder

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
```

- [ ] **Step 4: Viết `backend/app/encoders/hf_dual.py`**

```python
"""Encoder cho họ CLIP/SigLIP tải qua transformers.

Một lớp phục vụ cả openai/clip, OpenCLIP-LAION và SigLIP: ba model này khác
nhau ở checkpoint và chiến lược padding, không khác nhau ở cách gọi.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class HFDualEncoder:
    """Encode cả text và ảnh vào cùng một không gian.

    Model được nạp lười ở lần encode đầu tiên, nên khởi tạo lớp này rẻ và các
    test kiểm tra hợp đồng không phải tải hàng trăm MB.
    """

    def __init__(self, space: SpaceSpec, device: str, batch_size: int) -> None:
        self.name = space.name
        self.dim = space.dim
        self._space = space
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._model = AutoModel.from_pretrained(self._space.hf_id).to(self._device).eval()
        self._processor = AutoProcessor.from_pretrained(self._space.hf_id)

    def _finalize(self, vectors: np.ndarray) -> np.ndarray:
        """Kiểm tra dim rồi normalize nếu space yêu cầu.

        :raises ValueError: nếu dim thực tế khác dim khai báo trong registry —
            lệch dim mà lọt vào Qdrant sẽ thành lỗi rất khó truy nguyên.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: model trả dim {vectors.shape[1]}, registry khai {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode danh sách câu. Câu dài hơn giới hạn token của model bị cắt bớt."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        import torch

        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            inputs = self._processor(
                text=batch,
                padding=self._space.text_padding,
                truncation=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
            chunks.append(features.cpu().numpy())
        return self._finalize(np.concatenate(chunks, axis=0))

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Encode danh sách ảnh PIL đã ở chế độ RGB."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        import torch

        chunks: list[np.ndarray] = []
        for start in range(0, len(images), self._batch_size):
            batch = list(images[start : start + self._batch_size])
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
            chunks.append(features.cpu().numpy())
        return self._finalize(np.concatenate(chunks, axis=0))
```

- [ ] **Step 5: Viết `backend/app/encoders/st_multilingual.py`**

```python
"""Text tower đa ngôn ngữ, distill để nằm cùng không gian với CLIP ViT-B/32.

Chỉ encode text. Phía ảnh dùng lại collection của clip-b32 — giả định này được
test `test_multilingual_shares_the_clip_space` canh giữ.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.errors import ModeNotSupportedError
from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class MultilingualTextEncoder:
    def __init__(self, space: SpaceSpec, device: str, batch_size: int) -> None:
        self.name = space.name
        self.dim = space.dim
        self._space = space
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._space.hf_id, device=self._device)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, convert_to_numpy=True
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: model trả dim {vectors.shape[1]}, registry khai {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        raise ModeNotSupportedError(
            f"{self.name} chỉ encode text; phía ảnh dùng collection của clip-b32"
        )
```

- [ ] **Step 6: Viết `backend/app/encoders/resnet.py`**

```python
"""Baseline không multimodal: feature ImageNet của ResNet-50. Chỉ encode ảnh."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.errors import ModeNotSupportedError
from app.registry import SpaceSpec
from app.vecutil import l2_normalize


class ResNetEncoder:
    def __init__(self, space: SpaceSpec, device: str, batch_size: int) -> None:
        self.name = space.name
        self.dim = space.dim
        self._space = space
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        self._model = AutoModel.from_pretrained(self._space.hf_id).to(self._device).eval()
        self._processor = AutoImageProcessor.from_pretrained(self._space.hf_id)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise ModeNotSupportedError(
            f"{self.name} là baseline chỉ dùng cho ảnh→ảnh, không encode text"
        )

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        """Lấy pooled feature 2048 chiều, dàn phẳng từ shape (n, 2048, 1, 1)."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        import torch

        chunks: list[np.ndarray] = []
        for start in range(0, len(images), self._batch_size):
            batch = list(images[start : start + self._batch_size])
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            pooled = outputs.pooler_output
            chunks.append(pooled.reshape(pooled.shape[0], -1).cpu().numpy())
        vectors = np.concatenate(chunks, axis=0).astype(np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: model trả dim {vectors.shape[1]}, registry khai {self.dim}"
            )
        return l2_normalize(vectors) if self._space.normalized else vectors
```

- [ ] **Step 7: Viết `backend/app/encoders/__init__.py`**

```python
"""Factory encoder theo tên space, kèm cache LRU dùng chung cho cả API."""

import threading

from app.config import Settings, get_settings
from app.encoders.base import Encoder, LruEncoderCache
from app.encoders.hf_dual import HFDualEncoder
from app.encoders.resnet import ResNetEncoder
from app.encoders.st_multilingual import MultilingualTextEncoder
from app.registry import get_space

_ENCODER_CLASSES = {
    "hf_dual": HFDualEncoder,
    "st_multilingual": MultilingualTextEncoder,
    "resnet": ResNetEncoder,
}

_cache: LruEncoderCache | None = None
_cache_lock = threading.Lock()


def build_encoder(space_name: str, settings: Settings | None = None) -> Encoder:
    """Tạo một encoder mới, không qua cache.

    :raises UnknownSpaceError: tên space không có trong registry.
    :raises ValueError: space không dùng encoder nhúng (vd ``bm25-cap``, được
        phục vụ bởi ``app.encoders.bm25.Bm25Retriever`` chứ không phải ở đây).
    """
    settings = settings or get_settings()
    space = get_space(space_name)
    try:
        encoder_class = _ENCODER_CLASSES[space.encoder_key]
    except KeyError as exc:
        raise ValueError(
            f"Space '{space_name}' dùng backend '{space.backend}', không có encoder nhúng"
        ) from exc
    return encoder_class(space, settings.device, settings.batch_size)


def get_encoder(space_name: str, settings: Settings | None = None) -> Encoder:
    """Lấy encoder từ cache LRU dùng chung; nạp model nếu chưa có trong cache."""
    global _cache
    settings = settings or get_settings()
    with _cache_lock:
        if _cache is None:
            _cache = LruEncoderCache(
                settings.model_cache_size,
                lambda name: build_encoder(name, settings),
            )
    return _cache.get(space_name)


def reset_encoder_cache() -> None:
    """Xoá cache. Dùng trong test và khi cần giải phóng RAM."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.clear()
        _cache = None
```

- [ ] **Step 8: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_encoders.py -v`
Expected: PASS, 11 test.

- [ ] **Step 9: Viết test integration, gồm cửa kiểm chứng giả định của spec §5.3**

`backend/tests/test_encoders_integration.py`:

```python
"""Test tải model thật. Chạy bằng: python -m pytest -m integration

Test `test_multilingual_shares_the_clip_space` là **cửa chặn**: nếu nó đỏ thì
giả định "mclip-b32 dùng lại collection ảnh của clip-b32" sai, và phải sửa
registry cho mclip-b32 có collection riêng TRƯỚC khi build index.
"""

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.encoders import build_encoder

pytestmark = pytest.mark.integration

MULTILINGUAL_ALIGNMENT_THRESHOLD = 0.9
SAMPLE_CAPTION = "a man riding a horse on the beach"


@pytest.fixture(scope="module")
def settings():
    return Settings(_env_file=None, batch_size=4)


def test_clip_b32_encodes_text_and_images_to_unit_vectors(settings):
    encoder = build_encoder("clip-b32", settings)
    texts = encoder.encode_texts([SAMPLE_CAPTION, "a plate of pasta"])
    images = encoder.encode_images([Image.new("RGB", (64, 64), (10, 120, 200))])
    assert texts.shape == (2, 512)
    assert images.shape == (1, 512)
    np.testing.assert_allclose(np.linalg.norm(texts, axis=1), [1.0, 1.0], atol=1e-4)


def test_query_longer_than_the_token_limit_is_truncated_not_rejected(settings):
    encoder = build_encoder("clip-b32", settings)
    long_query = " ".join(["a very detailed photograph of a busy city street"] * 40)
    vectors = encoder.encode_texts([long_query])
    assert vectors.shape == (1, 512)
    assert not np.isnan(vectors).any()


def test_raw_space_returns_unnormalized_vectors(settings):
    encoder = build_encoder("clip-b32-raw", settings)
    norms = np.linalg.norm(encoder.encode_texts([SAMPLE_CAPTION]), axis=1)
    assert norms[0] > 1.5


def test_siglip_reports_its_own_dimension(settings):
    encoder = build_encoder("siglip-b16", settings)
    assert encoder.encode_texts([SAMPLE_CAPTION]).shape == (1, 768)


def test_multilingual_shares_the_clip_space(settings):
    """Cửa chặn cho giả định ở spec §5.3."""
    clip = build_encoder("clip-b32", settings)
    mclip = build_encoder("mclip-b32", settings)
    a = clip.encode_texts([SAMPLE_CAPTION])[0]
    b = mclip.encode_texts([SAMPLE_CAPTION])[0]
    cosine = float(np.dot(a, b))
    assert cosine > MULTILINGUAL_ALIGNMENT_THRESHOLD, (
        f"cosine={cosine:.3f} — mclip-b32 KHÔNG chung không gian với clip-b32. "
        "Sửa registry: cho mclip-b32 collection_suffix riêng và builds_index=True."
    )


def test_resnet_produces_2048_dimensional_features(settings):
    encoder = build_encoder("resnet50", settings)
    assert encoder.encode_images([Image.new("RGB", (64, 64))]).shape == (1, 2048)
```

- [ ] **Step 10: Chạy test integration — đây là cửa chặn, phải xanh trước khi build index**

Run: `python -m pytest backend/tests/test_encoders_integration.py -m integration -v`
Expected: PASS, 6 test. Lần đầu sẽ tải khoảng 2.5GB checkpoint và chạy vài phút.
Nếu `test_multilingual_shares_the_clip_space` FAIL: sửa `registry.py` cho `mclip-b32` thành `collection_suffix="mclip_b32"`, `builds_index=True`, cập nhật `test_multilingual_reuses_baseline_image_collection` ở Task 2 thành kỳ vọng mới, rồi mới đi tiếp.

- [ ] **Step 11: Commit**

```bash
git add backend/app/encoders backend/tests/test_encoders.py backend/tests/test_encoders_integration.py
git commit -m "feat: encoder CLIP/SigLIP/multilingual/ResNet với cache LRU có lock"
```

---

### Task 7: Baseline BM25 trên caption

**Files:**
- Create: `backend/app/encoders/bm25.py`
- Create: `backend/tests/test_bm25.py`

**Interfaces:**
- Consumes: `app.corpus.Corpus`/`record_payload`, `app.vectordb.Hit`, `rank_bm25.BM25Okapi`.
- Produces: `tokenize(text: str) -> list[str]`; `Bm25Retriever` với `.build(corpus: Corpus) -> Bm25Retriever` (classmethod), `.save(path: Path) -> None`, `.load(path: Path, corpus: Corpus) -> Bm25Retriever` (classmethod), `.search(query: str, k: int, allowed_ids: set[int] | None = None) -> list[Hit]`.

- [ ] **Step 1: Viết test thất bại**

`backend/tests/test_bm25.py`:

```python
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
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.encoders.bm25'`

- [ ] **Step 3: Viết `backend/app/encoders/bm25.py`**

```python
"""Baseline không ngữ nghĩa: BM25 khớp từ khoá trên caption.

Space này không nằm trong Qdrant. Nó tồn tại để trả lời một câu hỏi duy nhất
trong báo cáo: tìm kiếm ngữ nghĩa hơn khớp từ khoá bao nhiêu?
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.corpus import Corpus, record_payload
from app.vectordb import Hit

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tách token đơn giản: hạ chữ thường, giữ chữ và số, bỏ dấu câu.

    Cố tình đơn giản. Một baseline phải dễ giải thích; nếu nó thắng CLIP ở đâu
    thì ta muốn biết chắc đó không phải nhờ một bước tiền xử lý tinh vi.
    """
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Bm25Retriever:
    """Index BM25 trên từng caption, điểm mỗi ảnh là điểm cao nhất trong caption của nó."""

    bm25: BM25Okapi
    caption_image_ids: np.ndarray
    unique_image_ids: np.ndarray
    corpus: Corpus
    _rows: np.ndarray = field(init=False, repr=False)
    _row_of: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._row_of = {int(i): row for row, i in enumerate(self.unique_image_ids)}
        self._rows = np.array(
            [self._row_of[int(i)] for i in self.caption_image_ids], dtype=np.int64
        )

    @classmethod
    def build(cls, corpus: Corpus) -> "Bm25Retriever":
        """Dựng index từ mọi caption trong corpus."""
        pairs = corpus.caption_pairs()
        documents = [tokenize(text) for _, _, text in pairs]
        return cls(
            bm25=BM25Okapi(documents),
            caption_image_ids=np.array([img for img, _, _ in pairs], dtype=np.int64),
            unique_image_ids=np.array(corpus.image_ids(), dtype=np.int64),
            corpus=corpus,
        )

    def save(self, path: Path) -> None:
        """Pickle phần index. Corpus không được pickle — nó nạp lại từ corpus.jsonl."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "caption_image_ids": self.caption_image_ids,
                    "unique_image_ids": self.unique_image_ids,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path, corpus: Corpus) -> "Bm25Retriever":
        """Nạp index đã pickle và ghép lại với corpus.

        :raises FileNotFoundError: nếu chưa build. Người gọi nên dịch lỗi này
            thành thông báo "chạy python tasks.py build --space bm25-cap".
        """
        if not path.exists():
            raise FileNotFoundError(f"Chưa có index BM25 tại {path}")
        with path.open("rb") as fh:
            state = pickle.load(fh)
        return cls(corpus=corpus, **state)

    def search(self, query: str, k: int, allowed_ids: set[int] | None = None) -> list[Hit]:
        """Top-k ảnh theo điểm BM25.

        Điểm của một ảnh là điểm cao nhất trong các caption của nó (không phải
        trung bình): một caption nói đúng thứ người dùng tìm là đủ, và lấy
        trung bình sẽ phạt ảnh có nhiều caption nói về khía cạnh khác.

        Ảnh có điểm 0 (không khớp token nào) bị loại thay vì xếp cuối, để kết
        quả không lẫn những ảnh hoàn toàn không liên quan.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        caption_scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        per_image = np.zeros(len(self.unique_image_ids), dtype=np.float64)
        np.maximum.at(per_image, self._rows, caption_scores)

        if allowed_ids is not None:
            mask = np.array(
                [int(i) in allowed_ids for i in self.unique_image_ids], dtype=bool
            )
            per_image = np.where(mask, per_image, 0.0)

        order = np.argsort(-per_image, kind="stable")[:k]
        hits: list[Hit] = []
        for row in order:
            score = float(per_image[row])
            if score <= 0.0:
                break
            image_id = int(self.unique_image_ids[row])
            hits.append(
                Hit(
                    id=image_id,
                    score=score,
                    payload=record_payload(self.corpus.by_image_id[image_id]),
                )
            )
        return hits
```

- [ ] **Step 4: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_bm25.py -v`
Expected: PASS, 11 test.

- [ ] **Step 5: Commit**

```bash
git add backend/app/encoders/bm25.py backend/tests/test_bm25.py
git commit -m "feat: baseline BM25 trên caption, điểm ảnh lấy max theo caption"
```

---

### Task 8: Chuẩn hoá ảnh upload và schema request/response

**Files:**
- Create: `backend/app/imageio.py`, `backend/app/schemas.py`
- Modify: `backend/app/errors.py` (thêm `ValidationRangeError`)
- Create: `backend/tests/test_imageio.py`, `backend/tests/test_schemas.py`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.errors.BadImageError`/`ImageTooLargeError`.
- Produces: `load_upload_image(data: bytes, content_type: str | None, settings: Settings) -> PIL.Image.Image` và `downscale(image, max_pixels: int) -> PIL.Image.Image` (trong `app/imageio.py`); `ValidationRangeError(SearchError)` (trong `app/errors.py`, map sang HTTP 422); và trong `app/schemas.py`: `Filters`, `TextSearchRequest`, `SearchResultItem`, `SearchResponse`, `SpaceInfo`, `HealthResponse`, `ExampleQuery` với đúng các field mô tả ở spec §4.

- [ ] **Step 1: Viết test thất bại cho xử lý ảnh (hai dòng của Review Focus)**

`backend/tests/test_imageio.py`:

```python
import io

import pytest
from PIL import Image

from app.config import Settings
from app.errors import BadImageError, ImageTooLargeError
from app.imageio import downscale, load_upload_image


@pytest.fixture
def settings():
    return Settings(_env_file=None, max_upload_mb=1, max_image_pixels=10_000)


def encode(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_grayscale_upload_becomes_rgb(settings):
    data = encode(Image.new("L", (20, 20), 128))
    assert load_upload_image(data, "image/png", settings).mode == "RGB"


def test_rgba_upload_becomes_rgb(settings):
    data = encode(Image.new("RGBA", (20, 20), (1, 2, 3, 128)))
    assert load_upload_image(data, "image/png", settings).mode == "RGB"


def test_cmyk_upload_becomes_rgb(settings):
    data = encode(Image.new("CMYK", (20, 20)), fmt="JPEG")
    assert load_upload_image(data, "image/jpeg", settings).mode == "RGB"


def test_oversized_image_is_downscaled_below_the_pixel_budget(settings):
    data = encode(Image.new("RGB", (400, 300)))
    out = load_upload_image(data, "image/png", settings)
    assert out.size[0] * out.size[1] <= settings.max_image_pixels
    assert out.size[0] / out.size[1] == pytest.approx(400 / 300, rel=0.05)


def test_small_image_is_left_alone(settings):
    data = encode(Image.new("RGB", (50, 50)))
    assert load_upload_image(data, "image/png", settings).size == (50, 50)


def test_downscale_never_produces_a_zero_dimension():
    out = downscale(Image.new("RGB", (1000, 2)), max_pixels=10)
    assert min(out.size) >= 1


def test_disallowed_content_type_is_rejected(settings):
    with pytest.raises(BadImageError):
        load_upload_image(encode(Image.new("RGB", (10, 10))), "image/gif", settings)


def test_bytes_that_are_not_an_image_are_rejected(settings):
    with pytest.raises(BadImageError):
        load_upload_image(b"this is not an image", "image/png", settings)


def test_upload_above_the_byte_limit_is_rejected(settings):
    with pytest.raises(ImageTooLargeError):
        load_upload_image(b"x" * (settings.max_upload_bytes + 1), "image/png", settings)


def test_missing_content_type_is_allowed_if_bytes_decode(settings):
    data = encode(Image.new("RGB", (10, 10)))
    assert load_upload_image(data, None, settings).size == (10, 10)
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_imageio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.imageio'`

- [ ] **Step 3: Viết `backend/app/imageio.py`**

```python
"""Chuẩn hoá ảnh upload trước khi đưa vào encoder.

Ảnh người dùng gửi lên không giống ảnh trong dataset: có thể là PNG trong
suốt, ảnh xám từ máy scan, ảnh CMYK từ nhà in, hoặc ảnh điện thoại 12MP nằm
ngang vì EXIF. Encoder chỉ nhận RGB ở kích thước hợp lý, nên mọi việc quy về
một dạng chuẩn xảy ra ở đây và chỉ ở đây.
"""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.errors import BadImageError, ImageTooLargeError


def downscale(image: Image.Image, max_pixels: int) -> Image.Image:
    """Thu nhỏ ảnh về dưới ``max_pixels`` pixel, giữ đúng tỉ lệ khung.

    Trả về chính ảnh đầu vào nếu nó đã đủ nhỏ. Cạnh nhỏ nhất luôn còn ít nhất
    1 pixel, nên ảnh cực dẹt (vd 1000×2) không bị co thành rỗng.
    """
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / (width * height)) ** 0.5
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def load_upload_image(
    data: bytes, content_type: str | None, settings: Settings
) -> Image.Image:
    """Biến bytes upload thành ảnh RGB đã xoay đúng và đủ nhỏ để encode.

    :param content_type: MIME type do client khai; ``None`` thì bỏ qua và chỉ
        dựa vào việc bytes có giải mã được hay không.
    :raises ImageTooLargeError: vượt ``MAX_UPLOAD_MB`` (kiểm tra trước khi giải
        mã, để một file rác 500MB không bị nạp vào RAM).
    :raises BadImageError: content type không cho phép, hoặc bytes không phải ảnh.
    """
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"Ảnh {len(data) / 1024 / 1024:.1f}MB vượt giới hạn "
            f"{settings.max_upload_mb}MB"
        )
    if content_type and content_type not in settings.allowed_image_types_set:
        raise BadImageError(
            f"Định dạng '{content_type}' không hỗ trợ. Cho phép: "
            f"{', '.join(sorted(settings.allowed_image_types_set))}"
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise BadImageError(f"Không giải mã được ảnh: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return downscale(image, settings.max_image_pixels)
```

- [ ] **Step 4: Thêm `ValidationRangeError` vào `backend/app/errors.py`**

```python
class ValidationRangeError(SearchError):
    """Tham số hợp kiểu nhưng ngoài khoảng cho phép (vd k > MAX_TOP_K) → HTTP 422."""
```

- [ ] **Step 5: Chạy test ảnh để chắc nó pass**

Run: `python -m pytest backend/tests/test_imageio.py -v`
Expected: PASS, 10 test.

- [ ] **Step 6: Viết test thất bại cho schema**

`backend/tests/test_schemas.py`:

```python
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
```

- [ ] **Step 7: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas'`

- [ ] **Step 8: Viết `backend/app/schemas.py`**

```python
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
    results: list[SearchResultItem]
    latency_ms: float
    encode_ms: float
    search_ms: float
    space: str
    exact: bool
    total_searched: int
    query_echo: str


class SpaceInfo(BaseModel):
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
    status: str
    qdrant: str
    qdrant_mode: str
    spaces_ready: list[str]
    spaces_missing: list[str]


class ExampleQuery(BaseModel):
    label: str
    query: str
    space: str
    language: str
```

- [ ] **Step 9: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_schemas.py backend/tests/test_imageio.py -v`
Expected: PASS, 19 test.

- [ ] **Step 10: Commit**

```bash
git add backend/app/imageio.py backend/app/schemas.py backend/app/errors.py backend/tests/test_imageio.py backend/tests/test_schemas.py
git commit -m "feat: chuẩn hoá ảnh upload (RGB, EXIF, downscale) và schema API"
```

---

### Task 9: SearchService

**Files:**
- Create: `backend/app/search.py`
- Modify: `backend/app/corpus.py` (thêm `CAPTION_ID_STRIDE`, `caption_point_id`, `caption_payload`)
- Modify: `backend/app/imageio.py` (thêm `load_corpus_image`)
- Create: `backend/tests/test_search_service.py`

**Interfaces:**
- Consumes: `Settings`, `Corpus`, `VectorStore`, `build_filter`, `Hit`, `get_encoder`, `Bm25Retriever`, registry, schemas, errors.
- Produces: trong `app/corpus.py`: `CAPTION_ID_STRIDE = 100`, `caption_point_id(image_id: int, caption_index: int) -> int`, `caption_payload(record: CorpusRecord, caption_index: int) -> dict`. Trong `app/imageio.py`: `load_corpus_image(path: Path, settings: Settings) -> PIL.Image.Image`. Trong `app/search.py`: `THUMB_URL_PREFIX = "/thumbs"`, `SearchService(settings, store, corpus, encoder_factory=get_encoder)` với `.search_text(request: TextSearchRequest) -> SearchResponse`, `.search_image(*, space: str, k: int | None, exact: bool, hnsw_ef: int | None, filters: Filters, image=None, image_id: int | None = None) -> SearchResponse`, `.space_infos() -> list[SpaceInfo]`.

- [ ] **Step 1: Thêm hàm layout caption vào `backend/app/corpus.py`**

```python
CAPTION_ID_STRIDE = 100


def caption_point_id(image_id: int, caption_index: int) -> int:
    """Id point Qdrant cho một caption.

    Trộn image_id với chỉ số caption theo stride cố định, nên id trong
    collection caption không bao giờ đụng id trong collection ảnh và từ một id
    caption luôn suy lại được ảnh gốc.

    :raises ValueError: nếu ``caption_index`` >= CAPTION_ID_STRIDE, vì khi đó
        công thức mất tính đơn ánh.
    """
    if not 0 <= caption_index < CAPTION_ID_STRIDE:
        raise ValueError(
            f"caption_index {caption_index} ngoài khoảng [0, {CAPTION_ID_STRIDE})"
        )
    return image_id * CAPTION_ID_STRIDE + caption_index


def caption_payload(record: CorpusRecord, caption_index: int) -> dict:
    """Payload cho một point caption: metadata của ảnh + chính caption đó."""
    return {
        **record_payload(record),
        "caption_index": caption_index,
        "caption": record.captions[caption_index],
    }
```

- [ ] **Step 2: Thêm `load_corpus_image` vào `backend/app/imageio.py`**

```python
def load_corpus_image(path: Path, settings: Settings) -> Image.Image:
    """Nạp một ảnh có sẵn trong corpus, chuẩn hoá giống ảnh upload.

    Dùng cho nút "Tìm ảnh tương tự": người dùng không upload gì, hệ encode lại
    ảnh đã có trên đĩa. Encode lại cho ra đúng vector đã nằm trong index (model
    tất định), nên không cần đường đọc vector ra khỏi Qdrant.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return downscale(image, settings.max_image_pixels)
```

Thêm `from pathlib import Path` vào phần import của file.

- [ ] **Step 3: Viết test thất bại**

`backend/tests/test_search_service.py`:

```python
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
```

- [ ] **Step 4: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_search_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.search'`

- [ ] **Step 5: Viết `backend/app/search.py`**

```python
"""Điều phối tìm kiếm.

Module này không biết CLIP hay SigLIP là gì: nó hỏi registry space nào dùng
backend nào, gọi encoder tương ứng, rồi trả về một hình dạng response duy nhất
cho cả hai chiều truy vấn.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.corpus import Corpus
from app.encoders import get_encoder
from app.encoders.bm25 import Bm25Retriever
from app.errors import (
    BadRequestError,
    IndexNotBuiltError,
    ModeNotSupportedError,
    ValidationRangeError,
)
from app.imageio import load_corpus_image
from app.registry import (
    BACKEND_BM25,
    MODE_IMAGE2IMAGE,
    MODE_IMAGE2TEXT,
    MODE_TEXT2IMAGE,
    SPACES,
    SpaceSpec,
    collection_name,
    get_space,
)
from app.schemas import (
    Filters,
    SearchResponse,
    SearchResultItem,
    SpaceInfo,
    TextSearchRequest,
)
from app.vectordb import Hit, VectorStore, build_filter

THUMB_URL_PREFIX = "/thumbs"


@dataclass
class SearchService:
    """Một instance dùng chung cho cả process API.

    :param encoder_factory: điểm tiêm phụ thuộc — test truyền encoder giả vào
        đây để kiểm tra điều phối mà không tải model thật.
    """

    settings: Settings
    store: VectorStore
    corpus: Corpus
    encoder_factory: Callable[[str], Any] = get_encoder
    _bm25: Bm25Retriever | None = field(default=None, init=False, repr=False)

    def _resolve_k(self, k: int | None) -> int:
        """Áp mặc định và trần cấu hình cho top-k.

        :raises ValidationRangeError: nếu k vượt ``MAX_TOP_K``.
        """
        resolved = self.settings.top_k_default if k is None else k
        if resolved > self.settings.max_top_k:
            raise ValidationRangeError(
                f"k={resolved} vượt giới hạn MAX_TOP_K={self.settings.max_top_k}"
            )
        return resolved

    @staticmethod
    def _require_mode(space: SpaceSpec, mode: str) -> None:
        """:raises ModeNotSupportedError: nếu space không làm được chiều này."""
        if mode not in space.modes:
            raise ModeNotSupportedError(
                f"Space '{space.name}' không hỗ trợ '{mode}'. "
                f"Hỗ trợ: {', '.join(space.modes)}"
            )

    def _allowed_ids(self, filters: Filters) -> set[int] | None:
        """Tập image_id thoả filter, tính từ corpus.

        Chỉ dùng cho backend BM25 — nó không có payload filter như Qdrant, nên
        việc lọc phải làm ở phía ta.
        """
        if not filters.categories and not filters.supercategories:
            return None
        wanted_cats = set(filters.categories)
        wanted_supers = set(filters.supercategories)
        return {
            record.image_id
            for record in self.corpus.records
            if (not wanted_cats or wanted_cats & set(record.categories))
            and (not wanted_supers or wanted_supers & set(record.supercategories))
        }

    def _bm25_retriever(self) -> Bm25Retriever:
        """Nạp lười index BM25.

        :raises IndexNotBuiltError: nếu chưa build, kèm đúng lệnh cần chạy.
        """
        if self._bm25 is None:
            try:
                self._bm25 = Bm25Retriever.load(self.settings.bm25_path, self.corpus)
            except FileNotFoundError as exc:
                raise IndexNotBuiltError(
                    "Chưa build index BM25. Chạy: python tasks.py build --space bm25-cap"
                ) from exc
        return self._bm25

    def _require_index(self, space: SpaceSpec, target: str) -> tuple[str, int]:
        """Trả về ``(tên collection, số point)``.

        :raises IndexNotBuiltError: nếu collection rỗng hoặc chưa tồn tại.
        """
        name = collection_name(space, self.settings, target=target)
        count = self.store.count(name)
        if count == 0:
            raise IndexNotBuiltError(
                f"Space '{space.name}' (target={target}) chưa có index. "
                f"Chạy: python tasks.py build --space {space.name}"
            )
        return name, count

    def _items(self, hits: list[Hit], target: str) -> list[SearchResultItem]:
        """Đổi Hit thành item response. Một hình dạng duy nhất cho mọi chiều."""
        items: list[SearchResultItem] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload
            items.append(
                SearchResultItem(
                    image_id=int(payload["image_id"]),
                    file_name=payload["file_name"],
                    thumb_url=f"{THUMB_URL_PREFIX}/{payload['file_name']}",
                    score=hit.score,
                    captions=payload.get("captions", []),
                    categories=payload.get("categories", []),
                    rank=rank,
                    matched_caption=payload.get("caption") if target == "caption" else None,
                    caption_index=(
                        payload.get("caption_index") if target == "caption" else None
                    ),
                )
            )
        return items

    def search_text(self, request: TextSearchRequest) -> SearchResponse:
        """Tìm bằng câu chữ. Trả top-k ảnh, hoặc top-k caption nếu target='caption'."""
        started = time.perf_counter()
        space = get_space(request.space)
        mode = MODE_IMAGE2TEXT if request.target == "caption" else MODE_TEXT2IMAGE
        self._require_mode(space, mode)
        k = self._resolve_k(request.k)
        text = (
            request.prompt_template.format(request.query)
            if request.prompt_template
            else request.query
        )

        if space.backend == BACKEND_BM25:
            encode_ms = 0.0
            search_started = time.perf_counter()
            hits = self._bm25_retriever().search(
                text, k, allowed_ids=self._allowed_ids(request.filters)
            )
            search_ms = (time.perf_counter() - search_started) * 1000
            total = len(self.corpus)
        else:
            name, total = self._require_index(space, request.target)
            encode_started = time.perf_counter()
            vector = self.encoder_factory(space.name).encode_texts([text])[0]
            encode_ms = (time.perf_counter() - encode_started) * 1000
            search_started = time.perf_counter()
            hits = self.store.search(
                name,
                vector,
                k=k,
                exact=request.exact,
                hnsw_ef=request.hnsw_ef or self.settings.hnsw_ef_default,
                query_filter=build_filter(
                    request.filters.categories, request.filters.supercategories
                ),
            )
            search_ms = (time.perf_counter() - search_started) * 1000

        return SearchResponse(
            results=self._items(hits, request.target),
            latency_ms=(time.perf_counter() - started) * 1000,
            encode_ms=encode_ms,
            search_ms=search_ms,
            space=space.name,
            exact=request.exact,
            total_searched=total,
            query_echo=text,
        )

    def search_image(
        self,
        *,
        space: str,
        k: int | None,
        exact: bool,
        hnsw_ef: int | None,
        filters: Filters,
        image: Any = None,
        image_id: int | None = None,
    ) -> SearchResponse:
        """Tìm bằng ảnh: ảnh upload, hoặc một ảnh đã có trong corpus.

        Khi tìm bằng ``image_id``, ảnh query bị loại khỏi kết quả — nó luôn tự
        khớp với chính mình ở hạng 1, vừa chiếm một suất vô nghĩa trên UI vừa
        làm phồng P@k khi đo.

        :raises BadRequestError: nếu không có đúng một trong ``image``/``image_id``,
            hoặc ``image_id`` không có trong corpus.
        """
        started = time.perf_counter()
        if (image is None) == (image_id is None):
            raise BadRequestError("Cần đúng một trong hai: file ảnh hoặc image_id")

        spec = get_space(space)
        self._require_mode(spec, MODE_IMAGE2IMAGE)
        k = self._resolve_k(k)

        if image_id is not None:
            record = self.corpus.by_image_id.get(image_id)
            if record is None:
                raise BadRequestError(f"image_id {image_id} không có trong corpus")
            image = load_corpus_image(
                self.settings.images_dir / record.file_name, self.settings
            )

        name, total = self._require_index(spec, "image")
        encode_started = time.perf_counter()
        vector = self.encoder_factory(spec.name).encode_images([image])[0]
        encode_ms = (time.perf_counter() - encode_started) * 1000

        fetch = k + 1 if image_id is not None else k
        search_started = time.perf_counter()
        hits = self.store.search(
            name,
            vector,
            k=fetch,
            exact=exact,
            hnsw_ef=hnsw_ef or self.settings.hnsw_ef_default,
            query_filter=build_filter(filters.categories, filters.supercategories),
        )
        search_ms = (time.perf_counter() - search_started) * 1000
        if image_id is not None:
            hits = [h for h in hits if int(h.payload["image_id"]) != image_id][:k]

        return SearchResponse(
            results=self._items(hits, "image"),
            latency_ms=(time.perf_counter() - started) * 1000,
            encode_ms=encode_ms,
            search_ms=search_ms,
            space=spec.name,
            exact=exact,
            total_searched=total,
            query_echo=f"image_id={image_id}" if image_id is not None else "upload",
        )

    def space_infos(self) -> list[SpaceInfo]:
        """Liệt kê mọi space kèm trạng thái index, để frontend dựng dropdown."""
        infos: list[SpaceInfo] = []
        for space in SPACES.values():
            if space.backend == BACKEND_BM25:
                n_points = len(self.corpus) if self.settings.bm25_path.exists() else 0
            else:
                n_points = self.store.count(
                    collection_name(space, self.settings, target="image")
                )
            infos.append(
                SpaceInfo(
                    name=space.name,
                    hf_id=space.hf_id,
                    dim=space.dim,
                    backend=space.backend,
                    modes=list(space.modes),
                    languages=list(space.languages),
                    ready=n_points > 0,
                    n_points=n_points,
                    note=space.note,
                )
            )
        return infos
```

- [ ] **Step 6: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_search_service.py -v`
Expected: PASS, 21 test.

- [ ] **Step 7: Chạy toàn bộ suite nhanh để chắc không phá gì**

Run: `python -m pytest`
Expected: PASS toàn bộ, không có test `integration` nào chạy.

- [ ] **Step 8: Commit**

```bash
git add backend/app/search.py backend/app/corpus.py backend/app/imageio.py backend/tests/test_search_service.py
git commit -m "feat: SearchService điều phối cả text→ảnh, ảnh→ảnh và BM25"
```

---

### Task 10: FastAPI app và CLI gốc

**Files:**
- Create: `backend/app/main.py`, `tasks.py`
- Modify: `backend/tests/conftest.py` (nhận các fixture chung chuyển từ Task 9), `backend/tests/test_search_service.py` (bỏ phần fixture đã chuyển)
- Create: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `SearchService`, `VectorStore`, `load_corpus`, schemas, errors, `Settings`.
- Produces: `ERROR_STATUS: dict[type[SearchError], int]`, `EXAMPLE_QUERIES: tuple[dict, ...]`, `create_app(settings: Settings | None = None, service: SearchService | None = None) -> FastAPI`, `get_app() -> FastAPI` (factory cho uvicorn); `tasks.py` với subcommand `ingest`, `build`, `querysets`, `eval`, `serve`.

- [ ] **Step 1: Chuyển fixture dùng chung sang `backend/tests/conftest.py`**

Cắt nguyên vẹn `DIM`, `TEXT_KEYWORDS`, `SIZE_TO_INDEX`, `FakeEncoder`, và các fixture `encoder`, `settings`, `corpus`, `service` từ `backend/tests/test_search_service.py` sang cuối `backend/tests/conftest.py`, thêm các import mà chúng cần:

```python
import numpy as np
from qdrant_client import QdrantClient

from app.config import Settings
from app.corpus import caption_payload, caption_point_id, load_corpus, record_payload
from app.encoders.bm25 import Bm25Retriever
from app.search import SearchService
from app.vectordb import VectorStore
from cli.ingest import build_corpus, make_thumbnails
```

Trong fixture `service`, thêm một dòng sinh thumbnail (route `/thumbs` cần file thật để test):

```python
    make_thumbnails(corpus, mini_images_dir, settings.thumbs_dir, settings.thumb_size)
```

`test_search_service.py` giữ lại các hàm test và helper `text_request`, bỏ phần fixture, và import những gì còn dùng.

Run: `python -m pytest backend/tests/test_search_service.py -v`
Expected: PASS, 21 test như trước khi chuyển.

- [ ] **Step 2: Viết test contract thất bại**

`backend/tests/test_api.py`:

```python
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
```

- [ ] **Step 3: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 4: Viết `backend/app/main.py`**

```python
"""Tầng HTTP. Mỏng có chủ ý: mọi quyết định nghiệp vụ nằm ở SearchService.

Việc duy nhất tầng này làm ngoài định tuyến là dịch exception miền sang status
code và chặn path traversal khi serve file ảnh.
"""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.corpus import load_corpus
from app.errors import (
    BadImageError,
    BadRequestError,
    ImageTooLargeError,
    IndexNotBuiltError,
    ModeNotSupportedError,
    SearchError,
    UnknownSpaceError,
    ValidationRangeError,
    VectorStoreDownError,
)
from app.imageio import load_upload_image
from app.schemas import (
    ExampleQuery,
    Filters,
    HealthResponse,
    SearchResponse,
    SpaceInfo,
    TextSearchRequest,
)
from app.search import SearchService
from app.vectordb import VectorStore

ERROR_STATUS: dict[type[SearchError], int] = {
    UnknownSpaceError: 404,
    IndexNotBuiltError: 409,
    ModeNotSupportedError: 400,
    BadImageError: 400,
    BadRequestError: 400,
    ImageTooLargeError: 413,
    ValidationRangeError: 422,
    VectorStoreDownError: 503,
}

EXAMPLE_QUERIES: tuple[dict, ...] = (
    {"label": "Chó trên sofa", "query": "a dog lying on a sofa",
     "space": "clip-b32", "language": "en"},
    {"label": "Người cưỡi ngựa trên bãi biển",
     "query": "a man riding a horse on the beach",
     "space": "clip-b32", "language": "en"},
    {"label": "Đường phố ban đêm có đèn neon",
     "query": "a city street at night with neon lights",
     "space": "laion-b32", "language": "en"},
    {"label": "Ba con chó (thử đếm số)", "query": "three dogs",
     "space": "clip-b32", "language": "en"},
    {"label": "Con mèo đang ngủ trên giường",
     "query": "một con mèo đang ngủ trên giường",
     "space": "mclip-b32", "language": "vi"},
    {"label": "Hai người chơi tennis",
     "query": "hai người đang chơi tennis",
     "space": "mclip-b32", "language": "vi"},
    {"label": "Bàn ăn có pizza và rượu",
     "query": "bàn ăn có pizza và một ly rượu vang",
     "space": "mclip-b32", "language": "vi"},
    {"label": "Xe buýt màu đỏ trên phố",
     "query": "một chiếc xe buýt màu đỏ trên đường phố",
     "space": "mclip-b32", "language": "vi"},
)


def _safe_path(directory: Path, file_name: str) -> Path:
    """Giải đường dẫn file và chắc chắn nó nằm trong ``directory``.

    Chặn cả ``../`` và symlink trỏ ra ngoài. Mọi trường hợp không hợp lệ đều
    trả 404 giống như file không tồn tại, để không tiết lộ cấu trúc thư mục.
    """
    root = directory.resolve()
    candidate = (root / file_name).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file")
    return candidate


def create_app(
    settings: Settings | None = None, service: SearchService | None = None
) -> FastAPI:
    """Dựng app.

    :param service: tiêm sẵn service (test dùng); nếu ``None`` thì app tự dựng
        từ corpus.jsonl và Qdrant theo cấu hình.
    """
    settings = settings or get_settings()
    if service is None:
        store = VectorStore(settings)
        corpus = load_corpus(settings.corpus_path)
        service = SearchService(settings, store, corpus)

    app = FastAPI(title="COCO multimodal search")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(SearchError)
    async def handle_domain_error(_: Request, exc: SearchError) -> JSONResponse:
        status = ERROR_STATUS.get(type(exc), 500)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        qdrant = service.store.health()
        if qdrant != "ok":
            return HealthResponse(
                status="degraded", qdrant=qdrant, qdrant_mode=service.store.mode,
                spaces_ready=[], spaces_missing=[],
            )
        infos = service.space_infos()
        return HealthResponse(
            status="ok",
            qdrant=qdrant,
            qdrant_mode=service.store.mode,
            spaces_ready=[i.name for i in infos if i.ready],
            spaces_missing=[i.name for i in infos if not i.ready],
        )

    @app.get("/spaces", response_model=list[SpaceInfo])
    def spaces() -> list[SpaceInfo]:
        return service.space_infos()

    @app.get("/examples", response_model=list[ExampleQuery])
    def examples() -> list[ExampleQuery]:
        return [ExampleQuery(**item) for item in EXAMPLE_QUERIES]

    @app.post("/search/text", response_model=SearchResponse)
    def search_text(request: TextSearchRequest) -> SearchResponse:
        return service.search_text(request)

    @app.post("/search/image", response_model=SearchResponse)
    async def search_image(
        space: str = Form(...),
        k: int | None = Form(default=None),
        exact: bool = Form(default=True),
        hnsw_ef: int | None = Form(default=None),
        filters_json: str | None = Form(default=None),
        image_id: int | None = Form(default=None),
        file: UploadFile | None = File(default=None),
    ) -> SearchResponse:
        """Tìm bằng ảnh. Chỉ nhận multipart; phải có đúng một trong file / image_id."""
        try:
            filters = (
                Filters.model_validate_json(filters_json) if filters_json else Filters()
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=f"filters_json sai: {exc}") from exc

        image = None
        if file is not None:
            image = load_upload_image(await file.read(), file.content_type, settings)
        return service.search_image(
            space=space, k=k, exact=exact, hnsw_ef=hnsw_ef, filters=filters,
            image=image, image_id=image_id,
        )

    @app.get("/thumbs/{file_name:path}")
    def thumb(file_name: str) -> FileResponse:
        return FileResponse(_safe_path(settings.thumbs_dir, file_name),
                            media_type="image/jpeg")

    @app.get("/images/{file_name:path}")
    def image(file_name: str) -> FileResponse:
        return FileResponse(_safe_path(settings.images_dir, file_name),
                            media_type="image/jpeg")

    return app


def get_app() -> FastAPI:
    """Điểm vào cho uvicorn: `uvicorn app.main:get_app --factory`."""
    return create_app()
```

Chú ý: **không** tạo biến `app = create_app()` ở module level. Một instance dựng lúc import sẽ buộc corpus.jsonl phải tồn tại chỉ để import file, làm mọi test và mọi lệnh `--help` vỡ khi chưa ingest. Uvicorn nạp qua factory nên không cần biến đó.

- [ ] **Step 5: Viết `tasks.py`**

```python
"""CLI gốc của project. Thay cho Makefile để chạy được trên Windows.

Mỗi subcommand import module của nó **bên trong** hàm, nên `python tasks.py
serve` không đòi phải có sẵn module eval, và ngược lại.
"""

import argparse
import sys


def cmd_ingest(rest: list[str]) -> int:
    from cli.ingest import main

    return main(rest)


def cmd_build(rest: list[str]) -> int:
    from cli.build_index import main

    return main(rest)


def cmd_querysets(rest: list[str]) -> int:
    from cli.make_querysets import main

    return main(rest)


def cmd_eval(rest: list[str]) -> int:
    from cli.evaluate import main

    return main(rest)


def cmd_serve(rest: list[str]) -> int:
    import uvicorn

    from app.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="tasks.py serve")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(rest)
    uvicorn.run(
        "app.main:get_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=args.reload,
    )
    return 0


COMMANDS = {
    "ingest": cmd_ingest,
    "build": cmd_build,
    "querysets": cmd_querysets,
    "eval": cmd_eval,
    "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(f"Dùng: python tasks.py <{' | '.join(COMMANDS)}> [tham số...]")
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
```

`tasks.py` nằm ở gốc repo nên `backend/` phải có trong đường import. Thêm ngay dưới phần import:

```python
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))
```

- [ ] **Step 6: Chạy test contract để chắc nó pass**

Run: `python -m pytest backend/tests/test_api.py -v`
Expected: PASS, 24 test.

- [ ] **Step 7: Chạy toàn bộ suite nhanh**

Run: `python -m pytest`
Expected: PASS toàn bộ.

- [ ] **Step 8: Commit**

```bash
git add backend/app/main.py tasks.py backend/tests/conftest.py backend/tests/test_api.py backend/tests/test_search_service.py
git commit -m "feat: FastAPI app, map lỗi miền sang status code, CLI gốc tasks.py"
```

---

### Task 11: CLI build_index

**Files:**
- Create: `backend/cli/build_index.py`
- Create: `backend/tests/test_build_index.py`

**Interfaces:**
- Consumes: `Settings`, `Corpus`/`load_corpus`/`record_payload`/`caption_payload`/`caption_point_id`, `build_encoder`, `Bm25Retriever`, `VectorStore`, registry, `l2_normalize`.
- Produces: `CAPTION_SPACE = "clip-b32"`, `DERIVED_FROM: dict[str, str]`, `BUILD_ORDER: tuple[str, ...]`, `PAYLOAD_INDEX_FIELDS: tuple[str, ...]`, `git_commit() -> str`, `image_vectors(space, settings, corpus, encoder_factory) -> np.ndarray`, `build_space(space_name: str, settings, store, corpus, force: bool = False, encoder_factory=build_encoder) -> dict | None` (trả `index_meta`, hoặc `None` nếu bỏ qua), `build_caption_collection(settings, store, corpus, encoder_factory, force: bool = False) -> int`, `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Viết test thất bại**

`backend/tests/test_build_index.py`:

```python
import json

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.vectordb import VectorStore
from cli.build_index import build_caption_collection, build_space, image_vectors
from app.registry import get_space


@pytest.fixture
def store(settings):
    return VectorStore(settings, client=QdrantClient(location=":memory:"))


@pytest.fixture
def factory(encoder):
    return lambda name, _settings=None: encoder


def test_build_space_upserts_every_image(settings, store, corpus, factory):
    meta = build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    assert meta["n_points"] == 3
    assert store.count("coco_clip_b32") == 3


def test_index_meta_records_what_the_report_needs(settings, store, corpus, factory):
    meta = build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    assert set(meta) >= {"space", "hf_id", "dim", "distance", "normalized",
                         "n_points", "encode_seconds", "batch_size", "built_at",
                         "git_commit"}
    saved = json.loads((settings.index_meta_dir / "clip-b32.json").read_text("utf-8"))
    assert saved["space"] == "clip-b32"


def test_second_build_is_skipped(settings, store, corpus, factory):
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    assert build_space("clip-b32", settings, store, corpus, encoder_factory=factory) is None


def test_force_rebuilds(settings, store, corpus, factory):
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    meta = build_space("clip-b32", settings, store, corpus, force=True,
                       encoder_factory=factory)
    assert meta["n_points"] == 3


def test_space_that_reuses_another_collection_is_skipped(settings, store, corpus, factory):
    assert build_space("mclip-b32", settings, store, corpus, encoder_factory=factory) is None
    assert not (settings.index_meta_dir / "mclip-b32.json").exists()


def test_bm25_space_writes_its_pickle(settings, store, corpus, factory):
    meta = build_space("bm25-cap", settings, store, corpus, encoder_factory=factory)
    assert settings.bm25_path.exists()
    assert meta["n_points"] == 3


def test_caption_collection_holds_one_point_per_caption(settings, store, corpus, factory):
    assert build_caption_collection(settings, store, corpus, factory) == 4
    assert store.count("coco_cap_clip_b32") == 4


def test_image_vectors_are_cached_on_disk(settings, corpus, factory):
    first = image_vectors(get_space("clip-b32"), settings, corpus, factory)
    assert (settings.cache_dir / "imgvec_clip-b32.npy").exists()
    np.testing.assert_allclose(
        first, image_vectors(get_space("clip-b32"), settings, corpus, factory)
    )


def test_normalized_space_derives_from_the_raw_cache(settings, corpus, factory):
    raw = image_vectors(get_space("clip-b32-raw"), settings, corpus, factory)
    normalized = image_vectors(get_space("clip-b32"), settings, corpus, factory)
    assert raw.shape == normalized.shape
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1),
                               np.ones(len(normalized)), atol=1e-5)


def test_filterable_payload_survives_the_upsert(settings, store, corpus, factory):
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    from app.vectordb import build_filter

    hits = store.search("coco_clip_b32", np.eye(1, 4, dtype=np.float32)[0], k=3,
                        query_filter=build_filter(["zebra"], None))
    assert [h.id for h in hits] == [2]
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_build_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli.build_index'`

- [ ] **Step 3: Viết `backend/cli/build_index.py`**

```python
"""Encode corpus và nạp vector vào Qdrant, mỗi space một collection.

Idempotent: space đã build thì bỏ qua, trừ khi có --force. Vector encode được
cache ra .npy, nên build lại collection sau khi đổi cấu hình Qdrant không phải
chạy lại model.
"""

import argparse
import json
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from app.config import Settings, get_settings
from app.corpus import (
    Corpus,
    caption_payload,
    caption_point_id,
    load_corpus,
    record_payload,
)
from app.encoders import build_encoder
from app.encoders.bm25 import Bm25Retriever
from app.registry import (
    BACKEND_BM25,
    SPACES,
    SpaceSpec,
    collection_name,
    get_space,
)
from app.vecutil import l2_normalize
from app.vectordb import VectorStore

PAYLOAD_INDEX_FIELDS = ("categories", "supercategories")

#: Space sở hữu collection caption. Đổi giá trị này nếu muốn chiều ảnh→text
#: chạy trên một model khác.
CAPTION_SPACE = "clip-b32"

#: Space bên trái lấy vector từ space bên phải rồi tự normalize lại. Hai bên
#: dùng chung hf_id nên vector giống nhau từng bit — đúng điều kiện cần cho
#: ablation normalize: chỉ một biến thay đổi.
DERIVED_FROM = {"clip-b32": "clip-b32-raw"}

#: Thứ tự build cho `--space all`. clip-b32-raw đứng trước clip-b32 để lần
#: encode duy nhất phục vụ được cả hai.
BUILD_ORDER = (
    "clip-b32-raw",
    "clip-b32",
    "laion-b32",
    "clip-b16",
    "siglip-b16",
    "resnet50",
    "bm25-cap",
)


def git_commit() -> str:
    """Commit hiện tại, để mỗi index_meta truy được về đúng code đã sinh ra nó."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def encode_all_images(
    encoder, corpus: Corpus, images_dir: Path, batch_size: int
) -> np.ndarray:
    """Encode toàn bộ ảnh corpus theo batch, in tiến độ ra stdout."""
    chunks: list[np.ndarray] = []
    total = len(corpus)
    for start in range(0, total, batch_size):
        records = corpus.records[start : start + batch_size]
        images = [_load_image(images_dir / r.file_name) for r in records]
        chunks.append(encoder.encode_images(images))
        print(f"  encode ảnh {min(start + batch_size, total)}/{total}", end="\r")
    print()
    return np.concatenate(chunks, axis=0)


def image_vectors(
    space: SpaceSpec,
    settings: Settings,
    corpus: Corpus,
    encoder_factory: Callable[..., object] = build_encoder,
) -> np.ndarray:
    """Vector ảnh của một space, ưu tiên cache rồi mới tới encode.

    Thứ tự thử: cache của chính space → suy ra từ cache của space nguồn trong
    ``DERIVED_FROM`` → encode thật.
    """
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = settings.cache_dir / f"imgvec_{space.name}.npy"
    if cache.exists():
        return np.load(cache)

    source_name = DERIVED_FROM.get(space.name)
    if source_name:
        source_cache = settings.cache_dir / f"imgvec_{source_name}.npy"
        if source_cache.exists():
            vectors = l2_normalize(np.load(source_cache))
            np.save(cache, vectors)
            return vectors

    encoder = encoder_factory(space.name, settings)
    vectors = encode_all_images(encoder, corpus, settings.images_dir, settings.batch_size)
    np.save(cache, vectors)
    return vectors


def build_caption_collection(
    settings: Settings,
    store: VectorStore,
    corpus: Corpus,
    encoder_factory: Callable[..., object] = build_encoder,
    force: bool = False,
) -> int:
    """Nạp mọi caption vào collection caption. Trả về số point đã nạp (0 nếu bỏ qua)."""
    space = get_space(CAPTION_SPACE)
    name = collection_name(space, settings, target="caption")
    if store.count(name) > 0 and not force:
        return 0

    pairs = corpus.caption_pairs()
    encoder = encoder_factory(space.name, settings)
    vectors = encoder.encode_texts([text for _, _, text in pairs])
    ids = [caption_point_id(image_id, idx) for image_id, idx, _ in pairs]
    payloads = [
        caption_payload(corpus.by_image_id[image_id], idx) for image_id, idx, _ in pairs
    ]
    store.ensure_collection(name, dim=space.dim, distance=space.distance, recreate=force)
    store.upsert(name, ids=ids, vectors=vectors, payloads=payloads,
                 batch_size=settings.batch_size)
    store.create_payload_indexes(name, PAYLOAD_INDEX_FIELDS)
    return len(ids)


def build_space(
    space_name: str,
    settings: Settings,
    store: VectorStore,
    corpus: Corpus,
    force: bool = False,
    encoder_factory: Callable[..., object] = build_encoder,
) -> dict | None:
    """Build index cho một space.

    :return: dict ``index_meta`` đã ghi ra đĩa, hoặc ``None`` nếu bỏ qua (space
        dùng lại collection của space khác, hoặc index đã có mà không ``force``).
    """
    space = get_space(space_name)
    if not space.builds_index:
        print(f"{space_name}: bỏ qua — dùng lại collection của space khác ({space.note})")
        return None

    started = perf_counter()
    if space.backend == BACKEND_BM25:
        if settings.bm25_path.exists() and not force:
            print(f"{space_name}: đã có index, bỏ qua")
            return None
        Bm25Retriever.build(corpus).save(settings.bm25_path)
        n_points = len(corpus)
    else:
        name = collection_name(space, settings, target="image")
        if store.count(name) > 0 and not force:
            print(f"{space_name}: đã có {store.count(name)} point, bỏ qua")
            return None
        vectors = image_vectors(space, settings, corpus, encoder_factory)
        store.ensure_collection(name, dim=space.dim, distance=space.distance,
                                recreate=force)
        store.upsert(name, ids=corpus.image_ids(), vectors=vectors,
                     payloads=[record_payload(r) for r in corpus.records],
                     batch_size=settings.batch_size)
        store.create_payload_indexes(name, PAYLOAD_INDEX_FIELDS)
        n_points = len(corpus)

    meta = {
        "space": space.name,
        "hf_id": space.hf_id,
        "dim": space.dim,
        "distance": space.distance,
        "normalized": space.normalized,
        "backend": space.backend,
        "n_points": n_points,
        "encode_seconds": round(perf_counter() - started, 2),
        "batch_size": settings.batch_size,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    settings.index_meta_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_meta_dir / f"{space.name}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{space_name}: {n_points} point trong {meta['encode_seconds']}s")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build index vector cho từng space")
    parser.add_argument("--space", default="all",
                        help=f"tên space hoặc 'all'. Hợp lệ: {', '.join(SPACES)}")
    parser.add_argument("--force", action="store_true", help="build lại dù đã có index")
    parser.add_argument("--skip-captions", action="store_true",
                        help="không build collection caption")
    args = parser.parse_args(argv)

    settings = get_settings()
    corpus = load_corpus(settings.corpus_path)
    store = VectorStore(settings)
    if store.health() != "ok":
        print("Không kết nối được Qdrant. Chạy: docker compose up -d qdrant")
        return 1

    names = BUILD_ORDER if args.space == "all" else (args.space,)
    for name in names:
        build_space(name, settings, store, corpus, force=args.force)

    if not args.skip_captions and args.space in ("all", CAPTION_SPACE):
        added = build_caption_collection(settings, store, corpus, force=args.force)
        print(f"collection caption: {added} point mới")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_build_index.py -v`
Expected: PASS, 10 test.

- [ ] **Step 5: Commit**

```bash
git add backend/cli/build_index.py backend/tests/test_build_index.py
git commit -m "feat: CLI build_index idempotent, cache vector và ghi index_meta"
```

---

### Task 12: Chạy pipeline thật trên COCO val2017

Task này không viết code — nó biến hệ từ "test xanh trên 3 ảnh giả" thành "có index thật để đo". Mọi task sau đều cần dữ liệu thật, nên nó phải xong trước.

**Files:**
- Create: `results/index_meta/*.json` (bản sao để commit, vì `data/` bị gitignore)

**Interfaces:**
- Consumes: mọi thứ từ Task 1-11.
- Produces: `data/corpus.jsonl`, `data/thumbs/*.jpg`, 7 collection Qdrant đã nạp, `data/index_meta/*.json` và bản sao trong `results/index_meta/`.

- [ ] **Step 1: Bật Qdrant**

```bash
docker compose up -d qdrant
curl http://localhost:6333/readyz
```
Expected: `all shards are ready`. Nếu Docker Desktop chưa chạy thì bật nó trước — không dùng chế độ embedded ở đây, vì trục ANN vs exact sau này cần server thật.

- [ ] **Step 2: Chạy cửa chặn giả định multilingual trước khi tốn 35 phút encode**

Run: `python -m pytest backend/tests/test_encoders_integration.py -m integration -v`
Expected: PASS 6 test. Lần đầu tải khoảng 2.5GB checkpoint.
Nếu `test_multilingual_shares_the_clip_space` FAIL, dừng lại và sửa registry theo hướng dẫn ở Task 6 Step 10 trước khi đi tiếp.

- [ ] **Step 3: Ingest COCO val2017**

Run: `python tasks.py ingest`
Expected: in ra ba dòng, đại ý `corpus: 5000 ảnh, 25014 caption (mỗi ảnh 5-7)`, số ảnh không có category, và số thumbnail. Tải khoảng 1GB.
Ghi lại ba con số này — báo cáo dùng chính chúng, không dùng số phỏng đoán.

- [ ] **Step 4: Kiểm tra dữ liệu ingest**

```bash
python -c "from pathlib import Path; print('dòng corpus:', sum(1 for _ in Path('data/corpus.jsonl').open(encoding='utf-8')))"
python -c "from pathlib import Path; print('thumbnail:', len(list(Path('data/thumbs').glob('*.jpg'))))"
```
Expected: hai số bằng nhau và bằng số ảnh ở Step 3.

- [ ] **Step 5: Build toàn bộ index**

Run: `python tasks.py build --space all`
Expected: mỗi space in một dòng số point và thời gian; tổng khoảng 30-45 phút trên CPU. `mclip-b32` in dòng "bỏ qua". Cuối cùng in số point của collection caption (bằng số caption ở Step 3).

- [ ] **Step 6: Kiểm tra từng collection đã có đúng số point**

```bash
python -c "
import sys; sys.path.insert(0, 'backend')
from app.config import get_settings
from app.registry import SPACES, BACKEND_BM25, collection_name
from app.vectordb import VectorStore
s = get_settings(); store = VectorStore(s)
for space in SPACES.values():
    if space.backend == BACKEND_BM25 or not space.builds_index:
        continue
    name = collection_name(space, s)
    print(f'{space.name:14s} {name:24s} {store.count(name)}')
print('caption', store.count(collection_name(SPACES[\"clip-b32\"], s, target=\"caption\")))
"
```
Expected: 6 collection ảnh mỗi cái đúng số ảnh corpus, collection caption đúng số caption.

- [ ] **Step 7: Xem hệ thật trả về gì cho một query thật**

```bash
python tasks.py serve
```
Ở terminal khác:
```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/search/text -H "Content-Type: application/json" -d "{\"query\":\"a man riding a horse on the beach\",\"space\":\"clip-b32\",\"k\":5}"
```
Expected: `/health` báo `spaces_ready` gồm 7 space; query trả 5 kết quả mà caption thật sự nói về người và ngựa. Nếu kết quả vô nghĩa, dừng lại và tìm nguyên nhân — đừng đi tiếp để đo số liệu của một index sai.

- [ ] **Step 8: Sao index_meta vào results/ và commit**

```bash
mkdir -p results/index_meta
cp data/index_meta/*.json results/index_meta/
git add results/index_meta
git commit -m "chore: index_meta của lần build trên COCO val2017 đầy đủ"
```

---

### Task 13: Sinh ba query set và dịch bộ tiếng Việt

**Files:**
- Create: `backend/cli/make_querysets.py`
- Create: `backend/tests/test_querysets.py`
- Create: `data/queryset_vi.json`, `data/queryset_i2i.json`, `data/queryset_short.json` (commit vào repo)

**Interfaces:**
- Consumes: `Settings`, `Corpus`/`load_corpus`.
- Produces: `sample_vi_queries(corpus, size: int, seed: int) -> list[dict]` (mỗi dict `{image_id, caption_index, en, vi}`), `sample_i2i_queries(corpus, size: int, seed: int) -> list[int]`, `build_short_queries(corpus) -> list[dict]` (mỗi dict `{query, gold_category, n_gold}`), `main(argv) -> int`.

**Lệch khỏi spec, có chủ ý:** spec §6.2 ghi `queryset_short.json` gồm 200 query ngắn. Kế hoạch này sinh **một query cho mỗi COCO category có trong corpus (80 query)**, không phải 200. Lý do: để nhồi lên 200 thì phải ghép hai category ("a dog and a frisbee"), mà độ liên quan của query ghép cần ngữ nghĩa "chứa cả hai" — một định nghĩa metric khác với phần còn lại của báo cáo. Thêm cách phát biểu khác cho cùng một category thì lại lẫn vào chính trục prompt template đang muốn đo. 80 query một-category cho ground-truth sạch và đúng thứ trục 3 cần.

- [ ] **Step 1: Viết test thất bại**

`backend/tests/test_querysets.py`:

```python
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
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_querysets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli.make_querysets'`

- [ ] **Step 3: Viết `backend/cli/make_querysets.py`**

```python
"""Sinh các query set dùng cho eval. Mọi lấy mẫu đều có seed cố định.

Ba file sinh ra được commit vào repo, vì không reproduce được query set thì
không reproduce được số liệu.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from app.config import Settings, get_settings
from app.corpus import Corpus, load_corpus

SHORT_QUERY_TEMPLATE = "a {}"


def sample_vi_queries(corpus: Corpus, size: int, seed: int) -> list[dict]:
    """Lấy mẫu caption để dịch tay sang tiếng Việt.

    Trả về entry có ``vi`` rỗng — bước dịch sẽ điền vào. Lấy mẫu trên tập
    ``(image_id, caption_index)`` nên không có caption nào bị trùng.
    """
    pairs = corpus.caption_pairs()
    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(size, len(pairs)))
    return [
        {"image_id": image_id, "caption_index": index, "en": text, "vi": ""}
        for image_id, index, text in chosen
    ]


def sample_i2i_queries(corpus: Corpus, size: int, seed: int) -> list[int]:
    """Lấy mẫu image_id làm query cho chiều ảnh→ảnh."""
    rng = random.Random(seed)
    return sorted(rng.sample(corpus.image_ids(), min(size, len(corpus))))


def build_short_queries(corpus: Corpus) -> list[dict]:
    """Một query ngắn cho mỗi category thực sự xuất hiện trong corpus.

    ``n_gold`` là số ảnh chứa category đó — báo cáo cần con số này để người đọc
    biết mẫu số của P@k cho từng query.
    """
    counts = Counter(
        category for record in corpus.records for category in record.categories
    )
    return [
        {
            "query": SHORT_QUERY_TEMPLATE.format(category),
            "gold_category": category,
            "n_gold": count,
        }
        for category, count in sorted(counts.items())
        if count > 0
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh query set cho eval")
    parser.add_argument("--force", action="store_true",
                        help="ghi đè file đã có (mất bản dịch tiếng Việt!)")
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    corpus = load_corpus(settings.corpus_path)

    targets = {
        "queryset_vi.json": sample_vi_queries(
            corpus, settings.queryset_size, settings.random_seed
        ),
        "queryset_i2i.json": sample_i2i_queries(
            corpus, settings.queryset_size, settings.random_seed
        ),
        "queryset_short.json": build_short_queries(corpus),
    }
    for filename, payload in targets.items():
        path = settings.data_dir / filename
        if path.exists() and not args.force:
            print(f"{filename}: đã có, bỏ qua (dùng --force để ghi đè)")
            continue
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{filename}: {len(payload)} entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Chạy test để chắc nó pass (test dịch bị skip vì file chưa có)**

Run: `python -m pytest backend/tests/test_querysets.py -v`
Expected: PASS 9 test, SKIP 1 test.

- [ ] **Step 5: Sinh ba query set từ corpus thật**

Run: `python tasks.py querysets`
Expected: `queryset_vi.json: 200 entry`, `queryset_i2i.json: 200 entry`, `queryset_short.json: 80 entry`.

- [ ] **Step 6: Dịch toàn bộ 200 caption sang tiếng Việt**

Mở `data/queryset_vi.json` và điền field `vi` cho **cả 200** entry. Quy tắc dịch, để bộ query đo đúng thứ cần đo:

- Dịch nghĩa, không dịch từng từ. "A man riding a horse on the beach" → "Một người đàn ông cưỡi ngựa trên bãi biển".
- Giữ nguyên số lượng và quan hệ không gian, vì đó chính là những nhóm lỗi mà §7.3 của spec sẽ phân tích. "Two dogs" phải là "hai con chó", không phải "những con chó".
- Không thêm thông tin không có trong câu gốc và không lược bỏ chi tiết.
- Giữ tên riêng và chữ xuất hiện trong ảnh ở dạng gốc ("biển báo STOP").
- Không để entry nào có `vi` trùng `en` — test sẽ chặn.

- [ ] **Step 7: Kiểm tra bản dịch đã đủ**

Run: `python -m pytest backend/tests/test_querysets.py -v`
Expected: PASS 10 test, không còn SKIP.

- [ ] **Step 8: Commit**

```bash
git add -f data/queryset_vi.json data/queryset_i2i.json data/queryset_short.json
git add backend/cli/make_querysets.py backend/tests/test_querysets.py
git commit -m "feat: sinh ba query set có seed cố định và bộ 200 query tiếng Việt dịch tay"
```

---

### Task 14: CLI evaluate — năm trục ablation

**Files:**
- Create: `backend/cli/evaluate.py`
- Create: `backend/tests/test_evaluate.py`, `backend/tests/test_smoke_pipeline.py`

**Interfaces:**
- Consumes: `Settings`, `Corpus`, `VectorStore`, `build_encoder`, `Bm25Retriever`, `metrics.*`, registry, `build_index.image_vectors` (không dùng — eval chỉ đọc), `l2_normalize`.
- Produces: `T2I_SAMPLE_DEFAULT = 5000`, `PROMPT_TEMPLATES: tuple[str | None, ...]`, `ANN_EF_VALUES: tuple[int, ...]`, `EvalContext(settings, store, corpus, encoder_factory=build_encoder)`, `text_vectors(ctx, space_name, texts, cache_key) -> np.ndarray`, `search_many(ctx, collection, vectors, k, exact=True, hnsw_ef=None) -> tuple[list[list[int]], list[float]]`, `eval_text2image(ctx, space_name, queries: list[tuple[str, int]], ks=(1, 5, 10), exact=True, hnsw_ef=None, cache_key="captions", prompt_template=None) -> dict`, `eval_image2image(ctx, space_name, image_ids: list[int], k=10) -> dict`, `eval_short_queries(ctx, space_name, k=10, prompt_template=None, cache_key="short_raw") -> dict`, `eval_ann_sweep(ctx, space_name, queries, efs=ANN_EF_VALUES, k=10) -> list[dict]`, `eval_language(ctx, space_names: list[str], k=(1, 5, 10)) -> list[dict]`, `write_table(rows: list[dict], name: str, settings, title: str) -> None`, `main(argv) -> int`.

- [ ] **Step 1: Viết test thất bại**

`backend/tests/test_evaluate.py`:

```python
import json

import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.vectordb import VectorStore
from cli.build_index import build_caption_collection, build_space
from cli.evaluate import (
    EvalContext,
    eval_ann_sweep,
    eval_image2image,
    eval_language,
    eval_short_queries,
    eval_text2image,
    write_table,
)


@pytest.fixture
def factory(encoder):
    return lambda name, _settings=None: encoder


@pytest.fixture
def ctx(settings, corpus, factory):
    store = VectorStore(settings, client=QdrantClient(location=":memory:"))
    build_space("clip-b32", settings, store, corpus, encoder_factory=factory)
    build_space("clip-b32-raw", settings, store, corpus, encoder_factory=factory)
    build_space("bm25-cap", settings, store, corpus, encoder_factory=factory)
    build_caption_collection(settings, store, corpus, factory)
    return EvalContext(settings, store, corpus, encoder_factory=factory)


@pytest.fixture
def caption_queries(corpus):
    return [(text, image_id) for image_id, _, text in corpus.caption_pairs()]


def test_text2image_finds_the_source_image_of_every_caption(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1, 3))
    assert row["R@1"] == pytest.approx(1.0)
    assert row["n_queries"] == 4
    assert row["space"] == "clip-b32"


def test_text2image_reports_mrr_and_latency(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1,))
    assert 0.0 <= row["MRR@10"] <= 1.0
    assert row["search_ms_p50"] >= 0
    assert row["search_ms_p95"] >= row["search_ms_p50"]


def test_text2image_is_deterministic(ctx, caption_queries):
    first = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1, 3))
    second = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1, 3))
    assert first["R@1"] == second["R@1"]
    assert first["MRR@10"] == second["MRR@10"]


def test_bm25_baseline_is_evaluated_through_the_same_function(ctx, caption_queries):
    row = eval_text2image(ctx, "bm25-cap", caption_queries, ks=(1,))
    assert row["space"] == "bm25-cap"
    assert 0.0 <= row["R@1"] <= 1.0


def test_unnormalized_space_is_evaluated_on_its_own_collection(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32-raw", caption_queries, ks=(1,))
    assert row["space"] == "clip-b32-raw"


def test_prompt_template_is_recorded_in_the_row(ctx, caption_queries):
    row = eval_text2image(ctx, "clip-b32", caption_queries, ks=(1,),
                          prompt_template="a photo of {}", cache_key="tmpl")
    assert row["prompt_template"] == "a photo of {}"


def test_image2image_uses_the_category_proxy(ctx):
    row = eval_image2image(ctx, "clip-b32", image_ids=[1, 2, 3], k=2)
    assert row["n_queries"] == 3
    assert 0.0 <= row["P@2"] <= 1.0
    assert 0.0 <= row["mAP@2"] <= 1.0


def test_image2image_excludes_the_query_image(ctx):
    row = eval_image2image(ctx, "clip-b32", image_ids=[1], k=2)
    assert row["self_hits"] == 0


def test_short_queries_are_scored_by_category_hit(ctx, settings):
    (settings.data_dir / "queryset_short.json").write_text(
        json.dumps([{"query": "a dog", "gold_category": "dog", "n_gold": 1}]),
        encoding="utf-8",
    )
    row = eval_short_queries(ctx, "clip-b32", k=1)
    assert row["n_queries"] == 1
    assert 0.0 <= row["P@1"] <= 1.0


def test_ann_sweep_refuses_to_run_in_embedded_mode(ctx, caption_queries):
    with pytest.raises(RuntimeError):
        eval_ann_sweep(ctx, "clip-b32", caption_queries, efs=(16,), k=2)


def test_ann_sweep_reports_overlap_against_exact(settings, corpus, factory,
                                                 caption_queries):
    server_like = settings.model_copy(update={"qdrant_mode": "server"})
    store = VectorStore(server_like, client=QdrantClient(location=":memory:"))
    build_space("clip-b32", server_like, store, corpus, encoder_factory=factory)
    context = EvalContext(server_like, store, corpus, encoder_factory=factory)
    rows = eval_ann_sweep(context, "clip-b32", caption_queries, efs=(16, 64), k=2)
    assert [r["hnsw_ef"] for r in rows] == [16, 64]
    assert all(0.0 <= r["overlap@2"] <= 1.0 for r in rows)


def test_language_axis_scores_both_spaces_on_both_languages(ctx, settings):
    (settings.data_dir / "queryset_vi.json").write_text(
        json.dumps([
            {"image_id": 1, "caption_index": 0, "en": "a brown dog on a red sofa",
             "vi": "một con chó nâu trên ghế sofa đỏ"},
            {"image_id": 3, "caption_index": 0, "en": "a slice of pepperoni pizza",
             "vi": "một miếng pizza pepperoni"},
        ]),
        encoding="utf-8",
    )
    rows = eval_language(ctx, ["clip-b32"], k=(1,))
    assert {r["language"] for r in rows} == {"en", "vi"}
    assert all(r["n_queries"] == 2 for r in rows)


def test_write_table_emits_both_csv_and_markdown(settings):
    rows = [{"space": "clip-b32", "R@1": 0.3}, {"space": "laion-b32", "R@1": 0.35}]
    write_table(rows, "axis1_model_t2i", settings, title="Trục 1")
    csv_text = (settings.results_dir / "axis1_model_t2i.csv").read_text("utf-8")
    md_text = (settings.results_dir / "axis1_model_t2i.md").read_text("utf-8")
    assert csv_text.splitlines()[0] == "space,R@1"
    assert "| space | R@1 |" in md_text
    assert "Trục 1" in md_text


def test_write_table_with_no_rows_is_not_an_error(settings):
    write_table([], "empty_axis", settings, title="Rỗng")
    assert (settings.results_dir / "empty_axis.md").exists()
```

- [ ] **Step 2: Chạy test để chắc nó fail**

Run: `python -m pytest backend/tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli.evaluate'`

- [ ] **Step 3: Viết `backend/cli/evaluate.py`**

```python
"""Năm trục ablation → results/*.csv và results/*.md.

Eval truy vấn đúng collection Qdrant mà demo đang dùng, nhưng encode query
theo batch và gọi store trực tiếp thay vì đi qua SearchService: 25 nghìn lần
encode lẻ trên CPU sẽ mất hàng giờ mà không thay đổi kết quả, vì encoder và
phép chuẩn hoá là một.
"""

import argparse
import csv
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from app.config import Settings, get_settings
from app.corpus import Corpus, load_corpus
from app.encoders import build_encoder
from app.encoders.bm25 import Bm25Retriever
from app.metrics import map_at_k, mrr_at_k, overlap_at_k, precision_at_k, recall_at_k
from app.registry import BACKEND_BM25, collection_name, get_space
from app.vectordb import VectorStore

T2I_SAMPLE_DEFAULT = 5000
MRR_K = 10
PROMPT_TEMPLATES: tuple[str | None, ...] = (
    None,
    "a photo of {}",
    "a photo of {}, a type of scene",
)
ANN_EF_VALUES: tuple[int, ...] = (16, 64, 128, 256)


@dataclass
class EvalContext:
    settings: Settings
    store: VectorStore
    corpus: Corpus
    encoder_factory: Callable[..., object] = build_encoder
    _bm25: Bm25Retriever | None = field(default=None, init=False, repr=False)

    def bm25(self) -> Bm25Retriever:
        if self._bm25 is None:
            self._bm25 = Bm25Retriever.load(self.settings.bm25_path, self.corpus)
        return self._bm25


def text_vectors(
    ctx: EvalContext, space_name: str, texts: Sequence[str], cache_key: str
) -> np.ndarray:
    """Encode query text theo batch, cache ra .npy để lần chạy sau tức thì.

    ``cache_key`` phải phân biệt được bộ query và template, nếu không hai thí
    nghiệm khác nhau sẽ dùng chung một cache và cho ra số giống nhau một cách
    giả tạo.
    """
    ctx.settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ctx.settings.cache_dir / f"txt_{space_name}_{cache_key}_{len(texts)}.npy"
    if cache.exists():
        return np.load(cache)
    vectors = ctx.encoder_factory(space_name, ctx.settings).encode_texts(list(texts))
    np.save(cache, vectors)
    return vectors


def search_many(
    ctx: EvalContext,
    collection: str,
    vectors: np.ndarray,
    k: int,
    exact: bool = True,
    hnsw_ef: int | None = None,
) -> tuple[list[list[int]], list[float]]:
    """Truy vấn từng vector, trả ``(danh sách ranking image_id, latency mỗi query)``."""
    rankings: list[list[int]] = []
    latencies: list[float] = []
    for vector in vectors:
        started = perf_counter()
        hits = ctx.store.search(collection, vector, k=k, exact=exact, hnsw_ef=hnsw_ef)
        latencies.append((perf_counter() - started) * 1000)
        rankings.append([int(h.payload["image_id"]) for h in hits])
    return rankings, latencies


def _percentiles(latencies: Sequence[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return round(p50, 3), round(p95, 3)


def eval_text2image(
    ctx: EvalContext,
    space_name: str,
    queries: list[tuple[str, int]],
    ks: Sequence[int] = (1, 5, 10),
    exact: bool = True,
    hnsw_ef: int | None = None,
    cache_key: str = "captions",
    prompt_template: str | None = None,
) -> dict:
    """Recall@k và MRR cho chiều text→ảnh.

    :param queries: danh sách ``(câu query, image_id đúng)``.
    :param prompt_template: chuỗi có ``{}`` để bọc query, hoặc None dùng query thô.
    """
    space = get_space(space_name)
    texts = [
        prompt_template.format(text) if prompt_template else text for text, _ in queries
    ]
    gold = [image_id for _, image_id in queries]
    top_k = max(max(ks), MRR_K)

    if space.backend == BACKEND_BM25:
        retriever = ctx.bm25()
        rankings, latencies = [], []
        for text in texts:
            started = perf_counter()
            hits = retriever.search(text, top_k)
            latencies.append((perf_counter() - started) * 1000)
            rankings.append([h.id for h in hits])
    else:
        vectors = text_vectors(ctx, space_name, texts, cache_key)
        rankings, latencies = search_many(
            ctx,
            collection_name(space, ctx.settings),
            vectors,
            k=top_k,
            exact=exact,
            hnsw_ef=hnsw_ef,
        )

    recalls = recall_at_k(rankings, gold, ks)
    p50, p95 = _percentiles(latencies)
    row = {
        "space": space_name,
        "n_queries": len(queries),
        "exact": exact,
        "hnsw_ef": hnsw_ef,
        "prompt_template": prompt_template or "",
    }
    row.update({f"R@{k}": round(recalls[k], 4) for k in ks})
    row["MRR@10"] = round(mrr_at_k(rankings, gold, MRR_K), 4)
    row["search_ms_p50"] = p50
    row["search_ms_p95"] = p95
    return row


def eval_image2image(
    ctx: EvalContext, space_name: str, image_ids: list[int], k: int = 10
) -> dict:
    """P@k và mAP@k cho chiều ảnh→ảnh, dùng proxy trùng category.

    Ảnh query bị loại khỏi kết quả của chính nó; ``self_hits`` báo cáo số lần
    việc loại đó thất bại, để người đọc tin được con số P@k.
    """
    space = get_space(space_name)
    collection = collection_name(space, ctx.settings)
    cache = ctx.settings.cache_dir / f"imgvec_{space_name}.npy"
    if not cache.exists():
        raise FileNotFoundError(
            f"Thiếu {cache}. Chạy: python tasks.py build --space {space_name}"
        )
    all_vectors = np.load(cache)
    row_of = {image_id: row for row, image_id in enumerate(ctx.corpus.image_ids())}
    vectors = np.stack([all_vectors[row_of[i]] for i in image_ids])
    rankings, latencies = search_many(ctx, collection, vectors, k=k + 1)

    query_labels: list[set[str]] = []
    ranked_labels: list[list[set[str]]] = []
    self_hits = 0
    for image_id, ranking in zip(image_ids, rankings):
        self_hits += sum(1 for other in ranking if other == image_id)
        others = [other for other in ranking if other != image_id][:k]
        query_labels.append(set(ctx.corpus.by_image_id[image_id].categories))
        ranked_labels.append(
            [set(ctx.corpus.by_image_id[other].categories) for other in others]
        )

    p50, p95 = _percentiles(latencies)
    return {
        "space": space_name,
        "n_queries": len(image_ids),
        f"P@{k}": round(precision_at_k(ranked_labels, query_labels, k), 4),
        f"mAP@{k}": round(map_at_k(ranked_labels, query_labels, k), 4),
        "self_hits": self_hits,
        "search_ms_p50": p50,
        "search_ms_p95": p95,
    }


def eval_short_queries(
    ctx: EvalContext,
    space_name: str,
    k: int = 10,
    prompt_template: str | None = None,
    cache_key: str = "short_raw",
) -> dict:
    """P@k trên bộ query ngắn kiểu từ khoá; đúng = ảnh chứa category của query.

    :param cache_key: phải phân biệt từng template. Không dùng ``hash()`` của
        template để sinh khoá: hash của chuỗi trong Python được ngẫu nhiên hoá
        theo từng process, nên tên file cache sẽ đổi mỗi lần chạy — cache không
        bao giờ trúng, và tệ hơn là tên file không tất định.
    """
    entries = json.loads(
        (ctx.settings.data_dir / "queryset_short.json").read_text(encoding="utf-8")
    )
    space = get_space(space_name)
    texts = [
        prompt_template.format(e["query"]) if prompt_template else e["query"]
        for e in entries
    ]
    vectors = text_vectors(ctx, space_name, texts, cache_key)
    rankings, latencies = search_many(
        ctx, collection_name(space, ctx.settings), vectors, k=k
    )
    query_labels = [{e["gold_category"]} for e in entries]
    ranked_labels = [
        [set(ctx.corpus.by_image_id[i].categories) for i in ranking]
        for ranking in rankings
    ]
    p50, _ = _percentiles(latencies)
    return {
        "space": space_name,
        "prompt_template": prompt_template or "",
        "n_queries": len(entries),
        f"P@{k}": round(precision_at_k(ranked_labels, query_labels, k), 4),
        "search_ms_p50": p50,
    }


def eval_ann_sweep(
    ctx: EvalContext,
    space_name: str,
    queries: list[tuple[str, int]],
    efs: Sequence[int] = ANN_EF_VALUES,
    k: int = 10,
) -> list[dict]:
    """So HNSW với exact trên cùng một collection.

    :raises RuntimeError: nếu đang ở chế độ embedded — chế độ đó luôn brute
        force và bỏ qua HNSW, nên mọi con số thu được sẽ là exact đội lốt ANN.
    """
    if ctx.settings.qdrant_mode != "server":
        raise RuntimeError(
            "Trục ANN vs exact cần Qdrant server. Đặt QDRANT_MODE=server và chạy "
            "docker compose up -d qdrant"
        )
    space = get_space(space_name)
    collection = collection_name(space, ctx.settings)
    texts = [text for text, _ in queries]
    gold = [image_id for _, image_id in queries]
    vectors = text_vectors(ctx, space_name, texts, "captions")

    exact_rankings, exact_latencies = search_many(ctx, collection, vectors, k=k)
    exact_p50, exact_p95 = _percentiles(exact_latencies)
    rows = [
        {
            "space": space_name,
            "hnsw_ef": "exact",
            f"overlap@{k}": 1.0,
            "R@1": round(recall_at_k(exact_rankings, gold, (1,))[1], 4),
            "search_ms_p50": exact_p50,
            "search_ms_p95": exact_p95,
        }
    ]
    for ef in efs:
        rankings, latencies = search_many(
            ctx, collection, vectors, k=k, exact=False, hnsw_ef=ef
        )
        p50, p95 = _percentiles(latencies)
        rows.append(
            {
                "space": space_name,
                "hnsw_ef": ef,
                f"overlap@{k}": round(overlap_at_k(rankings, exact_rankings, k), 4),
                "R@1": round(recall_at_k(rankings, gold, (1,))[1], 4),
                "search_ms_p50": p50,
                "search_ms_p95": p95,
            }
        )
    return rows


def eval_language(
    ctx: EvalContext, space_names: list[str], k: Sequence[int] = (1, 5, 10)
) -> list[dict]:
    """Đo cùng 200 nội dung ở hai ngôn ngữ, cho từng space.

    Cùng một tập nội dung ở cả hai ngôn ngữ là điều kiện để con số so được với
    nhau: chênh lệch đọc ra được là chênh lệch do ngôn ngữ, không do bộ query.
    """
    entries = json.loads(
        (ctx.settings.data_dir / "queryset_vi.json").read_text(encoding="utf-8")
    )
    rows: list[dict] = []
    for space_name in space_names:
        for language in ("en", "vi"):
            queries = [(e[language], e["image_id"]) for e in entries]
            row = eval_text2image(
                ctx, space_name, queries, ks=k, cache_key=f"lang_{language}"
            )
            row["language"] = language
            rows.append(row)
    return rows


def write_table(rows: list[dict], name: str, settings: Settings, title: str) -> None:
    """Ghi một bảng ra cả CSV (để tính toán) và Markdown (để dán vào báo cáo)."""
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    csv_path = settings.results_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"### {title}", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    else:
        lines.append("_không có dòng nào_")
    lines.append("")
    (settings.results_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"results/{name}.csv + .md — {len(rows)} dòng")
```

- [ ] **Step 4: Viết hàm `main` ở cuối `backend/cli/evaluate.py`**

```python
AXES = ("model-t2i", "model-i2i", "normalize", "prompt", "ann", "language")

T2I_SPACES = ("clip-b32", "clip-b16", "laion-b32", "siglip-b16", "bm25-cap")
I2I_SPACES = ("clip-b32", "clip-b16", "laion-b32", "siglip-b16", "resnet50")


def sample_caption_queries(corpus: Corpus, size: int, seed: int) -> list[tuple[str, int]]:
    """Lấy mẫu caption làm query text→ảnh, cùng một mẫu cho mọi space.

    Dùng chung một mẫu là điều làm các dòng trong bảng so được với nhau. Corpus
    tra cứu vẫn là toàn bộ 5.000 ảnh, chỉ số lượng query bị giới hạn.
    """
    import random

    pairs = [(text, image_id) for image_id, _, text in corpus.caption_pairs()]
    if size >= len(pairs):
        return pairs
    return random.Random(seed).sample(pairs, size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy các trục ablation")
    parser.add_argument("--axis", default="all",
                        help=f"trục cần chạy hoặc 'all'. Hợp lệ: {', '.join(AXES)}")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="bằng với --axis all; có để đúng lệnh ghi trong spec §11")
    parser.add_argument("--sample", type=int, default=T2I_SAMPLE_DEFAULT,
                        help="số query text→ảnh (0 = dùng toàn bộ caption)")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    settings = get_settings()
    corpus = load_corpus(settings.corpus_path)
    store = VectorStore(settings)
    if store.health() != "ok":
        print("Không kết nối được Qdrant. Chạy: docker compose up -d qdrant")
        return 1
    ctx = EvalContext(settings, store, corpus)

    size = args.sample or len(corpus.caption_pairs())
    queries = sample_caption_queries(corpus, size, settings.random_seed)
    image_ids = json.loads(
        (settings.data_dir / "queryset_i2i.json").read_text(encoding="utf-8")
    )
    wanted = AXES if (args.run_all or args.axis == "all") else (args.axis,)

    if "model-t2i" in wanted:
        write_table([eval_text2image(ctx, s, queries) for s in T2I_SPACES],
                    "axis1_model_t2i", settings,
                    title=f"Trục 1 — text→ảnh ({len(queries)} query, 5.000 ảnh)")
    if "model-i2i" in wanted:
        write_table([eval_image2image(ctx, s, image_ids, k=args.k) for s in I2I_SPACES],
                    "axis1_model_i2i", settings,
                    title=f"Trục 1 — ảnh→ảnh ({len(image_ids)} ảnh query, proxy category)")
    if "normalize" in wanted:
        write_table([eval_text2image(ctx, s, queries) for s in ("clip-b32", "clip-b32-raw")],
                    "axis2_normalize", settings,
                    title="Trục 2 — cosine trên vector normalize vs dot trên vector thô")
    if "prompt" in wanted:
        rows = []
        for index, template in enumerate(PROMPT_TEMPLATES):
            rows.append(eval_text2image(ctx, "clip-b32", queries,
                                        cache_key=f"tmpl_{index}",
                                        prompt_template=template))
            rows.append(eval_short_queries(ctx, "clip-b32", k=args.k,
                                           prompt_template=template,
                                           cache_key=f"short_{index}"))
        write_table(rows, "axis3_prompt", settings,
                    title="Trục 3 — prompt template trên caption dài và trên query ngắn")
    if "ann" in wanted:
        write_table(eval_ann_sweep(ctx, "clip-b32", queries, k=args.k),
                    "axis4_ann", settings,
                    title=f"Trục 4 — HNSW vs exact trên 5.000 vector (k={args.k})")
    if "language" in wanted:
        write_table(eval_language(ctx, ["clip-b32", "mclip-b32"]),
                    "axis5_language", settings,
                    title="Trục 5 — cùng 200 nội dung, tiếng Anh vs tiếng Việt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Lưu ý khi ghép bảng trục 3: `eval_text2image` và `eval_short_queries` trả hai bộ khoá khác nhau, nên `write_table` sẽ lấy cột theo dòng đầu tiên. Sửa bằng cách chuẩn hoá trước khi ghi:

```python
def _align_columns(rows: list[dict]) -> list[dict]:
    """Điền ô rỗng cho các khoá thiếu, để bảng gộp nhiều loại dòng không mất cột."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{key: row.get(key, "") for key in columns} for row in rows]
```

Gọi `_align_columns(rows)` ngay đầu `write_table`.

- [ ] **Step 5: Chạy test để chắc nó pass**

Run: `python -m pytest backend/tests/test_evaluate.py -v`
Expected: PASS, 14 test.

- [ ] **Step 6: Viết test smoke chạy hết chuỗi trên fixture 20 ảnh**

Spec §10 đòi một test đi hết đường ingest → build → search → eval. Các test trước
mỗi cái chỉ kiểm một khâu; test này bắt lỗi ở chỗ ghép nối giữa các khâu.

`backend/tests/test_smoke_pipeline.py`:

```python
"""Một test đi hết chuỗi trên corpus 3 ảnh: ingest → build → search → eval."""

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
```

Run: `python -m pytest backend/tests/test_smoke_pipeline.py -v`
Expected: PASS, 1 test.

- [ ] **Step 7: Chạy toàn bộ suite nhanh**

Run: `python -m pytest`
Expected: PASS toàn bộ.

- [ ] **Step 8: Chạy eval thật trên COCO**

Run: `python tasks.py eval --axis all`
Expected: sinh 6 cặp file trong `results/`. Thời gian ước lượng 20-40 phút (encode query có cache nên lần chạy sau nhanh hơn nhiều).
Đối chiếu tỉnh táo: `clip-b32` R@1 text→ảnh nên nằm khoảng 0.25-0.35. Nếu ra gần 0 thì có lỗi ghép nối, không phải model kém — dừng lại và kiểm tra `image_id` trong payload có khớp `gold` hay không.

- [ ] **Step 9: Commit**

```bash
git add backend/cli/evaluate.py backend/tests/test_evaluate.py backend/tests/test_smoke_pipeline.py results
git commit -m "feat: CLI evaluate 5 trục ablation, test smoke toàn chuỗi, bảng kết quả đầu tiên"
```

---

### Task 15: Scaffold frontend, API client và hook tìm kiếm

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/index.html`, `frontend/.env.example`
- Create: `frontend/src/index.css`, `frontend/src/types.ts`, `frontend/src/api/client.ts`, `frontend/src/hooks/useSearch.ts`
- Create: `frontend/src/api/client.test.ts`

**Interfaces:**
- Consumes: API ở Task 10.
- Produces: kiểu `SpaceInfo`, `SearchResultItem`, `SearchResponse`, `ExampleQuery`, `Filters`, `SearchParams` (trong `src/types.ts`); `API_BASE`, `apiUrl(path: string): string`, `ApiError` (class, có `.status`), `fetchSpaces(): Promise<SpaceInfo[]>`, `fetchExamples(): Promise<ExampleQuery[]>`, `searchText(params: SearchParams): Promise<SearchResponse>`, `searchImage(params: SearchParams & { file?: File; imageId?: number }): Promise<SearchResponse>` (trong `src/api/client.ts`); `useSearch()` trả `{ response, loading, error, runText, runImage, runSimilar }`.

**Định hướng thiết kế** (giữ nguyên xuyên suốt các task frontend, để UI không thành chắp vá): nền trung tính `slate`, một màu nhấn duy nhất `indigo-500` dùng cho trạng thái chọn và nút chính. Chữ `text-slate-100` trên `bg-slate-900`. Thẻ kết quả bo `rounded-lg`, viền `border-slate-700`, ảnh `aspect-square object-cover`. Khoảng cách dùng thang 4px của Tailwind. Không inline style ở bất cứ đâu.

- [ ] **Step 1: Khởi tạo project và cài dependency**

```bash
cd frontend
npm init -y
npm install react@^18.3 react-dom@^18.3
npm install -D vite@^5.4 @vitejs/plugin-react@^4.3 typescript@^5.6 @types/react@^18.3 @types/react-dom@^18.3 tailwindcss@^3.4 postcss@^8.4 autoprefixer@^10.4 vitest@^2.1
```

Chọn Vite 5 và Tailwind 3 có lý do: Node ở máy này là 18.20.8, còn Tailwind 4 đòi Node 20+. Đừng nâng lên bản mới hơn rồi mất buổi tối gỡ lỗi toolchain.

`frontend/package.json` — phần `scripts`:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  }
}
```

- [ ] **Step 2: File cấu hình**

`frontend/vite.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
```

`frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "types": ["vite/client", "vitest/globals"]
  },
  "include": ["src"]
}
```

`frontend/tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

`frontend/postcss.config.js`:

```js
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
```

`frontend/index.html`:

```html
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Tìm kiếm ngữ nghĩa COCO</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`frontend/.env.example`:

```
VITE_API_BASE=http://localhost:8000
```

`frontend/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

`src/main.tsx` cố tình **chưa** tạo ở task này — nó import `./App`, mà `App.tsx` thuộc Task 16. Tạo nó bây giờ sẽ làm `npm run build` đỏ ở cuối task này.

Thêm `"type": "module"` vào `frontend/package.json` để các file cấu hình `export default` chạy được.

- [ ] **Step 3: Viết `frontend/src/types.ts`**

Các kiểu phải khớp đúng `backend/app/schemas.py`; lệch một tên field là lỗi chỉ hiện lúc chạy.

```ts
export interface SpaceInfo {
  name: string;
  hf_id: string | null;
  dim: number | null;
  backend: string;
  modes: string[];
  languages: string[];
  ready: boolean;
  n_points: number;
  note: string;
}

export interface SearchResultItem {
  image_id: number;
  file_name: string;
  thumb_url: string;
  score: number;
  captions: string[];
  categories: string[];
  rank: number;
  matched_caption: string | null;
  caption_index: number | null;
}

export interface SearchResponse {
  results: SearchResultItem[];
  latency_ms: number;
  encode_ms: number;
  search_ms: number;
  space: string;
  exact: boolean;
  total_searched: number;
  query_echo: string;
}

export interface ExampleQuery {
  label: string;
  query: string;
  space: string;
  language: string;
}

export interface Filters {
  categories: string[];
  supercategories: string[];
}

export interface SearchParams {
  query?: string;
  space: string;
  k?: number;
  exact?: boolean;
  hnswEf?: number | null;
  filters?: Filters;
  promptTemplate?: string | null;
}
```

- [ ] **Step 4: Viết test thất bại cho API client**

`frontend/src/api/client.test.ts`:

```ts
import { File } from "node:buffer";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiUrl, searchImage, searchText } from "./client";

// `File` chỉ thành global từ Node 20; máy này chạy Node 18 nên phải import
// tường minh từ node:buffer, nếu không test đổ ReferenceError.

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("apiUrl", () => {
  it("joins the base with the path", () => {
    expect(apiUrl("/spaces")).toMatch(/\/spaces$/);
  });

  it("does not double the slash", () => {
    expect(apiUrl("/spaces")).not.toMatch(/\/\/spaces$/);
  });
});

describe("searchText", () => {
  it("sends the documented json body", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    await searchText({ query: "a dog", space: "clip-b32", k: 5 });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(init?.body))).toMatchObject({
      query: "a dog",
      space: "clip-b32",
      k: 5,
    });
  });

  it("turns an error response into ApiError with its status and detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "chưa build index" }, 409),
    );
    await expect(searchText({ query: "a dog", space: "siglip-b16" })).rejects.toMatchObject({
      status: 409,
      message: "chưa build index",
    });
  });

  it("reports a network failure as ApiError with status 0", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("failed to fetch"));
    const error = await searchText({ query: "a dog", space: "clip-b32" }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
  });
});

describe("searchImage", () => {
  it("posts multipart with the file and omits image_id", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    const file = new File([new Uint8Array([1, 2, 3])], "q.png", { type: "image/png" });
    await searchImage({ space: "clip-b32", file });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("space")).toBe("clip-b32");
    expect(body.get("file")).toBeInstanceOf(File);
    expect(body.get("image_id")).toBeNull();
  });

  it("posts image_id when searching by a corpus image", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    await searchImage({ space: "clip-b32", imageId: 42 });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("image_id")).toBe("42");
    expect(body.get("file")).toBeNull();
  });

  it("serialises filters as json", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ results: [] }));
    await searchImage({
      space: "clip-b32",
      imageId: 1,
      filters: { categories: ["dog"], supercategories: [] },
    });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(JSON.parse(String(body.get("filters_json")))).toEqual({
      categories: ["dog"],
      supercategories: [],
    });
  });
});
```

- [ ] **Step 5: Chạy test để chắc nó fail**

Run: `cd frontend && npm test`
Expected: FAIL — không import được `./client`.

- [ ] **Step 6: Viết `frontend/src/api/client.ts`**

```ts
import type {
  ExampleQuery,
  SearchParams,
  SearchResponse,
  SpaceInfo,
} from "../types";

/** Địa chỉ API, lấy từ biến môi trường Vite; không viết cứng trong component. */
export const API_BASE = (
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
).replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Lỗi API kèm status, để UI phân biệt 409 "chưa build index" với 503 "Qdrant chết". */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readError(response: Response): Promise<ApiError> {
  let detail = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      detail = body.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ");
    }
  } catch {
    // Body không phải JSON: giữ thông báo mặc định theo status.
  }
  return new ApiError(response.status, detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch (cause) {
    throw new ApiError(0, `Không gọi được API tại ${API_BASE}. API đã chạy chưa?`);
  }
  if (!response.ok) {
    throw await readError(response);
  }
  return (await response.json()) as T;
}

export function fetchSpaces(): Promise<SpaceInfo[]> {
  return request<SpaceInfo[]>("/spaces");
}

export function fetchExamples(): Promise<ExampleQuery[]> {
  return request<ExampleQuery[]>("/examples");
}

export function searchText(params: SearchParams): Promise<SearchResponse> {
  return request<SearchResponse>("/search/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: params.query ?? "",
      space: params.space,
      k: params.k,
      exact: params.exact ?? true,
      hnsw_ef: params.hnswEf ?? null,
      filters: params.filters ?? { categories: [], supercategories: [] },
      prompt_template: params.promptTemplate ?? null,
    }),
  });
}

/**
 * Tìm bằng ảnh. Truyền `file` cho ảnh upload, hoặc `imageId` cho ảnh đã có
 * trong corpus — API yêu cầu đúng một trong hai, nên hàm này chỉ gắn field nào
 * thực sự có.
 */
export function searchImage(
  params: SearchParams & { file?: File; imageId?: number },
): Promise<SearchResponse> {
  const form = new FormData();
  form.set("space", params.space);
  form.set("exact", String(params.exact ?? true));
  if (params.k !== undefined) form.set("k", String(params.k));
  if (params.hnswEf != null) form.set("hnsw_ef", String(params.hnswEf));
  if (params.filters) form.set("filters_json", JSON.stringify(params.filters));
  if (params.file) form.set("file", params.file);
  if (params.imageId !== undefined) form.set("image_id", String(params.imageId));
  return request<SearchResponse>("/search/image", { method: "POST", body: form });
}
```

- [ ] **Step 7: Chạy test để chắc nó pass**

Run: `cd frontend && npm test`
Expected: PASS, 9 test.

- [ ] **Step 8: Viết `frontend/src/hooks/useSearch.ts`**

```ts
import { useCallback, useState } from "react";

import { ApiError, searchImage, searchText } from "../api/client";
import type { SearchParams, SearchResponse } from "../types";

interface SearchState {
  response: SearchResponse | null;
  loading: boolean;
  error: string | null;
}

/**
 * Một lần tìm kiếm và toàn bộ trạng thái của nó.
 *
 * Giữ kết quả cũ trong lúc đang tải để grid không nháy trắng giữa hai lần tìm;
 * chỉ xoá khi có lỗi. Trả về response để chỗ gọi (chế độ so sánh) dùng trực tiếp.
 */
export function useSearch() {
  const [state, setState] = useState<SearchState>({
    response: null,
    loading: false,
    error: null,
  });

  const run = useCallback(async (task: () => Promise<SearchResponse>) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const response = await task();
      setState({ response, loading: false, error: null });
      return response;
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Lỗi không xác định";
      setState({ response: null, loading: false, error: message });
      return null;
    }
  }, []);

  const runText = useCallback(
    (params: SearchParams) => run(() => searchText(params)),
    [run],
  );

  const runImage = useCallback(
    (params: SearchParams & { file: File }) => run(() => searchImage(params)),
    [run],
  );

  const runSimilar = useCallback(
    (params: SearchParams & { imageId: number }) => run(() => searchImage(params)),
    [run],
  );

  return { ...state, runText, runImage, runSimilar };
}
```

- [ ] **Step 9: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/tsconfig.json frontend/tailwind.config.js frontend/postcss.config.js frontend/index.html frontend/.env.example frontend/src
git commit -m "feat: scaffold frontend Vite/React/Tailwind, API client và hook tìm kiếm"
```

---

### Task 16: UI tìm kiếm cơ bản

**Files:**
- Create: `frontend/src/main.tsx`, `frontend/src/App.tsx`
- Create: `frontend/src/components/ModelSelect.tsx`, `SearchBar.tsx`, `ExampleChips.tsx`, `ResultCard.tsx`, `ResultGrid.tsx`, `AdvancedPanel.tsx`, `StatusBar.tsx`

**Interfaces:**
- Consumes: `useSearch`, `fetchSpaces`, `fetchExamples`, `API_BASE`, các kiểu ở `src/types.ts`.
- Produces: `App` (default export); component nhận props: `ModelSelect({ spaces, value, onChange, label })`, `SearchBar({ onSearchText, onSearchImage, disabled })`, `ExampleChips({ examples, onPick })`, `ResultCard({ item, onOpen })`, `ResultGrid({ items, loading, onOpen })`, `AdvancedPanel({ params, onChange, categories })`, `StatusBar({ response, error, loading })`.

- [ ] **Step 1: `frontend/src/main.tsx`**

```tsx
import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 2: `frontend/src/components/ModelSelect.tsx`**

Dropdown dựng **từ `/spaces`**, nên thêm model ở backend là tự hiện ở đây. Space chưa build index bị vô hiệu hoá kèm ghi chú, để người dùng không chọn rồi nhận lỗi 409.

```tsx
import type { SpaceInfo } from "../types";

interface Props {
  spaces: SpaceInfo[];
  value: string;
  onChange: (value: string) => void;
  label: string;
}

export default function ModelSelect({ spaces, value, onChange, label }: Props) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-slate-400">{label}</span>
      <select
        className="rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 focus:border-indigo-500 focus:outline-none"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {spaces.map((space) => (
          <option key={space.name} value={space.name} disabled={!space.ready}>
            {space.name}
            {space.languages.includes("vi") ? " · tiếng Việt" : ""}
            {space.ready ? "" : " (chưa build index)"}
          </option>
        ))}
      </select>
    </label>
  );
}
```

- [ ] **Step 3: `frontend/src/components/SearchBar.tsx`**

Một ô text và một vùng kéo-thả, cộng dán ảnh từ clipboard. Dán ảnh là đường nhanh nhất để thử chiều ảnh→ảnh khi demo, nên nó phải có.

```tsx
import { useRef, useState } from "react";

interface Props {
  onSearchText: (query: string) => void;
  onSearchImage: (file: File) => void;
  disabled: boolean;
}

export default function SearchBar({ onSearchText, onSearchImage, disabled }: Props) {
  const [text, setText] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  function pickFirstImage(items: FileList | null) {
    const file = items?.[0];
    if (file && file.type.startsWith("image/")) {
      onSearchImage(file);
    }
  }

  return (
    <div
      className={`flex flex-col gap-3 rounded-lg border-2 border-dashed p-4 transition-colors ${
        dragging ? "border-indigo-500 bg-slate-800" : "border-slate-700"
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        pickFirstImage(event.dataTransfer.files);
      }}
      onPaste={(event) => pickFirstImage(event.clipboardData.files)}
    >
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (text.trim()) onSearchText(text.trim());
        }}
      >
        <input
          className="flex-1 rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
          placeholder="Mô tả ảnh bạn muốn tìm, ví dụ: a man riding a horse on the beach"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <button
          className="rounded-md bg-indigo-500 px-4 py-2 font-medium text-white hover:bg-indigo-400 disabled:opacity-50"
          type="submit"
          disabled={disabled || !text.trim()}
        >
          Tìm
        </button>
      </form>

      <div className="flex items-center gap-3 text-sm text-slate-400">
        <button
          className="rounded-md border border-slate-700 px-3 py-1 hover:border-indigo-500"
          type="button"
          onClick={() => fileInput.current?.click()}
        >
          Chọn ảnh
        </button>
        <span>hoặc kéo-thả / dán ảnh vào khung này để tìm bằng ảnh</span>
        <input
          ref={fileInput}
          className="hidden"
          type="file"
          accept="image/*"
          onChange={(event) => pickFirstImage(event.target.files)}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: `frontend/src/components/ExampleChips.tsx`**

```tsx
import type { ExampleQuery } from "../types";

interface Props {
  examples: ExampleQuery[];
  onPick: (example: ExampleQuery) => void;
}

export default function ExampleChips({ examples, onPick }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {examples.map((example) => (
        <button
          key={example.label}
          className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:border-indigo-500 hover:text-slate-100"
          type="button"
          onClick={() => onPick(example)}
        >
          {example.language === "vi" ? "🇻🇳 " : ""}
          {example.label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: `frontend/src/components/ResultCard.tsx` và `ResultGrid.tsx`**

`ResultCard.tsx`:

```tsx
import { API_BASE } from "../api/client";
import type { SearchResultItem } from "../types";

interface Props {
  item: SearchResultItem;
  onOpen: (item: SearchResultItem) => void;
}

export default function ResultCard({ item, onOpen }: Props) {
  return (
    <button
      className="group relative overflow-hidden rounded-lg border border-slate-700 text-left hover:border-indigo-500"
      type="button"
      onClick={() => onOpen(item)}
    >
      <img
        className="aspect-square w-full object-cover"
        src={`${API_BASE}${item.thumb_url}`}
        alt={item.captions[0] ?? `ảnh ${item.image_id}`}
        loading="lazy"
      />
      <span className="absolute left-1 top-1 rounded bg-slate-900/80 px-1.5 py-0.5 text-xs text-slate-200">
        #{item.rank} · {item.score.toFixed(3)}
      </span>
      <span className="absolute inset-x-0 bottom-0 line-clamp-2 bg-slate-900/85 p-2 text-xs text-slate-200 opacity-0 transition-opacity group-hover:opacity-100">
        {item.matched_caption ?? item.captions[0] ?? ""}
      </span>
    </button>
  );
}
```

`ResultGrid.tsx`:

```tsx
import type { SearchResultItem } from "../types";
import ResultCard from "./ResultCard";

interface Props {
  items: SearchResultItem[];
  loading: boolean;
  onOpen: (item: SearchResultItem) => void;
}

export default function ResultGrid({ items, loading, onOpen }: Props) {
  if (!loading && items.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-500">
        Không có kết quả nào khớp. Thử bỏ filter hoặc đổi cách diễn đạt.
      </p>
    );
  }
  return (
    <div
      className={`grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 ${
        loading ? "opacity-50" : ""
      }`}
    >
      {items.map((item) => (
        <ResultCard key={`${item.image_id}-${item.rank}`} item={item} onOpen={onOpen} />
      ))}
    </div>
  );
}
```

- [ ] **Step 6: `frontend/src/components/AdvancedPanel.tsx`**

Panel này là thứ biến demo thành công cụ đo được: toggle `exact` và `hnsw_ef` cho người chấm tự thấy đánh đổi ANN, filter category cho thấy payload filter của Qdrant.

```tsx
import type { Filters } from "../types";

export interface AdvancedParams {
  k: number;
  exact: boolean;
  hnswEf: number;
  filters: Filters;
}

interface Props {
  params: AdvancedParams;
  onChange: (params: AdvancedParams) => void;
  categories: string[];
}

export default function AdvancedPanel({ params, onChange, categories }: Props) {
  return (
    <details className="rounded-lg border border-slate-700 p-3 text-sm">
      <summary className="cursor-pointer text-slate-300">Tuỳ chọn nâng cao</summary>
      <div className="mt-3 flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Số kết quả: {params.k}</span>
          <input
            type="range"
            min={5}
            max={50}
            step={5}
            value={params.k}
            onChange={(event) => onChange({ ...params, k: Number(event.target.value) })}
          />
        </label>

        <label className="flex items-center gap-2 text-slate-300">
          <input
            type="checkbox"
            checked={params.exact}
            onChange={(event) => onChange({ ...params, exact: event.target.checked })}
          />
          Tìm chính xác (tắt để dùng HNSW)
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-slate-400">hnsw_ef</span>
          <input
            className="w-24 rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100 disabled:opacity-40"
            type="number"
            min={8}
            max={512}
            value={params.hnswEf}
            disabled={params.exact}
            onChange={(event) =>
              onChange({ ...params, hnswEf: Number(event.target.value) })
            }
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Lọc theo category</span>
          <select
            className="h-24 w-48 rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-slate-100"
            multiple
            value={params.filters.categories}
            onChange={(event) =>
              onChange({
                ...params,
                filters: {
                  ...params.filters,
                  categories: Array.from(event.target.selectedOptions, (o) => o.value),
                },
              })
            }
          >
            {categories.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
        </label>
      </div>
    </details>
  );
}
```

- [ ] **Step 7: `frontend/src/components/StatusBar.tsx`**

```tsx
import type { SearchResponse } from "../types";

interface Props {
  response: SearchResponse | null;
  error: string | null;
  loading: boolean;
}

export default function StatusBar({ response, error, loading }: Props) {
  if (error) {
    return (
      <p className="rounded-md border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-200">
        {error}
      </p>
    );
  }
  if (loading) {
    return <p className="text-sm text-slate-400">Đang tìm…</p>;
  }
  if (!response) {
    return <p className="text-sm text-slate-500">Nhập câu chữ hoặc đưa vào một tấm ảnh.</p>;
  }
  return (
    <p className="text-sm text-slate-400">
      <span className="text-slate-200">{response.space}</span> ·{" "}
      {response.results.length}/{response.total_searched} ảnh ·{" "}
      {response.exact ? "exact" : "HNSW"} · {response.latency_ms.toFixed(1)}ms (encode{" "}
      {response.encode_ms.toFixed(1)} + tìm {response.search_ms.toFixed(1)})
    </p>
  );
}
```

- [ ] **Step 8: `frontend/src/App.tsx`**

```tsx
import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchExamples, fetchSpaces } from "./api/client";
import AdvancedPanel, { type AdvancedParams } from "./components/AdvancedPanel";
import ExampleChips from "./components/ExampleChips";
import ModelSelect from "./components/ModelSelect";
import ResultGrid from "./components/ResultGrid";
import SearchBar from "./components/SearchBar";
import StatusBar from "./components/StatusBar";
import { useSearch } from "./hooks/useSearch";
import type { ExampleQuery, SpaceInfo } from "./types";

const DEFAULT_PARAMS: AdvancedParams = {
  k: 20,
  exact: true,
  hnswEf: 128,
  filters: { categories: [], supercategories: [] },
};

export default function App() {
  const [spaces, setSpaces] = useState<SpaceInfo[]>([]);
  const [examples, setExamples] = useState<ExampleQuery[]>([]);
  const [space, setSpace] = useState("clip-b32");
  const [params, setParams] = useState<AdvancedParams>(DEFAULT_PARAMS);
  const [bootError, setBootError] = useState<string | null>(null);
  const { response, loading, error, runText, runImage } = useSearch();

  useEffect(() => {
    Promise.all([fetchSpaces(), fetchExamples()])
      .then(([spaceList, exampleList]) => {
        setSpaces(spaceList);
        setExamples(exampleList);
        const firstReady = spaceList.find((item) => item.ready);
        if (firstReady) setSpace(firstReady.name);
      })
      .catch((cause) => setBootError(String(cause.message ?? cause)));
  }, []);

  /** Category có mặt trong kết quả hiện tại — đủ để lọc mà không cần endpoint riêng. */
  const categories = useMemo(() => {
    const found = new Set<string>();
    response?.results.forEach((item) => item.categories.forEach((c) => found.add(c)));
    return Array.from(found).sort();
  }, [response]);

  const search = useCallback(
    (query: string, overrideSpace?: string) =>
      runText({
        query,
        space: overrideSpace ?? space,
        k: params.k,
        exact: params.exact,
        hnswEf: params.exact ? null : params.hnswEf,
        filters: params.filters,
      }),
    [runText, space, params],
  );

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-5 bg-slate-900 p-6 text-slate-100">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">Tìm kiếm ngữ nghĩa trên COCO val2017</h1>
        <p className="text-sm text-slate-400">
          Tìm bằng câu chữ hoặc bằng một tấm ảnh, trên 5.000 ảnh.
        </p>
      </header>

      {bootError && (
        <p className="rounded-md border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-200">
          {bootError}
        </p>
      )}

      <div className="flex flex-wrap items-end gap-4">
        <ModelSelect spaces={spaces} value={space} onChange={setSpace} label="Model" />
      </div>

      <SearchBar
        onSearchText={(query) => search(query)}
        onSearchImage={(file) =>
          runImage({
            file,
            space,
            k: params.k,
            exact: params.exact,
            hnswEf: params.exact ? null : params.hnswEf,
            filters: params.filters,
          })
        }
        disabled={loading}
      />

      <ExampleChips
        examples={examples}
        onPick={(example) => {
          setSpace(example.space);
          search(example.query, example.space);
        }}
      />

      <AdvancedPanel params={params} onChange={setParams} categories={categories} />
      <StatusBar response={response} error={error} loading={loading} />

      <ResultGrid
        items={response?.results ?? []}
        loading={loading}
        onOpen={() => undefined}
      />
    </main>
  );
}
```

`onOpen` tạm để rỗng — Task 17 gắn modal chi tiết vào đây.

- [ ] **Step 9: Kiểm tra kiểu và build**

Run: `cd frontend && npm run typecheck && npm test`
Expected: `tsc` không báo lỗi; 9 test PASS.

- [ ] **Step 10: Chạy thật và xem bằng mắt**

```bash
cp frontend/.env.example frontend/.env
cd frontend && npm run dev
```
Với API đang chạy ở terminal khác (`python tasks.py serve`), mở http://localhost:5173 và kiểm tra bốn điều:
1. Dropdown model liệt kê 8 space, `siglip-b16` chọn được, `mclip-b32` có nhãn "tiếng Việt".
2. Gõ `a man riding a horse on the beach` rồi Enter → grid hiện ảnh đúng chủ đề, StatusBar hiện tên space và latency.
3. Bấm chip tiếng Việt → tự đổi sang `mclip-b32` và ra kết quả hợp lý.
4. Kéo một ảnh bất kỳ từ máy vào khung → ra ảnh tương tự.

- [ ] **Step 11: Commit**

```bash
git add frontend/src
git commit -m "feat: UI tìm kiếm bằng text và bằng ảnh, panel nâng cao và query mẫu"
```

---

### Task 17: Modal chi tiết, "tìm ảnh tương tự", và chế độ so sánh model

Hai tính năng ở task này là thứ biến demo từ "một ô tìm kiếm" thành thứ minh hoạ được kết quả ablation ngay trước mặt người chấm.

**Files:**
- Create: `frontend/src/components/DetailModal.tsx`, `frontend/src/components/CompareView.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `searchText`, `searchImage`, `API_BASE`, `useSearch`, các kiểu ở `src/types.ts`.
- Produces: `DetailModal({ item, onClose, onFindSimilar })`, `CompareView({ spaces, defaultLeft, defaultRight, params })`.

- [ ] **Step 1: `frontend/src/components/DetailModal.tsx`**

Nút "Tìm ảnh tương tự" gọi `/search/image` với `image_id`, nên người chấm demo được chiều ảnh→ảnh mà không cần có file ảnh nào trong máy.

```tsx
import { API_BASE } from "../api/client";
import type { SearchResultItem } from "../types";

interface Props {
  item: SearchResultItem | null;
  onClose: () => void;
  onFindSimilar: (imageId: number) => void;
}

export default function DetailModal({ item, onClose, onFindSimilar }: Props) {
  if (!item) return null;
  return (
    <div
      className="fixed inset-0 z-10 flex items-center justify-center bg-slate-950/80 p-4"
      onClick={onClose}
    >
      <div
        className="flex max-h-full w-full max-w-3xl flex-col gap-4 overflow-auto rounded-lg border border-slate-700 bg-slate-900 p-4"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-medium">Ảnh {item.image_id}</h2>
            <p className="text-sm text-slate-400">
              hạng {item.rank} · điểm {item.score.toFixed(4)}
            </p>
          </div>
          <button
            className="rounded-md border border-slate-700 px-3 py-1 text-sm hover:border-indigo-500"
            type="button"
            onClick={onClose}
          >
            Đóng
          </button>
        </div>

        <img
          className="max-h-[50vh] w-full rounded-md object-contain"
          src={`${API_BASE}/images/${item.file_name}`}
          alt={item.captions[0] ?? `ảnh ${item.image_id}`}
        />

        <div className="flex flex-col gap-2 text-sm">
          <h3 className="text-slate-300">Caption của người viết</h3>
          <ul className="list-disc pl-5 text-slate-400">
            {item.captions.map((caption, index) => (
              <li key={index}>{caption}</li>
            ))}
          </ul>
          {item.categories.length > 0 && (
            <p className="text-slate-400">
              <span className="text-slate-300">Category: </span>
              {item.categories.join(", ")}
            </p>
          )}
        </div>

        <button
          className="self-start rounded-md bg-indigo-500 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-400"
          type="button"
          onClick={() => onFindSimilar(item.image_id)}
        >
          Tìm ảnh tương tự
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: `frontend/src/components/CompareView.tsx`**

```tsx
import { useState } from "react";

import { ApiError, searchText } from "../api/client";
import type { Filters, SearchResponse, SpaceInfo } from "../types";
import ModelSelect from "./ModelSelect";
import ResultGrid from "./ResultGrid";

interface Props {
  spaces: SpaceInfo[];
  defaultLeft: string;
  defaultRight: string;
  params: { k: number; exact: boolean; hnswEf: number; filters: Filters };
}

interface Side {
  space: string;
  response: SearchResponse | null;
  error: string | null;
}

/**
 * Cùng một query, hai model, hai cột. Hai truy vấn chạy song song để thời gian
 * chờ bằng truy vấn chậm hơn chứ không phải tổng hai bên.
 */
export default function CompareView({ spaces, defaultLeft, defaultRight, params }: Props) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [left, setLeft] = useState<Side>({ space: defaultLeft, response: null, error: null });
  const [right, setRight] = useState<Side>({
    space: defaultRight,
    response: null,
    error: null,
  });

  async function runBoth(text: string) {
    setLoading(true);
    const call = (space: string) =>
      searchText({
        query: text,
        space,
        k: params.k,
        exact: params.exact,
        hnswEf: params.exact ? null : params.hnswEf,
        filters: params.filters,
      })
        .then((response) => ({ response, error: null }))
        .catch((cause) => ({
          response: null,
          error: cause instanceof ApiError ? cause.message : "Lỗi không xác định",
        }));

    const [leftResult, rightResult] = await Promise.all([
      call(left.space),
      call(right.space),
    ]);
    setLeft({ ...left, ...leftResult });
    setRight({ ...right, ...rightResult });
    setLoading(false);
  }

  return (
    <section className="flex flex-col gap-4 rounded-lg border border-slate-700 p-4">
      <h2 className="text-lg font-medium">So sánh hai model trên cùng một query</h2>

      <form
        className="flex flex-wrap items-end gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (query.trim()) runBoth(query.trim());
        }}
      >
        <input
          className="min-w-64 flex-1 rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
          placeholder="Nhập một query rồi xem hai model xếp hạng khác nhau thế nào"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <ModelSelect
          spaces={spaces}
          value={left.space}
          onChange={(space) => setLeft({ ...left, space })}
          label="Model A"
        />
        <ModelSelect
          spaces={spaces}
          value={right.space}
          onChange={(space) => setRight({ ...right, space })}
          label="Model B"
        />
        <button
          className="rounded-md bg-indigo-500 px-4 py-2 font-medium text-white hover:bg-indigo-400 disabled:opacity-50"
          type="submit"
          disabled={loading || !query.trim()}
        >
          So sánh
        </button>
      </form>

      <div className="grid gap-4 lg:grid-cols-2">
        {[left, right].map((side, index) => (
          <div key={index} className="flex flex-col gap-2">
            <p className="text-sm text-slate-400">
              <span className="text-slate-200">{side.space}</span>
              {side.response
                ? ` · ${side.response.latency_ms.toFixed(1)}ms`
                : side.error
                  ? ` · ${side.error}`
                  : ""}
            </p>
            <ResultGrid
              items={side.response?.results ?? []}
              loading={loading}
              onOpen={() => undefined}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Gắn modal và chế độ so sánh vào `frontend/src/App.tsx`**

Thêm import:

```tsx
import CompareView from "./components/CompareView";
import DetailModal from "./components/DetailModal";
import type { SearchResultItem } from "./types";
```

Thêm state và handler bên trong `App`, cạnh các state đã có:

```tsx
  const [selected, setSelected] = useState<SearchResultItem | null>(null);
  const [comparing, setComparing] = useState(false);
```

Rồi mở rộng dòng khai báo hook **đã có** để lấy thêm `runSimilar` (không gọi `useSearch()` lần thứ hai — mỗi lần gọi tạo một state riêng, kết quả sẽ không hiện ra grid):

```tsx
  const { response, loading, error, runText, runImage, runSimilar } = useSearch();
```

Thêm handler:

```tsx
  const findSimilar = useCallback(
    (imageId: number) => {
      setSelected(null);
      runSimilar({
        imageId,
        space,
        k: params.k,
        exact: params.exact,
        hnswEf: params.exact ? null : params.hnswEf,
        filters: params.filters,
      });
    },
    [runSimilar, space, params],
  );
```

Đổi `onOpen={() => undefined}` của `ResultGrid` thành `onOpen={setSelected}`.

Thêm nút bật/tắt chế độ so sánh, ngay cạnh `ModelSelect`:

```tsx
        <button
          className={`rounded-md border px-3 py-2 text-sm ${
            comparing
              ? "border-indigo-500 text-indigo-300"
              : "border-slate-700 text-slate-300 hover:border-indigo-500"
          }`}
          type="button"
          onClick={() => setComparing((value) => !value)}
        >
          {comparing ? "Đang ở chế độ so sánh" : "So sánh hai model"}
        </button>
```

Và ở cuối `main`, sau `ResultGrid`:

```tsx
      {comparing && (
        <CompareView
          spaces={spaces}
          defaultLeft="clip-b32"
          defaultRight="laion-b32"
          params={params}
        />
      )}

      <DetailModal
        item={selected}
        onClose={() => setSelected(null)}
        onFindSimilar={findSimilar}
      />
```

- [ ] **Step 4: Kiểm tra kiểu và test**

Run: `cd frontend && npm run typecheck && npm test`
Expected: `tsc` sạch, 9 test PASS.

- [ ] **Step 5: Kiểm tra bằng mắt ba luồng mới**

Với API và frontend đang chạy:
1. Tìm một query, bấm vào một ảnh → modal hiện ảnh full, đủ 5 caption, danh sách category.
2. Trong modal bấm "Tìm ảnh tương tự" → modal đóng, grid đổi sang ảnh tương tự, **ảnh query không xuất hiện trong kết quả**.
3. Bật "So sánh hai model", chọn `clip-b32` và `laion-b32`, nhập `a street sign covered in snow` → hai cột kết quả khác nhau, mỗi cột có latency riêng.

- [ ] **Step 6: Commit**

```bash
git add frontend/src
git commit -m "feat: modal chi tiết, tìm ảnh tương tự và chế độ so sánh hai model"
```

---

### Task 18: Phân tích định tính, báo cáo và README

**Files:**
- Create: `docs/report.md`, `README.md`
- Create: `results/qualitative.md`, `results/figures/*.png`

**Interfaces:**
- Consumes: `results/*.csv` và `results/*.md` từ Task 14, `results/index_meta/*.json` từ Task 12, hệ đang chạy từ Task 16-17.
- Produces: báo cáo hoàn chỉnh và README chạy lại được.

- [ ] **Step 1: Chạy 10 query phân tích lỗi và chụp ảnh kết quả**

Với hệ đang chạy, chạy lần lượt 10 query dưới đây trên `clip-b32`, `k=10`, chụp ảnh màn hình grid kết quả lưu vào `results/figures/` theo tên đã ghi:

| Nhóm lỗi | Query | File ảnh |
|---|---|---|
| Đếm số lượng | `three dogs` | `bad_count_three_dogs.png` |
| Đếm số lượng | `exactly two people sitting` | `bad_count_two_people.png` |
| Phủ định | `a street with no cars` | `bad_negation_no_cars.png` |
| Phủ định | `a plate without any vegetables` | `bad_negation_no_vegetables.png` |
| Quan hệ không gian | `a cup to the left of a laptop` | `bad_spatial_cup_left.png` |
| Quan hệ không gian | `a dog under a table` | `bad_spatial_dog_under.png` |
| Chữ trong ảnh | `a red sign that says STOP` | `bad_ocr_stop_sign.png` |
| Thuộc tính mịn | `a man in a striped blue shirt` | `bad_attr_striped_shirt.png` |
| Tổ hợp hiếm | `a person riding a zebra` | `bad_rare_riding_zebra.png` |
| Tổ hợp hiếm | `a cat wearing sunglasses` | `bad_rare_cat_sunglasses.png` |

Và 4 query **tốt** để đối chiếu, cùng thư mục:

| Query | File ảnh |
|---|---|
| `a man riding a horse on the beach` | `good_horse_beach.png` |
| `a plate of pizza on a wooden table` | `good_pizza_table.png` |
| `a double decker bus on a city street` | `good_bus_street.png` |
| `skiers on a snowy mountain slope` | `good_skiers_snow.png` |

- [ ] **Step 2: Viết `results/qualitative.md`**

Với mỗi query ở Step 1, viết: query, ảnh chụp, **số kết quả trong top-10 thực sự đúng ý** (đếm bằng mắt), và một câu giải thích cơ chế hỏng. Ví dụ định dạng cho một mục:

```markdown
### Đếm số lượng — "three dogs"

![three dogs](figures/bad_count_three_dogs.png)

Đúng ý: 2/10. Cả 10 ảnh đều có chó, nhưng số lượng sai (1 con hoặc 5+ con).
CLIP học bằng contrastive trên cặp ảnh-caption, không có tín hiệu nào buộc nó
phân biệt "three dogs" với "dogs"; hai câu đó gần như cùng một vector. Đây là
giới hạn của phương pháp, không phải lỗi cài đặt.
```

Thêm một mục tổng kết cuối file: bảng 5 nhóm lỗi × số query đúng ý trung bình, và nhận định nhóm nào hỏng nặng nhất.

- [ ] **Step 3: Viết `docs/report.md`**

Cấu trúc bảy phần, mọi con số **copy từ `results/`**, không gõ lại từ ký ức:

1. **Bài toán và dataset** — COCO val2017, số ảnh và số caption lấy từ output Task 12 Step 3, metadata category. Nêu vì sao chọn dataset có ground-truth.
2. **Kiến trúc hệ thống** — dán sơ đồ ở §3 của spec, mô tả bốn thành phần và vai trò của registry.
3. **Phương pháp** — bảng 5 space + 2 baseline, và quan trọng nhất: **từng cặp so sánh cô lập biến nào** (§5.2 của spec). Nêu rõ `mclip-b32` dùng lại collection ảnh của `clip-b32` và cosine kiểm chứng đo được là bao nhiêu.
4. **Kết quả định lượng** — dán 6 bảng từ `results/axis*.md`. Dưới mỗi bảng viết 2-3 câu đọc bảng, trả lời đúng câu hỏi mà trục đó đặt ra. Thời gian build lấy từ `results/index_meta/*.json`.
5. **Phân tích tốt/xấu** — dẫn `results/qualitative.md`, kèm 4-6 hình tiêu biểu.
6. **Giới hạn và hướng mở rộng** — nêu thẳng: proxy category cho ảnh→ảnh thiên vị ảnh nhiều object; query text→ảnh chỉ lấy mẫu 5.000 caption chứ không phải toàn bộ; val2017 khác Karpathy test split nên số không so trực tiếp với paper CLIP được; bộ query ngắn là 80 category chứ không phải 200. Hướng mở rộng: re-ranking bằng cross-encoder, fine-tune trên domain, dataset thứ hai để kiểm tra cross-domain.
7. **Cách chạy lại** — trỏ sang README.

Độ dài mục tiêu khoảng 5 trang. Mục 4 và 5 là phần được chấm nặng nhất, nên đừng để chúng ngắn hơn mục 2.

- [ ] **Step 4: Viết `README.md`**

```markdown
# Tìm kiếm ngữ nghĩa đa phương thức trên MS-COCO

Tìm ảnh bằng câu chữ tự do hoặc bằng một tấm ảnh, trên 5.000 ảnh COCO val2017.
Backend FastAPI + Qdrant, frontend React. 5 không gian nhúng + 2 baseline, kèm
bộ số liệu ablation trong `results/`.

## Yêu cầu

- Python 3.11, Node 18+
- Docker Desktop đang chạy (Qdrant)
- Khoảng 6GB đĩa trống (ảnh COCO + checkpoint model), không cần GPU

## Chạy lại từ đầu

```bash
docker compose up -d qdrant
pip install -r requirements.txt
cp .env.example .env

python tasks.py ingest                 # tải COCO, sinh corpus.jsonl + thumbnail
python tasks.py build --space all      # encode + nạp Qdrant, ~35 phút trên CPU
python tasks.py querysets              # sinh query set (bản dịch tiếng Việt đã commit)
python tasks.py eval --axis all        # sinh results/*.csv + *.md

python tasks.py serve                  # API tại http://localhost:8000
```

Terminal khác:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev                            # UI tại http://localhost:5173
```

## Test

```bash
python -m pytest                       # suite nhanh, không tải model
python -m pytest -m integration        # test tải model thật
cd frontend && npm test && npm run typecheck
```

## Tài liệu

- Thiết kế: `docs/superpowers/specs/2026-09-23-multimodal-search-design.md`
- Báo cáo: `docs/report.md`
- Bảng số liệu: `results/`
- Phân tích tốt/xấu: `results/qualitative.md`
```

- [ ] **Step 5: Kiểm tra README bằng cách làm đúng theo nó**

Xoá `data/corpus.jsonl` và một collection Qdrant, rồi chạy lại đúng các lệnh trong README để chắc chúng đủ và đúng thứ tự. Nếu phải sửa gì ngoài README mới chạy được thì README còn thiếu — bổ sung rồi thử lại.

- [ ] **Step 6: Chạy toàn bộ kiểm tra lần cuối**

```bash
python -m pytest
cd frontend && npm test && npm run typecheck
```
Expected: tất cả PASS. Ghi lại số test thực tế — đừng viết "mọi test đều xanh" mà chưa chạy.

- [ ] **Step 7: Quay video demo 2-3 phút**

Thứ tự quay, để video đi đúng các mục rubric:
1. Query text tiếng Anh → kết quả tốt (15 giây)
2. Query tiếng Việt với `mclip-b32`, rồi lặp lại cùng query đó với `clip-b32` để thấy nó sụp (30 giây)
3. Kéo một ảnh từ máy vào → ảnh→ảnh (20 giây)
4. Bấm một ảnh → modal → "Tìm ảnh tương tự" (20 giây)
5. Chế độ so sánh `clip-b32` vs `laion-b32` trên một query (25 giây)
6. Tắt `exact`, đặt `hnsw_ef=16`, chỉ ra latency và kết quả đổi thế nào (20 giây)
7. Một query thuộc nhóm lỗi, nói thẳng nó hỏng và vì sao (20 giây)

Mục 7 quan trọng: nêu được giới hạn của hệ mình xây cho thấy hiểu phương pháp.

- [ ] **Step 8: Commit**

```bash
git add README.md docs/report.md results
git commit -m "docs: báo cáo, phân tích tốt/xấu theo 5 nhóm lỗi, và README chạy lại được"
```

---
