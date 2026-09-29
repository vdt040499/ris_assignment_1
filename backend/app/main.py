"""HTTP layer. Deliberately thin: all business decisions live in SearchService.

The only thing this layer does besides routing is translate domain exceptions
into status codes and block path traversal when serving image files.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.corpus import load_corpus
from app.encoders import warmup
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
    {"label": "Dog on a sofa", "query": "a dog lying on a sofa",
     "space": "clip-b32", "language": "en"},
    {"label": "Man riding a horse on the beach",
     "query": "a man riding a horse on the beach",
     "space": "clip-b32", "language": "en"},
    {"label": "City street at night with neon lights",
     "query": "a city street at night with neon lights",
     "space": "laion-b32", "language": "en"},
    {"label": "Three dogs (counting test)", "query": "three dogs",
     "space": "clip-b32", "language": "en"},
    {"label": "Cat sleeping on a bed",
     "query": "một con mèo đang ngủ trên giường",
     "space": "mclip-b32", "language": "vi"},
    {"label": "Two people playing tennis",
     "query": "hai người đang chơi tennis",
     "space": "mclip-b32", "language": "vi"},
    {"label": "Dinner table with pizza and wine",
     "query": "bàn ăn có pizza và một ly rượu vang",
     "space": "mclip-b32", "language": "vi"},
    {"label": "Red bus on the street",
     "query": "một chiếc xe buýt màu đỏ trên đường phố",
     "space": "mclip-b32", "language": "vi"},
)


def _safe_path(directory: Path, file_name: str) -> Path:
    """Resolve the file path and make sure it stays within ``directory``.

    Blocks both ``../`` and symlinks pointing outside. Every invalid case
    returns 404 just like a missing file, so the directory structure isn't
    leaked.
    """
    root = directory.resolve()
    candidate = (root / file_name).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


def create_app(
    settings: Settings | None = None, service: SearchService | None = None
) -> FastAPI:
    """Build the app.

    :param service: pre-injected service (used by tests); if ``None`` the app
        builds itself from corpus.jsonl and Qdrant according to the configuration.
    """
    settings = settings or get_settings()
    if service is None:
        store = VectorStore(settings)
        corpus = load_corpus(settings.corpus_path)
        service = SearchService(settings, store, corpus)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Chạy trong thread để không chặn event loop; server nhận request sau khi xong.
        await run_in_threadpool(warmup, settings)
        yield

    app = FastAPI(title="COCO multimodal search", lifespan=lifespan)
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
        k: int | None = Form(default=None, ge=1),
        exact: bool = Form(default=True),
        hnsw_ef: int | None = Form(default=None, ge=1),
        filters_json: str | None = Form(default=None),
        image_id: int | None = Form(default=None),
        file: UploadFile | None = File(default=None),
    ) -> SearchResponse:
        """Search by image. Only accepts multipart; must have exactly one of file / image_id."""
        try:
            filters = (
                Filters.model_validate_json(filters_json) if filters_json else Filters()
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=f"invalid filters_json: {exc}") from exc

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
    """Entry point for uvicorn: `uvicorn app.main:get_app --factory`."""
    return create_app()
