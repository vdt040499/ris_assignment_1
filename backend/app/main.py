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
        k: int | None = Form(default=None, ge=1),
        exact: bool = Form(default=True),
        hnsw_ef: int | None = Form(default=None, ge=1),
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
