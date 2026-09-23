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
