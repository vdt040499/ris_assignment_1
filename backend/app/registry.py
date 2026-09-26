"""Registry of embedding spaces.

This is the sole interface between the model layer and the rest of the system:
the API, frontend, and eval only know about SpaceSpec, not what CLIP or SigLIP
are. Adding a model means adding an entry here and changing nothing else.
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
    """Declaration of an embedding space.

    :param collection_suffix: Qdrant collection name suffix; ``None`` if the
        space is not stored in Qdrant (e.g. BM25).
    :param encoder_key: selects which encoder class builds this space.
    :param text_padding: the tokenizer's padding strategy; SigLIP requires
        ``max_length``, CLIP uses ``longest``.
    :param builds_index: ``False`` means the space reuses another space's
        image collection, so the build step must skip it.
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
        note="changes the patch size relative to the baseline",
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
        note="same architecture as the baseline, trained on LAION-2B data instead",
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
        note="switches the loss function to sigmoid",
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
        note="multilingual text tower, reuses clip-b32's image collection",
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
        note="control group for the normalization ablation",
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
        note="non-multimodal baseline",
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
        note="non-semantic baseline, keyword matching on captions",
    ),
}


def get_space(name: str) -> SpaceSpec:
    """Look up a space by name.

    :raises UnknownSpaceError: if the name is not in the registry, including
        the list of valid names so the caller can fix it right away.
    """
    try:
        return SPACES[name]
    except KeyError as exc:
        raise UnknownSpaceError(
            f"Space '{name}' does not exist. Valid: {', '.join(sorted(SPACES))}"
        ) from exc


def list_space_names() -> list[str]:
    return sorted(SPACES)


def collection_name(space: SpaceSpec, settings: Settings, target: str = "image") -> str:
    """Qdrant collection name for a space.

    :param target: ``"image"`` for the image collection, ``"caption"`` for the
        caption collection (used for the image→text direction).
    :raises ValueError: if the space is not stored in Qdrant, or target is invalid.
    """
    if space.backend != BACKEND_QDRANT or space.collection_suffix is None:
        raise ValueError(f"Space '{space.name}' does not use Qdrant")
    if target not in ("image", "caption"):
        raise ValueError(f"invalid target: {target}")
    suffix = space.collection_suffix if target == "image" else f"cap_{space.collection_suffix}"
    return f"{settings.collection_prefix}_{suffix}"
