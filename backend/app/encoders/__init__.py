"""Encoder factory by space name, with an LRU cache shared across the API."""

import logging
import threading

from PIL import Image

from app.config import Settings, get_settings
from app.encoders.base import Encoder, LruEncoderCache
from app.encoders.hf_dual import HFDualEncoder
from app.encoders.resnet import ResNetEncoder
from app.encoders.st_multilingual import MultilingualTextEncoder
from app.registry import MODE_TEXT2IMAGE, get_space

logger = logging.getLogger(__name__)

WARMUP_TEXT = "warm-up"

_ENCODER_CLASSES = {
    "hf_dual": HFDualEncoder,
    "st_multilingual": MultilingualTextEncoder,
    "resnet": ResNetEncoder,
}

_cache: LruEncoderCache | None = None
_cache_lock = threading.Lock()


def build_encoder(space_name: str, settings: Settings | None = None) -> Encoder:
    """Create a new encoder, bypassing the cache.

    :param space_name: space name in the registry (e.g. ``"clip-b32"``).
    :param settings: settings used to get ``device``/``batch_size``; defaults
        to ``get_settings()``.
    :raises UnknownSpaceError: the space name is not in the registry.
    :raises ValueError: the space does not use an embedding encoder (e.g. ``bm25-cap``,
        which is served by ``app.encoders.bm25.Bm25Retriever`` instead of here).
    """
    settings = settings or get_settings()
    space = get_space(space_name)
    try:
        encoder_class = _ENCODER_CLASSES[space.encoder_key]
    except KeyError as exc:
        raise ValueError(
            f"Space '{space_name}' uses backend '{space.backend}', which has no embedding encoder"
        ) from exc
    return encoder_class(space, settings.device, settings.batch_size)


def get_encoder(space_name: str, settings: Settings | None = None) -> Encoder:
    """Get an encoder from the shared LRU cache; load the model if not already cached.

    :param space_name: space name in the registry.
    :param settings: settings used to build the cache on first use (cache size,
        device, batch_size); defaults to ``get_settings()``.

    Binds ``_cache`` to a local variable **inside** the lock, then calls ``.get()``
    outside the lock. If the global ``_cache`` were read again after releasing the
    lock, an interleaved call to ``reset_encoder_cache()`` could set it back to
    ``None``, causing ``.get()`` to raise ``AttributeError`` — binding to a local
    variable closes that gap. Concurrent loading of the same key is still locked
    and handled by ``LruEncoderCache.get()`` itself; this fix is unrelated to that.
    """
    global _cache
    settings = settings or get_settings()
    with _cache_lock:
        if _cache is None:
            _cache = LruEncoderCache(
                settings.model_cache_size,
                lambda name: build_encoder(name, settings),
            )
        cache = _cache
    return cache.get(space_name)


def warmup(settings: Settings | None = None) -> list[str]:
    """Nạp sẵn model của ``settings.warmup_spaces`` vào cache dùng chung.

    Encoder nạp model lười ở lần ``encode_*`` đầu, nên ngoài ``get_encoder``
    còn phải encode thử một mẫu. Space chỉ nhận ảnh (vd resnet50) được thử bằng
    ảnh trắng cỡ ``warmup_image_size``. Space không có encoder nhúng (vd
    ``bm25-cap``) bị bỏ qua. Một space lỗi (offline, thiếu weight) chỉ bị log,
    không chặn các space còn lại hay việc khởi động server.

    :returns: tên các space đã nạp thành công.
    """
    settings = settings or get_settings()
    names = [n for n in settings.warmup_spaces_list if get_space(n).encoder_key in _ENCODER_CLASSES]
    if len(names) > settings.model_cache_size:
        logger.warning(
            "warmup_spaces (%d) nhiều hơn model_cache_size (%d): space nạp trước sẽ bị đẩy khỏi cache",
            len(names), settings.model_cache_size,
        )
    loaded: list[str] = []
    for name in names:
        try:
            encoder = get_encoder(name, settings)
            if MODE_TEXT2IMAGE in get_space(name).modes:
                encoder.encode_texts([WARMUP_TEXT])
            else:
                size = (settings.warmup_image_size,) * 2
                encoder.encode_images([Image.new("RGB", size)])
            loaded.append(name)
        except Exception:
            logger.exception("Warm-up space '%s' thất bại", name)
    return loaded


def reset_encoder_cache() -> None:
    """Clear the cache. Used in tests and when RAM needs to be freed."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.clear()
        _cache = None
