"""Factory encoder theo tên space, kèm cache LRU dùng chung cho cả API."""

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
    """Tạo một encoder mới, không qua cache.

    :param space_name: tên space trong registry (vd ``"clip-b32"``).
    :param settings: cấu hình dùng để lấy ``device``/``batch_size``; mặc định
        dùng ``get_settings()``.
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
    """Lấy encoder từ cache LRU dùng chung; nạp model nếu chưa có trong cache.

    :param space_name: tên space trong registry.
    :param settings: cấu hình dùng để dựng cache lần đầu (kích thước cache,
        device, batch_size); mặc định dùng ``get_settings()``.

    Lấy ``_cache`` vào biến cục bộ **bên trong** lock rồi mới gọi ``.get()``
    bên ngoài lock. Nếu đọc lại global ``_cache`` sau khi đã nhả lock, một
    lệnh gọi ``reset_encoder_cache()`` xen giữa có thể set nó về ``None``,
    khiến ``.get()`` ném ``AttributeError`` — bind vào biến cục bộ đóng kín
    khoảng hở đó. Việc load model đồng thời cho cùng một key vẫn được
    ``LruEncoderCache.get()`` tự khoá và xử lý, không liên quan gì tới sửa
    đổi này.
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
    """Xoá cache. Dùng trong test và khi cần giải phóng RAM."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.clear()
        _cache = None
