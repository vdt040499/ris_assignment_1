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


class ValidationRangeError(SearchError):
    """Tham số hợp kiểu nhưng ngoài khoảng cho phép (vd k > MAX_TOP_K) → HTTP 422."""
