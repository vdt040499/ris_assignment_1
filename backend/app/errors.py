"""Domain exceptions. The HTTP layer maps these classes to status codes in app/main.py."""


class SearchError(Exception):
    """Root of every domain error in the system."""


class UnknownSpaceError(SearchError):
    """Embedding space name not found in the registry → HTTP 404."""


class IndexNotBuiltError(SearchError):
    """Valid space but the index hasn't been built yet → HTTP 409."""


class ModeNotSupportedError(SearchError):
    """Space doesn't support the requested query direction → HTTP 400."""


class VectorStoreDownError(SearchError):
    """Could not connect to Qdrant → HTTP 503."""


class BadImageError(SearchError):
    """Uploaded image has the wrong format or couldn't be decoded → HTTP 400."""


class ImageTooLargeError(SearchError):
    """Uploaded image exceeds the byte limit → HTTP 413."""


class BadRequestError(SearchError):
    """Request has invalid logic (e.g. missing both file and image_id) → HTTP 400."""


class CorpusError(SearchError):
    """corpus.jsonl is missing, empty, or has an invalid schema."""


class ValidationRangeError(SearchError):
    """Parameter has the right type but is out of the allowed range (e.g. k > MAX_TOP_K) → HTTP 422."""
