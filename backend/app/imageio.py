"""Normalize uploaded images before feeding them into the encoder.

Images users upload don't look like images in the dataset: they can be
transparent PNGs, grayscale scans, CMYK images from a print shop, or 12MP
phone photos rotated by EXIF. The encoder only accepts RGB at a reasonable
size, so all the work of converting to a standard form happens here, and only here.
"""

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.errors import BadImageError, ImageTooLargeError


def downscale(image: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale the image to under ``max_pixels`` pixels, preserving the aspect ratio.

    Returns the input image itself if it's already small enough. The
    smallest edge always keeps at least 1 pixel, so an extremely flat image
    (e.g. 1000×2) doesn't shrink to nothing.

    :param image: PIL Image object to potentially downscale.
    :param max_pixels: Maximum total pixel count allowed (width × height).
    :return: Image, either original if under budget or downscaled version.
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
    """Turn uploaded bytes into an RGB image, correctly oriented and small enough to encode.

    :param data: Raw image bytes to decode and normalize.
    :param content_type: MIME type declared by the client; ``None`` skips the
        check and relies solely on whether the bytes can be decoded.
    :param settings: Configuration object containing upload limits and allowed types.
    :return: PIL Image in RGB mode, correctly oriented, and downscaled as needed.
    :raises ImageTooLargeError: exceeds ``MAX_UPLOAD_MB`` (checked before
        decoding, so a 500MB junk file never gets loaded into RAM). Also guards
        against decompression bombs (small files with huge declared dimensions).
    :raises BadImageError: content type is not allowed, or the bytes aren't a valid image.
    """
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(
            f"Image is {len(data) / 1024 / 1024:.1f}MB, exceeding the "
            f"{settings.max_upload_mb}MB limit"
        )
    if content_type and content_type not in settings.allowed_image_types_set:
        raise BadImageError(
            f"Format '{content_type}' is not supported. Allowed: "
            f"{', '.join(sorted(settings.allowed_image_types_set))}"
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BadImageError(f"Could not decode image: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return downscale(image, settings.max_image_pixels)


def load_corpus_image(path: Path, settings: Settings) -> Image.Image:
    """Load an image already present in the corpus, normalized the same way as an upload.

    Used for the "Find similar images" button: the user doesn't upload
    anything, the system re-encodes an image that's already on disk.
    Re-encoding produces the exact same vector already in the index (the
    model is deterministic), so there's no need for a path that reads the
    vector back out of Qdrant.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return downscale(image, settings.max_image_pixels)
