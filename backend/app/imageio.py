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
    """Biến bytes upload thành ảnh RGB đã xoay đúng và đủ nhỏ để encode.

    :param data: Raw image bytes to decode and normalize.
    :param content_type: MIME type do client khai; ``None`` thì bỏ qua và chỉ
        dựa vào việc bytes có giải mã được hay không.
    :param settings: Configuration object containing upload limits and allowed types.
    :return: PIL Image in RGB mode, correctly oriented, and downscaled as needed.
    :raises ImageTooLargeError: vượt ``MAX_UPLOAD_MB`` (kiểm tra trước khi giải
        mã, để một file rác 500MB không bị nạp vào RAM). Also guards against
        decompression bombs (small files with huge declared dimensions).
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
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BadImageError(f"Không giải mã được ảnh: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return downscale(image, settings.max_image_pixels)
