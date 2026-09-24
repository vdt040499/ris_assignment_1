import io

import pytest
from PIL import Image

from app.config import Settings
from app.errors import BadImageError, ImageTooLargeError
from app.imageio import downscale, load_upload_image


@pytest.fixture
def settings():
    return Settings(_env_file=None, max_upload_mb=1, max_image_pixels=10_000)


def encode(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_grayscale_upload_becomes_rgb(settings):
    data = encode(Image.new("L", (20, 20), 128))
    assert load_upload_image(data, "image/png", settings).mode == "RGB"


def test_rgba_upload_becomes_rgb(settings):
    data = encode(Image.new("RGBA", (20, 20), (1, 2, 3, 128)))
    assert load_upload_image(data, "image/png", settings).mode == "RGB"


def test_cmyk_upload_becomes_rgb(settings):
    data = encode(Image.new("CMYK", (20, 20)), fmt="JPEG")
    assert load_upload_image(data, "image/jpeg", settings).mode == "RGB"


def test_oversized_image_is_downscaled_below_the_pixel_budget(settings):
    data = encode(Image.new("RGB", (400, 300)))
    out = load_upload_image(data, "image/png", settings)
    assert out.size[0] * out.size[1] <= settings.max_image_pixels
    assert out.size[0] / out.size[1] == pytest.approx(400 / 300, rel=0.05)


def test_small_image_is_left_alone(settings):
    data = encode(Image.new("RGB", (50, 50)))
    assert load_upload_image(data, "image/png", settings).size == (50, 50)


def test_downscale_never_produces_a_zero_dimension():
    out = downscale(Image.new("RGB", (1000, 2)), max_pixels=10)
    assert min(out.size) >= 1


def test_disallowed_content_type_is_rejected(settings):
    with pytest.raises(BadImageError):
        load_upload_image(encode(Image.new("RGB", (10, 10))), "image/gif", settings)


def test_bytes_that_are_not_an_image_are_rejected(settings):
    with pytest.raises(BadImageError):
        load_upload_image(b"this is not an image", "image/png", settings)


def test_upload_above_the_byte_limit_is_rejected(settings):
    with pytest.raises(ImageTooLargeError):
        load_upload_image(b"x" * (settings.max_upload_bytes + 1), "image/png", settings)


def test_missing_content_type_is_allowed_if_bytes_decode(settings):
    data = encode(Image.new("RGB", (10, 10)))
    assert load_upload_image(data, None, settings).size == (10, 10)
