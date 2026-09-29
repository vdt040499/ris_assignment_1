"""Test that loads real models. Run with: python -m pytest -m integration

The test `test_multilingual_shares_the_clip_space` is a **gate**: if it goes
red, the assumption that "mclip-b32 reuses clip-b32's image collection" is
wrong, and the registry must be fixed to give mclip-b32 its own collection
BEFORE building the index.
"""

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.encoders import build_encoder

pytestmark = pytest.mark.integration

MULTILINGUAL_ALIGNMENT_THRESHOLD = 0.9
SAMPLE_CAPTION = "a man riding a horse on the beach"


@pytest.fixture(scope="module")
def settings():
    return Settings(_env_file=None, batch_size=4)


def test_clip_b32_encodes_text_and_images_to_unit_vectors(settings):
    encoder = build_encoder("clip-b32", settings)
    texts = encoder.encode_texts([SAMPLE_CAPTION, "a plate of pasta"])
    images = encoder.encode_images([Image.new("RGB", (64, 64), (10, 120, 200))])
    assert texts.shape == (2, 512)
    assert images.shape == (1, 512)
    np.testing.assert_allclose(np.linalg.norm(texts, axis=1), [1.0, 1.0], atol=1e-4)


def test_query_longer_than_the_token_limit_is_truncated_not_rejected(settings):
    encoder = build_encoder("clip-b32", settings)
    long_query = " ".join(["a very detailed photograph of a busy city street"] * 40)
    vectors = encoder.encode_texts([long_query])
    assert vectors.shape == (1, 512)
    assert not np.isnan(vectors).any()


def test_raw_space_returns_unnormalized_vectors(settings):
    encoder = build_encoder("clip-b32-raw", settings)
    norms = np.linalg.norm(encoder.encode_texts([SAMPLE_CAPTION]), axis=1)
    assert norms[0] > 1.5


def test_siglip_reports_its_own_dimension(settings):
    encoder = build_encoder("siglip-b16", settings)
    assert encoder.encode_texts([SAMPLE_CAPTION]).shape == (1, 768)


def test_multilingual_shares_the_clip_space(settings):
    """Gate for the assumption in spec §5.3."""
    clip = build_encoder("clip-b32", settings)
    mclip = build_encoder("mclip-b32", settings)
    a = clip.encode_texts([SAMPLE_CAPTION])[0]
    b = mclip.encode_texts([SAMPLE_CAPTION])[0]
    cosine = float(np.dot(a, b))
    assert cosine > MULTILINGUAL_ALIGNMENT_THRESHOLD, (
        f"cosine={cosine:.3f} — mclip-b32 does NOT share a space with clip-b32. "
        "Fix the registry: give mclip-b32 its own collection_suffix and builds_index=True."
    )


def test_resnet_produces_2048_dimensional_features(settings):
    encoder = build_encoder("resnet50", settings)
    assert encoder.encode_images([Image.new("RGB", (64, 64))]).shape == (1, 2048)
