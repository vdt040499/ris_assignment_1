import pytest

from app.config import Settings
from app.errors import UnknownSpaceError
from app.registry import (
    BACKEND_BM25,
    BACKEND_QDRANT,
    MODE_IMAGE2IMAGE,
    MODE_TEXT2IMAGE,
    SPACES,
    collection_name,
    get_space,
    list_space_names,
)

KNOWN_ENCODERS = {"hf_dual", "st_multilingual", "resnet", "bm25"}


@pytest.fixture
def settings():
    return Settings(_env_file=None)


def test_all_eight_spaces_declared():
    assert set(list_space_names()) == {
        "clip-b32", "clip-b16", "laion-b32", "siglip-b16",
        "mclip-b32", "clip-b32-raw", "resnet50", "bm25-cap",
    }


def test_dict_key_matches_space_name():
    for key, space in SPACES.items():
        assert key == space.name


def test_every_space_uses_a_known_encoder():
    for space in SPACES.values():
        assert space.encoder_key in KNOWN_ENCODERS


def test_dims_match_spec():
    assert get_space("clip-b32").dim == 512
    assert get_space("clip-b16").dim == 512
    assert get_space("laion-b32").dim == 512
    assert get_space("siglip-b16").dim == 768
    assert get_space("mclip-b32").dim == 512
    assert get_space("resnet50").dim == 2048
    assert get_space("bm25-cap").dim is None


def test_unknown_space_raises():
    with pytest.raises(UnknownSpaceError):
        get_space("clip-b99")


def test_multilingual_reuses_baseline_image_collection(settings):
    assert collection_name(get_space("mclip-b32"), settings) == collection_name(
        get_space("clip-b32"), settings
    )
    assert get_space("mclip-b32").builds_index is False


def test_raw_space_is_unnormalized_dot(settings):
    raw = get_space("clip-b32-raw")
    assert raw.distance == "Dot"
    assert raw.normalized is False
    assert raw.builds_index is True
    assert collection_name(raw, settings) == "coco_clip_b32_raw"


def test_caption_target_gets_its_own_collection(settings):
    assert collection_name(get_space("clip-b32"), settings, target="caption") == (
        "coco_cap_clip_b32"
    )


def test_bm25_space_has_no_qdrant_collection(settings):
    space = get_space("bm25-cap")
    assert space.backend == BACKEND_BM25
    with pytest.raises(ValueError):
        collection_name(space, settings)


def test_siglip_needs_max_length_padding():
    assert get_space("siglip-b16").text_padding == "max_length"
    assert get_space("clip-b32").text_padding == "longest"


def test_modes_reflect_what_each_space_can_do():
    assert MODE_TEXT2IMAGE in get_space("clip-b32").modes
    assert get_space("resnet50").modes == (MODE_IMAGE2IMAGE,)
    assert get_space("mclip-b32").modes == (MODE_TEXT2IMAGE,)
    assert get_space("bm25-cap").modes == (MODE_TEXT2IMAGE,)


def test_qdrant_spaces_all_declare_dim_and_distance():
    for space in SPACES.values():
        if space.backend == BACKEND_QDRANT:
            assert space.dim is not None and space.distance is not None


def test_vietnamese_supported_only_by_multilingual():
    assert "vi" in get_space("mclip-b32").languages
    assert "vi" not in get_space("clip-b32").languages
