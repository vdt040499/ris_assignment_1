import threading

import numpy as np
import pytest

from app.config import Settings
from app.encoders import build_encoder, get_encoder, reset_encoder_cache, warmup
from app.encoders.base import LruEncoderCache
from app.errors import ModeNotSupportedError, UnknownSpaceError
from app.registry import get_space


class FakeEncoder:
    def __init__(self, name):
        self.name = name
        self.dim = 4

    def encode_texts(self, texts):
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def encode_images(self, images):
        return np.zeros((len(images), self.dim), dtype=np.float32)


@pytest.fixture
def settings():
    return Settings(_env_file=None, model_cache_size=2)


@pytest.fixture(autouse=True)
def clean_cache():
    reset_encoder_cache()
    yield
    reset_encoder_cache()


def test_cache_returns_the_same_instance_for_one_key():
    cache = LruEncoderCache(2, FakeEncoder)
    assert cache.get("a") is cache.get("a")
    assert len(cache) == 1


def test_cache_evicts_the_least_recently_used():
    cache = LruEncoderCache(2, FakeEncoder)
    first = cache.get("a")
    cache.get("b")
    cache.get("c")
    assert cache.keys() == ["b", "c"]
    assert cache.get("a") is not first


def test_reading_a_key_makes_it_recent():
    cache = LruEncoderCache(2, FakeEncoder)
    a = cache.get("a")
    cache.get("b")
    assert cache.get("a") is a
    cache.get("c")
    assert cache.keys() == ["a", "c"]


def test_maxsize_below_one_is_rejected():
    with pytest.raises(ValueError):
        LruEncoderCache(0, FakeEncoder)


def test_concurrent_gets_build_the_encoder_once():
    calls: list[str] = []

    def counting_factory(name):
        calls.append(name)
        return FakeEncoder(name)

    cache = LruEncoderCache(2, counting_factory)
    threads = [threading.Thread(target=cache.get, args=("a",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["a"]


def test_bm25_space_has_no_embedding_encoder(settings):
    with pytest.raises(ValueError):
        build_encoder("bm25-cap", settings)


def test_unknown_space_raises(settings):
    with pytest.raises(UnknownSpaceError):
        build_encoder("clip-b99", settings)


def test_resnet_refuses_text_before_loading_any_model(settings):
    encoder = build_encoder("resnet50", settings)
    with pytest.raises(ModeNotSupportedError):
        encoder.encode_texts(["a dog"])


def test_multilingual_refuses_images_before_loading_any_model(settings):
    encoder = build_encoder("mclip-b32", settings)
    with pytest.raises(ModeNotSupportedError):
        encoder.encode_images([object()])


def test_empty_input_returns_empty_matrix_without_loading(settings):
    encoder = build_encoder("clip-b32", settings)
    assert encoder.encode_texts([]).shape == (0, get_space("clip-b32").dim)
    assert encoder.encode_images([]).shape == (0, get_space("clip-b32").dim)


def test_get_encoder_uses_the_shared_cache(settings):
    assert get_encoder("clip-b32", settings) is get_encoder("clip-b32", settings)


def test_warmup_encodes_one_sample_per_space_and_skips_bm25(monkeypatch):
    calls = []

    class Recorder(FakeEncoder):
        def encode_texts(self, texts):
            calls.append((self.name, "text"))
            return super().encode_texts(texts)

        def encode_images(self, images):
            calls.append((self.name, "image"))
            return super().encode_images(images)

    monkeypatch.setattr("app.encoders.build_encoder", lambda name, settings=None: Recorder(name))
    cfg = Settings(_env_file=None, warmup_spaces="clip-b32, resnet50,bm25-cap")
    assert warmup(cfg) == ["clip-b32", "resnet50"]
    assert calls == [("clip-b32", "text"), ("resnet50", "image")]


def test_warmup_survives_a_failing_space(monkeypatch):
    def factory(name, settings=None):
        if name == "clip-b32":
            raise OSError("offline")
        return FakeEncoder(name)

    monkeypatch.setattr("app.encoders.build_encoder", factory)
    cfg = Settings(_env_file=None, warmup_spaces="clip-b32,mclip-b32")
    assert warmup(cfg) == ["mclip-b32"]
