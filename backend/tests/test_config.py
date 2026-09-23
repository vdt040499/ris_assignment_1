from pathlib import Path

from app.config import Settings, get_settings


def test_defaults_match_env_example():
    s = Settings(_env_file=None)
    assert s.top_k_default == 20
    assert s.max_top_k == 100
    assert s.device == "cpu"
    assert s.collection_prefix == "coco"
    assert s.qdrant_mode == "server"


def test_derived_paths_hang_off_data_dir():
    s = Settings(_env_file=None, data_dir=Path("mydata"))
    assert s.corpus_path == Path("mydata/corpus.jsonl")
    assert s.images_dir == Path("mydata/val2017")
    assert s.annotations_dir == Path("mydata/annotations")
    assert s.thumbs_dir == Path("mydata/thumbs")
    assert s.cache_dir == Path("mydata/cache")
    assert s.index_meta_dir == Path("mydata/index_meta")
    assert s.bm25_path == Path("mydata/bm25_index.pkl")
    assert s.qdrant_path == Path("mydata/qdrant_local")


def test_csv_fields_parse_into_collections():
    s = Settings(_env_file=None, allowed_image_types="image/jpeg, image/png",
                 cors_origins="http://a.test,http://b.test")
    assert s.allowed_image_types_set == frozenset({"image/jpeg", "image/png"})
    assert s.cors_origins_list == ["http://a.test", "http://b.test"]


def test_upload_limit_converts_mb_to_bytes():
    s = Settings(_env_file=None, max_upload_mb=3)
    assert s.max_upload_bytes == 3 * 1024 * 1024


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("TOP_K_DEFAULT", "7")
    get_settings.cache_clear()
    assert get_settings().top_k_default == 7
    get_settings.cache_clear()
