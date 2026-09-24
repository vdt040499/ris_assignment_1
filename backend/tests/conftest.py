import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from qdrant_client import QdrantClient

from app.config import Settings
from app.corpus import caption_payload, caption_point_id, load_corpus, record_payload
from app.encoders.bm25 import Bm25Retriever
from app.search import SearchService
from app.vectordb import VectorStore
from cli.ingest import build_corpus, make_thumbnails

MINI_IMAGES = [
    {"id": 1, "file_name": "000000000001.jpg", "width": 80, "height": 60},
    {"id": 2, "file_name": "000000000002.jpg", "width": 60, "height": 90},
    {"id": 3, "file_name": "000000000003.jpg", "width": 40, "height": 40},
]


@pytest.fixture
def mini_annotations_dir(tmp_path: Path) -> Path:
    """Sinh cặp file annotation COCO thu nhỏ: 3 ảnh, ảnh #3 không có category."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    captions = {
        "images": MINI_IMAGES,
        "annotations": [
            {"image_id": 1, "id": 10, "caption": "a dog on a red sofa"},
            {"image_id": 1, "id": 11, "caption": "a brown dog sleeping"},
            {"image_id": 2, "id": 12, "caption": "two zebras in a field"},
            {"image_id": 3, "id": 13, "caption": "a slice of pepperoni pizza"},
        ],
    }
    instances = {
        "images": MINI_IMAGES,
        "categories": [
            {"id": 18, "name": "dog", "supercategory": "animal"},
            {"id": 24, "name": "zebra", "supercategory": "animal"},
            {"id": 63, "name": "couch", "supercategory": "furniture"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 18},
            {"image_id": 1, "category_id": 63},
            {"image_id": 1, "category_id": 18},
            {"image_id": 2, "category_id": 24},
        ],
    }
    (ann_dir / "captions_val2017.json").write_text(json.dumps(captions), encoding="utf-8")
    (ann_dir / "instances_val2017.json").write_text(json.dumps(instances), encoding="utf-8")
    return ann_dir


@pytest.fixture
def mini_images_dir(tmp_path: Path) -> Path:
    """Sinh 3 file JPEG thật đúng kích thước khai báo trong annotation."""
    img_dir = tmp_path / "val2017"
    img_dir.mkdir()
    for spec in MINI_IMAGES:
        Image.new("RGB", (spec["width"], spec["height"]), color=(120, 60, 30)).save(
            img_dir / spec["file_name"], format="JPEG"
        )
    return img_dir


DIM = 4
TEXT_KEYWORDS = {"dog": 0, "puppy": 0, "couch": 0, "sofa": 0, "zebra": 1, "pizza": 2}
SIZE_TO_INDEX = {(80, 60): 0, (60, 90): 1, (40, 40): 2}


class FakeEncoder:
    """Encoder tất định 4 chiều: đủ để kiểm tra điều phối mà không tải model."""

    name = "fake"
    dim = DIM

    def __init__(self):
        self.seen_texts: list[str] = []

    def encode_texts(self, texts):
        self.seen_texts.extend(texts)
        basis = np.eye(DIM, dtype=np.float32)
        return np.stack([
            basis[next((i for kw, i in TEXT_KEYWORDS.items() if kw in t.lower()), 3)]
            for t in texts
        ])

    def encode_images(self, images):
        basis = np.eye(DIM, dtype=np.float32)
        return np.stack([basis[SIZE_TO_INDEX.get(img.size, 3)] for img in images])


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, qdrant_mode="embedded",
                    top_k_default=2, max_top_k=5)


@pytest.fixture
def corpus(mini_annotations_dir, settings):
    build_corpus(mini_annotations_dir, settings.corpus_path)
    return load_corpus(settings.corpus_path)


@pytest.fixture
def service(settings, corpus, encoder, mini_images_dir):
    store = VectorStore(settings, client=QdrantClient(location=":memory:"))
    images = [
        Image.open(mini_images_dir / r.file_name).convert("RGB") for r in corpus.records
    ]
    vectors = encoder.encode_images(images)

    store.ensure_collection("coco_clip_b32", dim=DIM, distance="Cosine")
    store.upsert("coco_clip_b32", ids=corpus.image_ids(), vectors=vectors,
                 payloads=[record_payload(r) for r in corpus.records], batch_size=8)
    store.create_payload_indexes("coco_clip_b32", ["categories", "supercategories"])

    caption_ids, caption_vectors, caption_payloads = [], [], []
    for record in corpus.records:
        for idx in range(len(record.captions)):
            caption_ids.append(caption_point_id(record.image_id, idx))
            caption_vectors.append(encoder.encode_texts([record.captions[idx]])[0])
            caption_payloads.append(caption_payload(record, idx))
    store.ensure_collection("coco_cap_clip_b32", dim=DIM, distance="Cosine")
    store.upsert("coco_cap_clip_b32", ids=caption_ids,
                 vectors=np.stack(caption_vectors), payloads=caption_payloads,
                 batch_size=8)

    Bm25Retriever.build(corpus).save(settings.bm25_path)
    make_thumbnails(corpus, mini_images_dir, settings.thumbs_dir, settings.thumb_size)
    return SearchService(settings, store, corpus, encoder_factory=lambda name: encoder)
