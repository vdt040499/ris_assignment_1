import json
from pathlib import Path

import pytest
from PIL import Image

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
            {"image_id": 3, "id": 13, "caption": "a slice of pizza"},
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
