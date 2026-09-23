"""Tải COCO val2017 và sinh corpus.jsonl + thumbnail.

Chạy một lần. Mọi bước đều resume được: file đã tải không tải lại, thumbnail
đã có không sinh lại.
"""

import argparse
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageOps

from app.config import Settings, get_settings
from app.corpus import Corpus, CorpusRecord, load_corpus, write_corpus

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DOWNLOAD_CHUNK_BYTES = 1 << 20
CAPTIONS_FILE = "captions_val2017.json"
INSTANCES_FILE = "instances_val2017.json"


@dataclass
class IngestStats:
    """Số liệu đếm được từ dữ liệu thật, để báo cáo không phải giả định."""

    n_images: int
    n_captions: int
    n_without_category: int
    min_captions: int
    max_captions: int


def download_if_missing(url: str, dest: Path) -> bool:
    """Tải ``url`` về ``dest`` nếu chưa có. Trả True nếu lần này thật sự tải.

    Ghi vào file ``.part`` rồi mới đổi tên, nên một lần tải bị ngắt không để
    lại file hỏng trông như đã xong. Zip tải xong được kiểm tra tính toàn vẹn.
    """
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                fh.write(chunk)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(part) as zf:
            if zf.testzip() is not None:
                part.unlink()
                raise RuntimeError(f"Zip tải về bị hỏng: {url}")
    part.rename(dest)
    return True


def extract_zip(zip_path: Path, dest_dir: Path, marker: Path) -> bool:
    """Giải nén nếu ``marker`` chưa tồn tại. Trả True nếu lần này thật sự giải nén."""
    if marker.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return True


def build_corpus(annotations_dir: Path, out_path: Path) -> IngestStats:
    """Gộp caption và category của COCO thành corpus.jsonl.

    Ảnh không có annotation category vẫn được giữ với mảng rỗng — bỏ chúng đi
    sẽ làm lệch mẫu số của mọi metric sau này. Ảnh không có caption nào thì bị
    loại, vì nó không dùng được cho cả eval lẫn baseline BM25.
    """
    captions_raw = json.loads((annotations_dir / CAPTIONS_FILE).read_text(encoding="utf-8"))
    instances_raw = json.loads((annotations_dir / INSTANCES_FILE).read_text(encoding="utf-8"))

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in captions_raw["annotations"]:
        captions_by_image[ann["image_id"]].append(ann["caption"].strip())

    category_by_id = {c["id"]: c for c in instances_raw["categories"]}
    cats_by_image: dict[int, set[str]] = defaultdict(set)
    supercats_by_image: dict[int, set[str]] = defaultdict(set)
    for ann in instances_raw["annotations"]:
        cat = category_by_id.get(ann["category_id"])
        if cat is None:
            continue
        cats_by_image[ann["image_id"]].add(cat["name"])
        supercats_by_image[ann["image_id"]].add(cat["supercategory"])

    records: list[CorpusRecord] = []
    for img in sorted(captions_raw["images"], key=lambda i: i["id"]):
        image_id = img["id"]
        captions = [c for c in captions_by_image.get(image_id, []) if c]
        if not captions:
            continue
        records.append(
            CorpusRecord(
                image_id=image_id,
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
                captions=captions,
                categories=sorted(cats_by_image.get(image_id, set())),
                supercategories=sorted(supercats_by_image.get(image_id, set())),
            )
        )

    write_corpus(records, out_path)
    caption_counts = [len(r.captions) for r in records]
    return IngestStats(
        n_images=len(records),
        n_captions=sum(caption_counts),
        n_without_category=sum(1 for r in records if not r.categories),
        min_captions=min(caption_counts),
        max_captions=max(caption_counts),
    )


def make_thumbnails(corpus: Corpus, images_dir: Path, thumbs_dir: Path, size: int) -> int:
    """Sinh thumbnail cạnh dài ``size`` px. Trả về số thumbnail mới tạo lần này."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for record in corpus.records:
        target = thumbs_dir / record.file_name
        if target.exists():
            continue
        with Image.open(images_dir / record.file_name) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((size, size))
            img.save(target, format="JPEG", quality=85)
        made += 1
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tải COCO val2017 và sinh corpus.jsonl")
    parser.add_argument("--skip-download", action="store_true",
                        help="Bỏ qua bước tải, dùng file đã có trong DATA_DIR")
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        images_zip = settings.data_dir / "val2017.zip"
        ann_zip = settings.data_dir / "annotations_trainval2017.zip"
        download_if_missing(COCO_IMAGES_URL, images_zip)
        download_if_missing(COCO_ANNOTATIONS_URL, ann_zip)
        extract_zip(images_zip, settings.data_dir, marker=settings.images_dir)
        extract_zip(ann_zip, settings.data_dir,
                    marker=settings.annotations_dir / CAPTIONS_FILE)

    stats = build_corpus(settings.annotations_dir, settings.corpus_path)
    corpus = load_corpus(settings.corpus_path)
    made = make_thumbnails(corpus, settings.images_dir, settings.thumbs_dir,
                           settings.thumb_size)

    print(f"corpus: {stats.n_images} ảnh, {stats.n_captions} caption "
          f"(mỗi ảnh {stats.min_captions}-{stats.max_captions})")
    print(f"ảnh không có category: {stats.n_without_category}")
    print(f"thumbnail mới sinh: {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
