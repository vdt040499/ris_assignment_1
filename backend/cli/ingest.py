"""Download COCO val2017 and generate corpus.jsonl + thumbnails.

Run once. Every step is resumable: a file already downloaded is not
downloaded again, a thumbnail already generated is not regenerated.
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
    """Figures counted from the real data, so the report doesn't have to assume anything."""

    n_images: int
    n_captions: int
    n_without_category: int
    min_captions: int
    max_captions: int


def download_if_missing(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest`` if not already present. Returns True if a download actually happened this time.

    Writes to a ``.part`` file before renaming it, so an interrupted download
    doesn't leave behind a corrupt file that looks complete. A downloaded zip
    is checked for integrity.
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
                raise RuntimeError(f"Downloaded zip is corrupt: {url}")
    part.rename(dest)
    return True


def extract_zip(zip_path: Path, dest_dir: Path, marker: Path) -> bool:
    """Extract if ``marker`` doesn't exist yet. Returns True if extraction actually happened this time."""
    if marker.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return True


def build_corpus(annotations_dir: Path, out_path: Path) -> IngestStats:
    """Merge COCO captions and categories into corpus.jsonl.

    Images with no category annotation are still kept, with an empty array —
    dropping them would skew the denominator of every metric downstream.
    Images with no captions at all are excluded, since they can't be used for
    either evaluation or the BM25 baseline.
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
    """Generate thumbnails with long edge ``size`` px. Returns the number of thumbnails newly created this time."""
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
    parser = argparse.ArgumentParser(description="Download COCO val2017 and generate corpus.jsonl")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the download step, use files already in DATA_DIR")
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

    print(f"corpus: {stats.n_images} images, {stats.n_captions} captions "
          f"({stats.min_captions}-{stats.max_captions} per image)")
    print(f"images without category: {stats.n_without_category}")
    print(f"newly generated thumbnails: {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
