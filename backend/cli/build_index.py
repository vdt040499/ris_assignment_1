"""Encode the corpus and load vectors into Qdrant, one collection per space.

Idempotent: a space that has already been built is skipped, unless --force is
given. Encoded vectors are cached to .npy, so rebuilding a collection after
changing the Qdrant configuration doesn't require re-running the model.
"""

import argparse
import json
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from app.config import Settings, get_settings
from app.corpus import (
    Corpus,
    caption_payload,
    caption_point_id,
    load_corpus,
    record_payload,
)
from app.encoders import build_encoder
from app.encoders.bm25 import Bm25Retriever
from app.registry import (
    BACKEND_BM25,
    SPACES,
    SpaceSpec,
    collection_name,
    get_space,
)
from app.vecutil import l2_normalize
from app.vectordb import VectorStore

PAYLOAD_INDEX_FIELDS = ("categories", "supercategories")

#: Space that owns the caption collection. Change this value if you want the
#: image→text direction to run on a different model.
CAPTION_SPACE = "clip-b32"

#: The space on the left derives its vectors from the space on the right and
#: then normalizes them itself. Both sides share the same hf_id, so the
#: vectors are bit-for-bit identical — exactly the condition needed for the
#: normalize ablation: only one variable changes.
DERIVED_FROM = {"clip-b32": "clip-b32-raw"}

#: Build order for `--space all`. clip-b32-raw comes before clip-b32 so a
#: single encode pass can serve both.
BUILD_ORDER = (
    "clip-b32-raw",
    "clip-b32",
    "laion-b32",
    "clip-b16",
    "siglip-b16",
    "resnet50",
    "bm25-cap",
)


def git_commit() -> str:
    """Current commit, so each index_meta can be traced back to the exact code that generated it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def encode_all_images(
    encoder, corpus: Corpus, images_dir: Path, batch_size: int
) -> np.ndarray:
    """Encode the entire corpus of images in batches, printing progress to stdout."""
    chunks: list[np.ndarray] = []
    total = len(corpus)
    for start in range(0, total, batch_size):
        records = corpus.records[start : start + batch_size]
        images = [_load_image(images_dir / r.file_name) for r in records]
        chunks.append(encoder.encode_images(images))
        print(f"  encoding images {min(start + batch_size, total)}/{total}", end="\r")
    print()
    return np.concatenate(chunks, axis=0)


def image_vectors(
    space: SpaceSpec,
    settings: Settings,
    corpus: Corpus,
    encoder_factory: Callable[..., object] = build_encoder,
) -> np.ndarray:
    """Image vectors for a space, preferring the cache before falling back to encoding.

    Order tried: the space's own cache → derived from the source space's cache
    in ``DERIVED_FROM`` → actual encoding.
    """
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = settings.cache_dir / f"imgvec_{space.name}.npy"
    if cache.exists():
        return np.load(cache)

    source_name = DERIVED_FROM.get(space.name)
    if source_name:
        source_cache = settings.cache_dir / f"imgvec_{source_name}.npy"
        if source_cache.exists():
            vectors = l2_normalize(np.load(source_cache))
            np.save(cache, vectors)
            return vectors

    encoder = encoder_factory(space.name, settings)
    vectors = encode_all_images(encoder, corpus, settings.images_dir, settings.batch_size)
    np.save(cache, vectors)
    return vectors


def build_caption_collection(
    settings: Settings,
    store: VectorStore,
    corpus: Corpus,
    encoder_factory: Callable[..., object] = build_encoder,
    force: bool = False,
) -> int:
    """Load every caption into the caption collection. Returns the number of points loaded (0 if skipped)."""
    space = get_space(CAPTION_SPACE)
    name = collection_name(space, settings, target="caption")
    if store.count(name) > 0 and not force:
        return 0

    pairs = corpus.caption_pairs()
    encoder = encoder_factory(space.name, settings)
    vectors = encoder.encode_texts([text for _, _, text in pairs])
    ids = [caption_point_id(image_id, idx) for image_id, idx, _ in pairs]
    payloads = [
        caption_payload(corpus.by_image_id[image_id], idx) for image_id, idx, _ in pairs
    ]
    # Use the vector's actual dimension (not space.dim as declared in the registry):
    # correct in production since the two values always match, and independent of
    # the registry when a test injects a fake encoder with a different dimension
    # to run fast without needing the model.
    store.ensure_collection(name, dim=vectors.shape[1], distance=space.distance,
                            recreate=force)
    store.upsert(name, ids=ids, vectors=vectors, payloads=payloads,
                 batch_size=settings.batch_size)
    store.create_payload_indexes(name, PAYLOAD_INDEX_FIELDS)
    return len(ids)


def build_space(
    space_name: str,
    settings: Settings,
    store: VectorStore,
    corpus: Corpus,
    force: bool = False,
    encoder_factory: Callable[..., object] = build_encoder,
) -> dict | None:
    """Build the index for a space.

    :return: the ``index_meta`` dict written to disk, or ``None`` if skipped
        (the space reuses another space's collection, or the index already
        exists and ``force`` was not given).
    """
    space = get_space(space_name)
    if not space.builds_index:
        print(f"{space_name}: skipped — reuses another space's collection ({space.note})")
        return None

    started = perf_counter()
    if space.backend == BACKEND_BM25:
        if settings.bm25_path.exists() and not force:
            print(f"{space_name}: index already exists, skipping")
            return None
        Bm25Retriever.build(corpus).save(settings.bm25_path)
        n_points = len(corpus)
    else:
        name = collection_name(space, settings, target="image")
        if store.count(name) > 0 and not force:
            print(f"{space_name}: already has {store.count(name)} points, skipping")
            return None
        vectors = image_vectors(space, settings, corpus, encoder_factory)
        # See the note in build_caption_collection: use the vector's actual
        # dimension, not space.dim as declared in the registry.
        store.ensure_collection(name, dim=vectors.shape[1], distance=space.distance,
                                recreate=force)
        store.upsert(name, ids=corpus.image_ids(), vectors=vectors,
                     payloads=[record_payload(r) for r in corpus.records],
                     batch_size=settings.batch_size)
        store.create_payload_indexes(name, PAYLOAD_INDEX_FIELDS)
        n_points = len(corpus)

    meta = {
        "space": space.name,
        "hf_id": space.hf_id,
        "dim": space.dim,
        "distance": space.distance,
        "normalized": space.normalized,
        "backend": space.backend,
        "n_points": n_points,
        "encode_seconds": round(perf_counter() - started, 2),
        "batch_size": settings.batch_size,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    settings.index_meta_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_meta_dir / f"{space.name}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{space_name}: {n_points} points in {meta['encode_seconds']}s")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the vector index for each space")
    parser.add_argument("--space", default="all",
                        help=f"space name or 'all'. Valid: {', '.join(SPACES)}")
    parser.add_argument("--force", action="store_true", help="rebuild even if the index already exists")
    parser.add_argument("--skip-captions", action="store_true",
                        help="don't build the caption collection")
    args = parser.parse_args(argv)

    settings = get_settings()
    corpus = load_corpus(settings.corpus_path)
    store = VectorStore(settings)
    if store.health() != "ok":
        print("Could not connect to Qdrant. Run: docker compose up -d qdrant")
        return 1

    names = BUILD_ORDER if args.space == "all" else (args.space,)
    for name in names:
        build_space(name, settings, store, corpus, force=args.force)

    if not args.skip_captions and args.space in ("all", CAPTION_SPACE):
        added = build_caption_collection(settings, store, corpus, force=args.force)
        print(f"caption collection: {added} new points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
