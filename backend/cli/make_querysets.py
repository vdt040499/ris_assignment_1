"""Generate the query sets used for evaluation. Every sampling step uses a fixed seed.

The three generated files are committed to the repo, because if the query set
can't be reproduced, the numbers can't be reproduced either.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from app.config import Settings, get_settings
from app.corpus import Corpus, load_corpus

SHORT_QUERY_TEMPLATE = "a {}"


def sample_vi_queries(corpus: Corpus, size: int, seed: int) -> list[dict]:
    """Sample captions for manual translation into Vietnamese.

    Args:
        corpus: The loaded corpus, providing all ``(image_id,
            caption_index, text)`` triples to sample from.
        size: Number of captions to sample; clamped to the actual number of
            caption pairs available if ``size`` is larger.
        seed: Seed for a local random number generator (``random.Random``,
            not the global ``random`` module) so sampling is deterministic
            and reproducible across runs.

    Returns:
        A list of dicts ``{image_id, caption_index, en, vi}``, where ``vi``
        is always empty — the manual translation step fills it in later.
        Sampling is done over the set of ``(image_id, caption_index)``
        pairs without replacement, so no caption is duplicated.
    """
    pairs = corpus.caption_pairs()
    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(size, len(pairs)))
    return [
        {"image_id": image_id, "caption_index": index, "en": text, "vi": ""}
        for image_id, index, text in chosen
    ]


def sample_i2i_queries(corpus: Corpus, size: int, seed: int) -> list[int]:
    """Sample image_ids to use as queries for the image→image direction.

    Args:
        corpus: The loaded corpus, providing the list of ``image_ids()``.
        size: Number of images to sample; clamped to the actual number of
            images in the corpus if ``size`` is larger.
        seed: Seed for a local ``random.Random`` for deterministic sampling.

    Returns:
        A list of unique ``image_id`` values (no duplicates), sorted ascending.
    """
    rng = random.Random(seed)
    return sorted(rng.sample(corpus.image_ids(), min(size, len(corpus))))


def build_short_queries(corpus: Corpus) -> list[dict]:
    """One short query for each category that actually appears in the corpus.

    Args:
        corpus: The loaded corpus; each ``CorpusRecord`` carries that image's
            list of ``categories``.

    Returns:
        A list of dicts ``{query, gold_category, n_gold}``, sorted by category
        name. ``n_gold`` is the number of images containing that category —
        the report needs this number so readers know the denominator of P@k
        for each query. A category with zero images (which can't actually
        happen, since Counter only counts categories that are present) is
        explicitly filtered out as a safety measure.
    """
    counts = Counter(
        category for record in corpus.records for category in record.categories
    )
    return [
        {
            "query": SHORT_QUERY_TEMPLATE.format(category),
            "gold_category": category,
            "n_gold": count,
        }
        for category, count in sorted(counts.items())
        if count > 0
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: generates three query set files into ``settings.data_dir``.

    Args:
        argv: List of command-line arguments (excluding the program name);
            uses ``sys.argv[1:]`` when left as ``None``.

    Returns:
        The process exit code (always ``0``).
    """
    parser = argparse.ArgumentParser(description="Generate query sets for evaluation")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files (loses the Vietnamese translation!)")
    args = parser.parse_args(argv)

    settings: Settings = get_settings()
    corpus = load_corpus(settings.corpus_path)

    targets = {
        "queryset_vi.json": sample_vi_queries(
            corpus, settings.queryset_size, settings.random_seed
        ),
        "queryset_i2i.json": sample_i2i_queries(
            corpus, settings.queryset_size, settings.random_seed
        ),
        "queryset_short.json": build_short_queries(corpus),
    }
    for filename, payload in targets.items():
        path = settings.data_dir / filename
        if path.exists() and not args.force:
            print(f"{filename}: already exists, skipping (use --force to overwrite)")
            continue
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{filename}: {len(payload)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
