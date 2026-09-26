"""Five ablation axes → results/*.csv and results/*.md.

Evaluation queries the exact Qdrant collection the demo uses, but encodes
queries in batches and calls the store directly instead of going through
SearchService: 25 thousand individual encode calls on CPU would take hours
without changing the result, since the encoder and the normalization are
the same either way.
"""

import argparse
import csv
import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from app.config import Settings, get_settings
from app.corpus import Corpus, load_corpus
from app.encoders import build_encoder
from app.encoders.bm25 import Bm25Retriever
from app.metrics import map_at_k, mrr_at_k, overlap_at_k, precision_at_k, recall_at_k
from app.registry import BACKEND_BM25, collection_name, get_space
from app.vectordb import VectorStore

T2I_SAMPLE_DEFAULT = 5000
MRR_K = 10
PROMPT_TEMPLATES: tuple[str | None, ...] = (
    None,
    "a photo of {}",
    "a photo of {}, a type of scene",
)
ANN_EF_VALUES: tuple[int, ...] = (16, 64, 128, 256)


@dataclass
class EvalContext:
    settings: Settings
    store: VectorStore
    corpus: Corpus
    encoder_factory: Callable[..., object] = build_encoder
    _bm25: Bm25Retriever | None = field(default=None, init=False, repr=False)

    def bm25(self) -> Bm25Retriever:
        if self._bm25 is None:
            self._bm25 = Bm25Retriever.load(self.settings.bm25_path, self.corpus)
        return self._bm25


def text_vectors(
    ctx: EvalContext, space_name: str, texts: Sequence[str], cache_key: str
) -> np.ndarray:
    """Encode query text in batches, caching to .npy so later runs are instant.

    ``cache_key`` must distinguish the query set and template, otherwise two
    different experiments would share the same cache and produce identical
    numbers artificially. The cache filename also includes a short hash of
    the ``texts`` content itself (not just ``cache_key`` and the count): if
    the query text content changes but the count stays the same — e.g.
    changing ``RANDOM_SEED`` during sampling, or later fixing a translation
    in ``queryset_vi.json`` — the old cache would still match the old
    `{cache_key}_{len(texts)}` filename and get reused for a different query
    set by mistake, silently producing wrong numbers (no error would be
    raised). Hashing the content makes different content always produce a
    different filename, so the old cache automatically stops matching and
    gets regenerated instead of being reused by mistake.
    """
    ctx.settings.cache_dir.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    cache = (
        ctx.settings.cache_dir
        / f"txt_{space_name}_{cache_key}_{len(texts)}_{content_hash}.npy"
    )
    if cache.exists():
        return np.load(cache)
    vectors = ctx.encoder_factory(space_name, ctx.settings).encode_texts(list(texts))
    np.save(cache, vectors)
    return vectors


def search_many(
    ctx: EvalContext,
    collection: str,
    vectors: np.ndarray,
    k: int,
    exact: bool = True,
    hnsw_ef: int | None = None,
) -> tuple[list[list[int]], list[float]]:
    """Query each vector, returning ``(list of image_id rankings, latency per query)``."""
    rankings: list[list[int]] = []
    latencies: list[float] = []
    for vector in vectors:
        started = perf_counter()
        hits = ctx.store.search(collection, vector, k=k, exact=exact, hnsw_ef=hnsw_ef)
        latencies.append((perf_counter() - started) * 1000)
        rankings.append([int(h.payload["image_id"]) for h in hits])
    return rankings, latencies


def _percentiles(latencies: Sequence[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return round(p50, 3), round(p95, 3)


def eval_text2image(
    ctx: EvalContext,
    space_name: str,
    queries: list[tuple[str, int]],
    ks: Sequence[int] = (1, 5, 10),
    exact: bool = True,
    hnsw_ef: int | None = None,
    cache_key: str = "captions",
    prompt_template: str | None = None,
) -> dict:
    """Recall@k and MRR for the text→image direction.

    :param queries: list of ``(query text, correct image_id)``.
    :param prompt_template: a string with ``{}`` to wrap the query, or None to use the raw query.
    """
    space = get_space(space_name)
    texts = [
        prompt_template.format(text) if prompt_template else text for text, _ in queries
    ]
    gold = [image_id for _, image_id in queries]
    top_k = max(max(ks), MRR_K)

    if space.backend == BACKEND_BM25:
        retriever = ctx.bm25()
        rankings, latencies = [], []
        for text in texts:
            started = perf_counter()
            hits = retriever.search(text, top_k)
            latencies.append((perf_counter() - started) * 1000)
            rankings.append([h.id for h in hits])
    else:
        vectors = text_vectors(ctx, space_name, texts, cache_key)
        rankings, latencies = search_many(
            ctx,
            collection_name(space, ctx.settings),
            vectors,
            k=top_k,
            exact=exact,
            hnsw_ef=hnsw_ef,
        )

    recalls = recall_at_k(rankings, gold, ks)
    p50, p95 = _percentiles(latencies)
    row = {
        "space": space_name,
        "n_queries": len(queries),
        "exact": exact,
        "hnsw_ef": hnsw_ef,
        "prompt_template": prompt_template or "",
    }
    row.update({f"R@{k}": round(recalls[k], 4) for k in ks})
    row["MRR@10"] = round(mrr_at_k(rankings, gold, MRR_K), 4)
    row["search_ms_p50"] = p50
    row["search_ms_p95"] = p95
    return row


def eval_image2image(
    ctx: EvalContext, space_name: str, image_ids: list[int], k: int = 10
) -> dict:
    """P@k and mAP@k for the image→image direction, using category overlap as a proxy.

    The query image is always excluded from ``others`` (the results used to
    compute P@k/mAP@k) via an explicit filter condition — that filtering is
    always correct by construction, so there's no need for a separate number
    to "confirm" it. ``self_hits`` measures something different: how many
    times the query image appears in the RAW ranking (before filtering),
    i.e. whether it's actually its own nearest neighbor. The normal value
    equals ``n_queries`` exactly (each image always matches its own vector
    perfectly). A lower value suggests the cached vector (used to build the
    query) doesn't match the vector loaded into the collection (cache and
    collection are out of sync); a higher value suggests the corpus has
    multiple points sharing the same image_id (duplicate data) — both are
    signs to check before trusting P@k.
    """
    space = get_space(space_name)
    collection = collection_name(space, ctx.settings)
    cache = ctx.settings.cache_dir / f"imgvec_{space_name}.npy"
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing {cache}. Run: python tasks.py build --space {space_name}"
        )
    all_vectors = np.load(cache)
    row_of = {image_id: row for row, image_id in enumerate(ctx.corpus.image_ids())}
    vectors = np.stack([all_vectors[row_of[i]] for i in image_ids])
    rankings, latencies = search_many(ctx, collection, vectors, k=k + 1)

    query_labels: list[set[str]] = []
    ranked_labels: list[list[set[str]]] = []
    self_hits = 0
    for image_id, ranking in zip(image_ids, rankings):
        # Counting on the RAW `ranking` (before filtering) — intentional. Counting
        # on `others` (after filtering) would always be 0 for any input, since
        # `others` is built right below using the exact exclusion condition
        # `other != image_id`: such a correct-by-construction number wouldn't
        # reflect anything about the real data, just restate a mathematical
        # fact. Counting on `ranking` actually depends on the data (see the
        # function's docstring for how to read this number).
        self_hits += sum(1 for other in ranking if other == image_id)
        others = [other for other in ranking if other != image_id][:k]
        query_labels.append(set(ctx.corpus.by_image_id[image_id].categories))
        ranked_labels.append(
            [set(ctx.corpus.by_image_id[other].categories) for other in others]
        )

    p50, p95 = _percentiles(latencies)
    return {
        "space": space_name,
        "n_queries": len(image_ids),
        f"P@{k}": round(precision_at_k(ranked_labels, query_labels, k), 4),
        f"mAP@{k}": round(map_at_k(ranked_labels, query_labels, k), 4),
        "self_hits": self_hits,
        "search_ms_p50": p50,
        "search_ms_p95": p95,
    }


def eval_short_queries(
    ctx: EvalContext,
    space_name: str,
    k: int = 10,
    prompt_template: str | None = None,
    cache_key: str = "short_raw",
) -> dict:
    """P@k on the short keyword-style query set; correct = image contains the query's category.

    :param cache_key: must distinguish each template. Don't use ``hash()`` of
        the template to generate the key: Python's string hash is randomized
        per process, so the cache filename would change every run — the cache
        would never hit, and worse, the filename wouldn't be deterministic.
    """
    entries = json.loads(
        (ctx.settings.data_dir / "queryset_short.json").read_text(encoding="utf-8")
    )
    space = get_space(space_name)
    texts = [
        prompt_template.format(e["query"]) if prompt_template else e["query"]
        for e in entries
    ]
    vectors = text_vectors(ctx, space_name, texts, cache_key)
    rankings, latencies = search_many(
        ctx, collection_name(space, ctx.settings), vectors, k=k
    )
    query_labels = [{e["gold_category"]} for e in entries]
    ranked_labels = [
        [set(ctx.corpus.by_image_id[i].categories) for i in ranking]
        for ranking in rankings
    ]
    p50, _ = _percentiles(latencies)
    return {
        "space": space_name,
        "prompt_template": prompt_template or "",
        "n_queries": len(entries),
        f"P@{k}": round(precision_at_k(ranked_labels, query_labels, k), 4),
        "search_ms_p50": p50,
    }


def eval_ann_sweep(
    ctx: EvalContext,
    space_name: str,
    queries: list[tuple[str, int]],
    efs: Sequence[int] = ANN_EF_VALUES,
    k: int = 10,
) -> list[dict]:
    """Compare HNSW against exact search on the same collection.

    Returns exactly one row per value in ``efs`` (with no separate "exact"
    baseline row): the exact baseline for this same space already exists in
    the axis 1 table (``eval_text2image`` defaults to ``exact=True``), so
    repeating it here would only skew the "hnsw_ef" column (mixing integers
    with the string "exact") without adding new information. The exact
    ranking is still computed internally — it's the reference used to
    compute ``overlap@k`` for each ``ef``.

    :raises RuntimeError: if running in embedded mode — that mode always does
        brute force and ignores HNSW, so any numbers obtained would be exact
        search disguised as ANN. Also raises if the collection exists but
        hasn't finished building HNSW (see finding C1 of the final review):
        Qdrant only builds HNSW once a segment exceeds the default
        ``indexing_threshold`` (20000 KB) — 5,000 small vectors never reach
        that threshold, so it's possible to be in genuine "server mode" while
        still doing a full scan, a more silent failure mode than the one
        embedded mode causes.
    """
    if ctx.settings.qdrant_mode != "server":
        raise RuntimeError(
            "The ANN vs exact axis requires the Qdrant server. Set QDRANT_MODE=server "
            "and run docker compose up -d qdrant"
        )
    space = get_space(space_name)
    collection = collection_name(space, ctx.settings)
    points, indexed = ctx.store.indexing_status(collection)
    if points > 0 and indexed != points:
        raise RuntimeError(
            f"Collection '{collection}' hasn't finished building HNSW "
            f"(indexed_vectors_count={indexed}/{points}). The ANN vs exact axis "
            "needs a fully built index, otherwise every 'ANN' result is actually "
            "a full scan (exact search disguised as ANN). Run update_collection "
            "with optimizer_config=OptimizersConfigDiff(indexing_threshold=1) "
            "(NOT 0 — 0 is the sentinel that disables indexing entirely) and wait "
            "until indexed_vectors_count == points_count before running again."
        )
    texts = [text for text, _ in queries]
    gold = [image_id for _, image_id in queries]
    vectors = text_vectors(ctx, space_name, texts, "captions")

    exact_rankings, _exact_latencies = search_many(ctx, collection, vectors, k=k)
    rows: list[dict] = []
    for ef in efs:
        rankings, latencies = search_many(
            ctx, collection, vectors, k=k, exact=False, hnsw_ef=ef
        )
        p50, p95 = _percentiles(latencies)
        rows.append(
            {
                "space": space_name,
                "hnsw_ef": ef,
                f"overlap@{k}": round(overlap_at_k(rankings, exact_rankings, k), 4),
                "R@1": round(recall_at_k(rankings, gold, (1,))[1], 4),
                "search_ms_p50": p50,
                "search_ms_p95": p95,
            }
        )
    return rows


def eval_language(
    ctx: EvalContext, space_names: list[str], k: Sequence[int] = (1, 5, 10)
) -> list[dict]:
    """Measure the same 200 items in both languages, for each space.

    Using the same set of items in both languages is the condition for the
    numbers to be comparable: the difference observed is due to language,
    not due to the query set.
    """
    entries = json.loads(
        (ctx.settings.data_dir / "queryset_vi.json").read_text(encoding="utf-8")
    )
    rows: list[dict] = []
    for space_name in space_names:
        for language in ("en", "vi"):
            queries = [(e[language], e["image_id"]) for e in entries]
            row = eval_text2image(
                ctx, space_name, queries, ks=k, cache_key=f"lang_{language}"
            )
            row["language"] = language
            rows.append(row)
    return rows


def _align_columns(rows: list[dict]) -> list[dict]:
    """Fill empty cells for missing keys, so a table combining multiple row types doesn't lose columns.

    ``eval_text2image`` and ``eval_short_queries`` return two different sets
    of keys (Recall/MRR on long captions vs. Precision on short queries). If
    ``write_table`` took its columns from the first row as usual, rows with
    different keys would get silently truncated or misaligned. This function
    takes the union of every row's keys in order of appearance, then fills
    ``""`` into the gaps, so every row has the full, correctly ordered set of
    columns when the table is written.

    :param rows: list of dicts that may have different sets of keys.
    :return: a new list of dicts, all sharing the same set of keys (the
        union), preserving row order.
    """
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{key: row.get(key, "") for key in columns} for row in rows]


def write_table(rows: list[dict], name: str, settings: Settings, title: str) -> None:
    """Write a table out to both CSV (for computation) and Markdown (to paste into the report).

    :param rows: data rows; may have different sets of keys (see
        ``_align_columns``), or be empty — in that case a valid but empty
        CSV/Markdown file is still written, instead of raising an error.
    """
    rows = _align_columns(rows)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    csv_path = settings.results_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"### {title}", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    else:
        lines.append("_no rows_")
    lines.append("")
    (settings.results_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"results/{name}.csv + .md — {len(rows)} rows")


AXES = ("model-t2i", "model-i2i", "normalize", "prompt", "ann", "language")

T2I_SPACES = ("clip-b32", "clip-b16", "laion-b32", "siglip-b16", "bm25-cap")
I2I_SPACES = ("clip-b32", "clip-b16", "laion-b32", "siglip-b16", "resnet50")


def sample_caption_queries(corpus: Corpus, size: int, seed: int) -> list[tuple[str, int]]:
    """Sample captions to use as text→image queries, the same sample for every space.

    Using one shared sample is what makes the rows in the table comparable.
    The corpus being searched is still all 5,000 images; only the number of
    queries is limited.
    """
    import random

    pairs = [(text, image_id) for image_id, _, text in corpus.caption_pairs()]
    if size >= len(pairs):
        return pairs
    return random.Random(seed).sample(pairs, size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ablation axes")
    parser.add_argument("--axis", default="all",
                        help=f"axis to run, or 'all'. Valid: {', '.join(AXES)}")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="equivalent to --axis all; provided to match the command in spec §11")
    parser.add_argument("--sample", type=int, default=T2I_SAMPLE_DEFAULT,
                        help="number of text→image queries (0 = use all captions)")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    settings = get_settings()
    corpus = load_corpus(settings.corpus_path)
    store = VectorStore(settings)
    if store.health() != "ok":
        print("Could not connect to Qdrant. Run: docker compose up -d qdrant")
        return 1
    ctx = EvalContext(settings, store, corpus)

    size = args.sample or len(corpus.caption_pairs())
    queries = sample_caption_queries(corpus, size, settings.random_seed)
    image_ids = json.loads(
        (settings.data_dir / "queryset_i2i.json").read_text(encoding="utf-8")
    )
    wanted = AXES if (args.run_all or args.axis == "all") else (args.axis,)

    if "model-t2i" in wanted:
        write_table([eval_text2image(ctx, s, queries) for s in T2I_SPACES],
                    "axis1_model_t2i", settings,
                    title=f"Axis 1 — text→image ({len(queries)} queries, 5,000 images)")
    if "model-i2i" in wanted:
        write_table([eval_image2image(ctx, s, image_ids, k=args.k) for s in I2I_SPACES],
                    "axis1_model_i2i", settings,
                    title=f"Axis 1 — image→image ({len(image_ids)} query images, category proxy)")
    if "normalize" in wanted:
        write_table([eval_text2image(ctx, s, queries) for s in ("clip-b32", "clip-b32-raw")],
                    "axis2_normalize", settings,
                    title="Axis 2 — cosine on normalized vectors vs dot product on raw vectors")
    if "prompt" in wanted:
        rows = []
        for index, template in enumerate(PROMPT_TEMPLATES):
            rows.append(eval_text2image(ctx, "clip-b32", queries,
                                        cache_key=f"tmpl_{index}",
                                        prompt_template=template))
            rows.append(eval_short_queries(ctx, "clip-b32", k=args.k,
                                           prompt_template=template,
                                           cache_key=f"short_{index}"))
        write_table(rows, "axis3_prompt", settings,
                    title="Axis 3 — prompt template on long captions and on short queries")
    if "ann" in wanted:
        write_table(eval_ann_sweep(ctx, "clip-b32", queries, k=args.k),
                    "axis4_ann", settings,
                    title=f"Axis 4 — HNSW vs exact on 5,000 vectors (k={args.k})")
    if "language" in wanted:
        write_table(eval_language(ctx, ["clip-b32", "mclip-b32"]),
                    "axis5_language", settings,
                    title="Axis 5 — same 200 items, English vs Vietnamese")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
