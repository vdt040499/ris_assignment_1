"""Năm trục ablation → results/*.csv và results/*.md.

Eval truy vấn đúng collection Qdrant mà demo đang dùng, nhưng encode query
theo batch và gọi store trực tiếp thay vì đi qua SearchService: 25 nghìn lần
encode lẻ trên CPU sẽ mất hàng giờ mà không thay đổi kết quả, vì encoder và
phép chuẩn hoá là một.
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
    """Encode query text theo batch, cache ra .npy để lần chạy sau tức thì.

    ``cache_key`` phải phân biệt được bộ query và template, nếu không hai thí
    nghiệm khác nhau sẽ dùng chung một cache và cho ra số giống nhau một cách
    giả tạo. Tên file cache còn bao gồm một hash ngắn của chính nội dung
    ``texts`` (không chỉ ``cache_key`` và số lượng): nếu nội dung câu query đổi
    nhưng số lượng câu giữ nguyên — ví dụ đổi ``RANDOM_SEED`` khi lấy mẫu, hay
    sau này sửa bản dịch trong ``queryset_vi.json`` — cache cũ vẫn khớp đúng
    tên file theo `{cache_key}_{len(texts)}` cũ và bị dùng nhầm cho một bộ
    query khác, cho ra số liệu sai một cách im lặng (không có lỗi nào được
    raise). Hash nội dung khiến nội dung khác luôn sinh tên file khác, nên
    cache cũ tự động không khớp nữa và được sinh lại, thay vì bị dùng nhầm.
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
    """Truy vấn từng vector, trả ``(danh sách ranking image_id, latency mỗi query)``."""
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
    """Recall@k và MRR cho chiều text→ảnh.

    :param queries: danh sách ``(câu query, image_id đúng)``.
    :param prompt_template: chuỗi có ``{}`` để bọc query, hoặc None dùng query thô.
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
    """P@k và mAP@k cho chiều ảnh→ảnh, dùng proxy trùng category.

    Ảnh query luôn bị loại khỏi ``others`` (kết quả dùng để tính P@k/mAP@k)
    bằng một điều kiện lọc tường minh — việc lọc đó luôn đúng theo cấu trúc
    code, nên không cần một con số riêng để "xác nhận" nó. ``self_hits`` đo
    một thứ khác: ảnh query xuất hiện bao nhiêu lần trong ranking THÔ (trước
    khi lọc), tức nó có thật sự là láng giềng gần nhất của chính nó không.
    Giá trị bình thường là đúng bằng ``n_queries`` (mỗi ảnh luôn khớp tuyệt
    đối với chính vector của nó). Thấp hơn gợi ý vector cache (dùng để tạo
    query) không khớp vector đã nạp vào collection (cache/collection lệch
    nhau); cao hơn gợi ý corpus có nhiều point cùng image_id (dữ liệu trùng
    lặp) — cả hai đều là dấu hiệu cần kiểm tra trước khi tin P@k.
    """
    space = get_space(space_name)
    collection = collection_name(space, ctx.settings)
    cache = ctx.settings.cache_dir / f"imgvec_{space_name}.npy"
    if not cache.exists():
        raise FileNotFoundError(
            f"Thiếu {cache}. Chạy: python tasks.py build --space {space_name}"
        )
    all_vectors = np.load(cache)
    row_of = {image_id: row for row, image_id in enumerate(ctx.corpus.image_ids())}
    vectors = np.stack([all_vectors[row_of[i]] for i in image_ids])
    rankings, latencies = search_many(ctx, collection, vectors, k=k + 1)

    query_labels: list[set[str]] = []
    ranked_labels: list[list[set[str]]] = []
    self_hits = 0
    for image_id, ranking in zip(image_ids, rankings):
        # Đếm trên `ranking` THÔ (trước lọc) — cố ý. Đếm trên `others` (sau
        # lọc) sẽ luôn ra 0 cho mọi input, vì `others` được xây dựng ngay bên
        # dưới bằng chính điều kiện loại trừ `other != image_id`: một con số
        # đúng-theo-cấu-trúc-code như vậy không phản ánh gì về dữ liệu thật,
        # chỉ lặp lại một sự thật toán học. Đếm trên `ranking` mới thật sự phụ
        # thuộc dữ liệu (xem docstring của hàm để biết cách đọc con số này).
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
    """P@k trên bộ query ngắn kiểu từ khoá; đúng = ảnh chứa category của query.

    :param cache_key: phải phân biệt từng template. Không dùng ``hash()`` của
        template để sinh khoá: hash của chuỗi trong Python được ngẫu nhiên hoá
        theo từng process, nên tên file cache sẽ đổi mỗi lần chạy — cache không
        bao giờ trúng, và tệ hơn là tên file không tất định.
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
    """So HNSW với exact trên cùng một collection.

    Trả về đúng một dòng cho mỗi giá trị trong ``efs`` (không kèm dòng baseline
    "exact" riêng): baseline exact cho cùng space này đã có sẵn trong bảng trục
    1 (``eval_text2image`` mặc định ``exact=True``), nên lặp lại ở đây chỉ gây
    lệch cột "hnsw_ef" (trộn lẫn số nguyên với chuỗi "exact") mà không thêm
    thông tin mới. Ranking exact vẫn được tính nội bộ — nó là chuẩn để so
    ``overlap@k`` cho từng ``ef``.

    :raises RuntimeError: nếu đang ở chế độ embedded — chế độ đó luôn brute
        force và bỏ qua HNSW, nên mọi con số thu được sẽ là exact đội lốt ANN.
        Cũng raise nếu collection tồn tại nhưng chưa build xong HNSW (xem
        finding C1 của final review): Qdrant chỉ build HNSW khi một segment
        vượt ``indexing_threshold`` mặc định (20000 KB) — 5.000 vector nhỏ
        không bao giờ chạm ngưỡng đó nên có thể "server mode" thật nhưng vẫn
        đang full-scan, một cách lặng lẽ hơn cách embedded-mode gây ra.
    """
    if ctx.settings.qdrant_mode != "server":
        raise RuntimeError(
            "Trục ANN vs exact cần Qdrant server. Đặt QDRANT_MODE=server và chạy "
            "docker compose up -d qdrant"
        )
    space = get_space(space_name)
    collection = collection_name(space, ctx.settings)
    points, indexed = ctx.store.indexing_status(collection)
    if points > 0 and indexed != points:
        raise RuntimeError(
            f"Collection '{collection}' chưa build xong HNSW "
            f"(indexed_vectors_count={indexed}/{points}). Trục ANN vs exact cần "
            "index đã build đầy đủ, nếu không mọi 'ANN' thực chất là full-scan "
            "(exact đội lốt ANN). Chạy update_collection với optimizer_config="
            "OptimizersConfigDiff(indexing_threshold=1) (KHÔNG phải 0 — 0 là "
            "sentinel tắt hẳn indexing) rồi đợi tới khi indexed_vectors_count "
            "== points_count trước khi chạy lại."
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
    """Đo cùng 200 nội dung ở hai ngôn ngữ, cho từng space.

    Cùng một tập nội dung ở cả hai ngôn ngữ là điều kiện để con số so được với
    nhau: chênh lệch đọc ra được là chênh lệch do ngôn ngữ, không do bộ query.
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
    """Điền ô rỗng cho các khoá thiếu, để bảng gộp nhiều loại dòng không mất cột.

    ``eval_text2image`` và ``eval_short_queries`` trả về hai bộ khoá khác nhau
    (Recall/MRR trên caption dài so với Precision trên query ngắn). Nếu
    ``write_table`` lấy cột theo dòng đầu tiên như bình thường, các dòng có
    khoá khác sẽ bị cắt cụt hoặc lệch cột một cách âm thầm. Hàm này hợp
    (union) khoá của mọi dòng theo đúng thứ tự xuất hiện, rồi điền ``""`` vào
    chỗ thiếu, để mọi dòng đều có đủ và đúng thứ tự cột khi ghi bảng.

    :param rows: danh sách dict có thể có tập khoá khác nhau.
    :return: danh sách dict mới, mọi dict có cùng tập khoá (union), giữ
        nguyên thứ tự dòng.
    """
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{key: row.get(key, "") for key in columns} for row in rows]


def write_table(rows: list[dict], name: str, settings: Settings, title: str) -> None:
    """Ghi một bảng ra cả CSV (để tính toán) và Markdown (để dán vào báo cáo).

    :param rows: các dòng dữ liệu; có thể có tập khoá khác nhau (xem
        ``_align_columns``), hoặc rỗng — khi đó vẫn ghi ra file CSV/Markdown
        hợp lệ nhưng không có dữ liệu, thay vì raise lỗi.
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
        lines.append("_không có dòng nào_")
    lines.append("")
    (settings.results_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"results/{name}.csv + .md — {len(rows)} dòng")


AXES = ("model-t2i", "model-i2i", "normalize", "prompt", "ann", "language")

T2I_SPACES = ("clip-b32", "clip-b16", "laion-b32", "siglip-b16", "bm25-cap")
I2I_SPACES = ("clip-b32", "clip-b16", "laion-b32", "siglip-b16", "resnet50")


def sample_caption_queries(corpus: Corpus, size: int, seed: int) -> list[tuple[str, int]]:
    """Lấy mẫu caption làm query text→ảnh, cùng một mẫu cho mọi space.

    Dùng chung một mẫu là điều làm các dòng trong bảng so được với nhau. Corpus
    tra cứu vẫn là toàn bộ 5.000 ảnh, chỉ số lượng query bị giới hạn.
    """
    import random

    pairs = [(text, image_id) for image_id, _, text in corpus.caption_pairs()]
    if size >= len(pairs):
        return pairs
    return random.Random(seed).sample(pairs, size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy các trục ablation")
    parser.add_argument("--axis", default="all",
                        help=f"trục cần chạy hoặc 'all'. Hợp lệ: {', '.join(AXES)}")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="bằng với --axis all; có để đúng lệnh ghi trong spec §11")
    parser.add_argument("--sample", type=int, default=T2I_SAMPLE_DEFAULT,
                        help="số query text→ảnh (0 = dùng toàn bộ caption)")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    settings = get_settings()
    corpus = load_corpus(settings.corpus_path)
    store = VectorStore(settings)
    if store.health() != "ok":
        print("Không kết nối được Qdrant. Chạy: docker compose up -d qdrant")
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
                    title=f"Trục 1 — text→ảnh ({len(queries)} query, 5.000 ảnh)")
    if "model-i2i" in wanted:
        write_table([eval_image2image(ctx, s, image_ids, k=args.k) for s in I2I_SPACES],
                    "axis1_model_i2i", settings,
                    title=f"Trục 1 — ảnh→ảnh ({len(image_ids)} ảnh query, proxy category)")
    if "normalize" in wanted:
        write_table([eval_text2image(ctx, s, queries) for s in ("clip-b32", "clip-b32-raw")],
                    "axis2_normalize", settings,
                    title="Trục 2 — cosine trên vector normalize vs dot trên vector thô")
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
                    title="Trục 3 — prompt template trên caption dài và trên query ngắn")
    if "ann" in wanted:
        write_table(eval_ann_sweep(ctx, "clip-b32", queries, k=args.k),
                    "axis4_ann", settings,
                    title=f"Trục 4 — HNSW vs exact trên 5.000 vector (k={args.k})")
    if "language" in wanted:
        write_table(eval_language(ctx, ["clip-b32", "mclip-b32"]),
                    "axis5_language", settings,
                    title="Trục 5 — cùng 200 nội dung, tiếng Anh vs tiếng Việt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
