"""Metric retrieval. Hàm thuần, không biết gì về Qdrant hay model.

Quy ước chung: ``rankings[i]`` là danh sách image_id đã xếp hạng cho query
thứ ``i``, phần tử đầu là kết quả hạng 1.
"""

from collections.abc import Sequence


def _check_lengths(rankings: Sequence, gold: Sequence) -> None:
    if len(rankings) != len(gold):
        raise ValueError(
            f"số ranking ({len(rankings)}) khác số nhãn đúng ({len(gold)})"
        )


def recall_at_k(
    rankings: Sequence[Sequence[int]], gold: Sequence[int], ks: Sequence[int]
) -> dict[int, float]:
    """Recall@k cho bài toán một-nhãn-đúng (mỗi caption có đúng một ảnh gốc).

    :return: dict ``{k: tỉ lệ query có ảnh gốc nằm trong top-k}``.
    :raises ValueError: nếu số ranking khác số nhãn đúng.
    """
    _check_lengths(rankings, gold)
    result: dict[int, float] = {}
    total = len(gold) or 1
    for k in ks:
        hits = sum(1 for ranked, want in zip(rankings, gold) if want in ranked[:k])
        result[k] = hits / total
    return result


def mrr_at_k(rankings: Sequence[Sequence[int]], gold: Sequence[int], k: int) -> float:
    """Mean Reciprocal Rank across all queries, ranked within top-k.

    Each query contributes 1/position of its first hit (or 0 if no hit within top-k).
    Queries with no hit contribute 0 to the sum but are included in the denominator
    (total number of queries), so MRR reflects performance across the entire query set.
    Excluding misses would let a system with few but perfect hits appear artificially
    strong; fixed denominator penalizes incomplete recall.

    :param rankings: list of ranked item sequences, one per query.
    :param gold: list of ground-truth items, one per query.
    :param k: cutoff rank; only consider top-k items from rankings.
    :return: mean reciprocal rank (float in [0, 1]).
    :raises ValueError: if len(rankings) != len(gold).
    """
    _check_lengths(rankings, gold)
    total = 0.0
    for ranked, want in zip(rankings, gold):
        for position, item in enumerate(ranked[:k], start=1):
            if item == want:
                total += 1.0 / position
                break
    return total / (len(gold) or 1)


def precision_at_k(
    ranked_labels: Sequence[Sequence[set[str]]],
    query_labels: Sequence[set[str]],
    k: int,
) -> float:
    """Precision@k for image-to-image search using label overlap as relevance proxy.

    A result is considered relevant if it shares at least one label with the query.
    This is a proxy heuristic, not ground truth; it biases toward images with many
    objects. Queries with no labels contribute 0; rank lists shorter than k are
    handled gracefully by not contributing to that query's precision.

    :param ranked_labels: list of label sets for ranked results, one list per query.
    :param query_labels: list of label sets for queries, one set per query.
    :param k: cutoff rank; only consider top-k results for precision calculation.
    :return: mean precision@k across all queries (float in [0, 1]).
    :raises ValueError: if len(ranked_labels) != len(query_labels).
    """
    _check_lengths(ranked_labels, query_labels)
    total = 0.0
    for ranked, want in zip(ranked_labels, query_labels):
        top = ranked[:k]
        if not top:
            continue
        relevant = sum(1 for labels in top if want and labels & want)
        total += relevant / len(top)
    return total / (len(query_labels) or 1)


def map_at_k(
    ranked_labels: Sequence[Sequence[set[str]]],
    query_labels: Sequence[set[str]],
    k: int,
) -> float:
    """Mean Average Precision@k using label overlap as relevance proxy.

    A result is relevant if it shares at least one label with the query.
    Average Precision@k is normalized by the count of relevant items found
    in top-k (not by corpus size), since corpus-wide relevance counts are
    meaningless for label proxies where thousands of images share 'person'.

    :param ranked_labels: list of label sets for ranked results, one list per query.
    :param query_labels: list of label sets for queries, one set per query.
    :param k: cutoff rank; compute AP over top-k results.
    :return: mean average precision@k across all queries (float in [0, 1]).
    :raises ValueError: if len(ranked_labels) != len(query_labels).
    """
    _check_lengths(ranked_labels, query_labels)
    total = 0.0
    for ranked, want in zip(ranked_labels, query_labels):
        hits = 0
        precision_sum = 0.0
        for position, labels in enumerate(ranked[:k], start=1):
            if want and labels & want:
                hits += 1
                precision_sum += hits / position
        if hits:
            total += precision_sum / hits
    return total / (len(query_labels) or 1)


def overlap_at_k(
    rankings_a: Sequence[Sequence[int]], rankings_b: Sequence[Sequence[int]], k: int
) -> float:
    """Overlap ratio between two top-k rankings, averaged across queries.

    Measures agreement between ranking systems (e.g., ANN approximation vs.
    exact search). Overlap is the size of intersection divided by the maximum
    of the two set sizes (Tanimoto-like similarity).

    :param rankings_a: first set of ranked item sequences, one per query.
    :param rankings_b: second set of ranked item sequences, one per query (often ground truth).
    :param k: cutoff rank; only consider top-k items from each ranking.
    :return: mean overlap ratio across all queries (float in [0, 1]).
    :raises ValueError: if len(rankings_a) != len(rankings_b).
    """
    _check_lengths(rankings_a, rankings_b)
    total = 0.0
    for first, second in zip(rankings_a, rankings_b):
        top_a, top_b = set(first[:k]), set(second[:k])
        denominator = max(len(top_a), len(top_b)) or 1
        total += len(top_a & top_b) / denominator
    return total / (len(rankings_a) or 1)
