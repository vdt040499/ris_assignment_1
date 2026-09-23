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
    """Mean Reciprocal Rank, tính trong top-k; query không có hit đóng góp 0."""
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
    """P@k cho ảnh→ảnh: một kết quả tính là liên quan nếu chia sẻ ≥1 nhãn với query.

    Đây là proxy, không phải ground-truth thật; nó thiên vị ảnh nhiều object.
    Query không có nhãn nào đóng góp 0.
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
    """mAP@k với độ liên quan = có chia sẻ nhãn.

    AP@k ở đây chuẩn hoá theo **số item liên quan tìm được trong top-k**, không
    theo tổng số item liên quan trong corpus (con số đó vô nghĩa với proxy
    nhãn, vì hàng nghìn ảnh cùng chứa 'person'). Định nghĩa này phải được ghi
    đúng như vậy trong báo cáo.
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
    """Tỉ lệ trùng nhau giữa hai top-k, bình quân trên các query.

    Dùng cho trục ANN vs exact: ``rankings_b`` là kết quả exact làm chuẩn vàng.
    """
    _check_lengths(rankings_a, rankings_b)
    total = 0.0
    for first, second in zip(rankings_a, rankings_b):
        top_a, top_b = set(first[:k]), set(second[:k])
        denominator = max(len(top_a), len(top_b)) or 1
        total += len(top_a & top_b) / denominator
    return total / (len(rankings_a) or 1)
