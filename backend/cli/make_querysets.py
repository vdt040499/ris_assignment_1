"""Sinh các query set dùng cho eval. Mọi lấy mẫu đều có seed cố định.

Ba file sinh ra được commit vào repo, vì không reproduce được query set thì
không reproduce được số liệu.
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
    """Lấy mẫu caption để dịch tay sang tiếng Việt.

    Args:
        corpus: Corpus đã nạp, cung cấp toàn bộ cặp ``(image_id,
            caption_index, text)`` để lấy mẫu.
        size: Số lượng caption muốn lấy mẫu; bị cắt xuống bằng số cặp caption
            thực có nếu ``size`` lớn hơn.
        seed: Seed cho bộ sinh số ngẫu nhiên cục bộ (``random.Random``, không
            phải module ``random`` toàn cục) để việc lấy mẫu tất định và có
            thể tái lập giữa các lần chạy.

    Returns:
        Danh sách dict ``{image_id, caption_index, en, vi}``, trong đó ``vi``
        luôn rỗng — bước dịch tay sẽ điền vào sau. Lấy mẫu trên tập
        ``(image_id, caption_index)`` không lặp lại nên không có caption nào
        bị trùng.
    """
    pairs = corpus.caption_pairs()
    rng = random.Random(seed)
    chosen = rng.sample(pairs, min(size, len(pairs)))
    return [
        {"image_id": image_id, "caption_index": index, "en": text, "vi": ""}
        for image_id, index, text in chosen
    ]


def sample_i2i_queries(corpus: Corpus, size: int, seed: int) -> list[int]:
    """Lấy mẫu image_id làm query cho chiều ảnh→ảnh.

    Args:
        corpus: Corpus đã nạp, cung cấp danh sách ``image_ids()``.
        size: Số lượng ảnh muốn lấy mẫu; bị cắt xuống bằng số ảnh thực có
            trong corpus nếu ``size`` lớn hơn.
        seed: Seed cho ``random.Random`` cục bộ để lấy mẫu tất định.

    Returns:
        Danh sách ``image_id`` duy nhất (không lặp), đã sắp xếp tăng dần.
    """
    rng = random.Random(seed)
    return sorted(rng.sample(corpus.image_ids(), min(size, len(corpus))))


def build_short_queries(corpus: Corpus) -> list[dict]:
    """Một query ngắn cho mỗi category thực sự xuất hiện trong corpus.

    Args:
        corpus: Corpus đã nạp; mỗi ``CorpusRecord`` mang danh sách
            ``categories`` của ảnh đó.

    Returns:
        Danh sách dict ``{query, gold_category, n_gold}``, sắp theo tên
        category. ``n_gold`` là số ảnh chứa category đó — báo cáo cần con số
        này để người đọc biết mẫu số của P@k cho từng query. Category không
        xuất hiện ảnh nào (không thể xảy ra vì Counter chỉ đếm cái có mặt)
        được lọc bỏ tường minh để an toàn.
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
    """CLI entrypoint: sinh ba file query set vào ``settings.data_dir``.

    Args:
        argv: Danh sách tham số dòng lệnh (không gồm tên chương trình); dùng
            ``sys.argv[1:]`` khi để ``None``.

    Returns:
        Mã thoát tiến trình (luôn ``0``).
    """
    parser = argparse.ArgumentParser(description="Sinh query set cho eval")
    parser.add_argument("--force", action="store_true",
                        help="ghi đè file đã có (mất bản dịch tiếng Việt!)")
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
            print(f"{filename}: đã có, bỏ qua (dùng --force để ghi đè)")
            continue
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{filename}: {len(payload)} entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
