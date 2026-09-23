"""Baseline không ngữ nghĩa: BM25 khớp từ khoá trên caption.

Space này không nằm trong Qdrant. Nó tồn tại để trả lời một câu hỏi duy nhất
trong báo cáo: tìm kiếm ngữ nghĩa hơn khớp từ khoá bao nhiêu?
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.corpus import Corpus, record_payload
from app.vectordb import Hit

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tách token đơn giản: hạ chữ thường, giữ chữ và số, bỏ dấu câu.

    Cố tình đơn giản. Một baseline phải dễ giải thích; nếu nó thắng CLIP ở đâu
    thì ta muốn biết chắc đó không phải nhờ một bước tiền xử lý tinh vi.
    """
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Bm25Retriever:
    """Index BM25 trên từng caption, điểm mỗi ảnh là điểm cao nhất trong caption của nó."""

    bm25: BM25Okapi
    caption_image_ids: np.ndarray
    unique_image_ids: np.ndarray
    corpus: Corpus
    _rows: np.ndarray = field(init=False, repr=False)
    _row_of: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._row_of = {int(i): row for row, i in enumerate(self.unique_image_ids)}
        self._rows = np.array(
            [self._row_of[int(i)] for i in self.caption_image_ids], dtype=np.int64
        )

    @classmethod
    def build(cls, corpus: Corpus) -> "Bm25Retriever":
        """Dựng index từ mọi caption trong corpus."""
        pairs = corpus.caption_pairs()
        documents = [tokenize(text) for _, _, text in pairs]
        return cls(
            bm25=BM25Okapi(documents),
            caption_image_ids=np.array([img for img, _, _ in pairs], dtype=np.int64),
            unique_image_ids=np.array(corpus.image_ids(), dtype=np.int64),
            corpus=corpus,
        )

    def save(self, path: Path) -> None:
        """Pickle phần index. Corpus không được pickle — nó nạp lại từ corpus.jsonl."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "caption_image_ids": self.caption_image_ids,
                    "unique_image_ids": self.unique_image_ids,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path, corpus: Corpus) -> "Bm25Retriever":
        """Nạp index đã pickle và ghép lại với corpus.

        :raises FileNotFoundError: nếu chưa build. Người gọi nên dịch lỗi này
            thành thông báo "chạy python tasks.py build --space bm25-cap".
        """
        if not path.exists():
            raise FileNotFoundError(f"Chưa có index BM25 tại {path}")
        with path.open("rb") as fh:
            state = pickle.load(fh)
        return cls(corpus=corpus, **state)

    def search(self, query: str, k: int, allowed_ids: set[int] | None = None) -> list[Hit]:
        """Top-k ảnh theo điểm BM25.

        Điểm của một ảnh là điểm cao nhất trong các caption của nó (không phải
        trung bình): một caption nói đúng thứ người dùng tìm là đủ, và lấy
        trung bình sẽ phạt ảnh có nhiều caption nói về khía cạnh khác.

        Ảnh có điểm 0 (không khớp token nào) bị loại thay vì xếp cuối, để kết
        quả không lẫn những ảnh hoàn toàn không liên quan.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        caption_scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        per_image = np.zeros(len(self.unique_image_ids), dtype=np.float64)
        np.maximum.at(per_image, self._rows, caption_scores)

        if allowed_ids is not None:
            mask = np.array(
                [int(i) in allowed_ids for i in self.unique_image_ids], dtype=bool
            )
            per_image = np.where(mask, per_image, 0.0)

        order = np.argsort(-per_image, kind="stable")[:k]
        hits: list[Hit] = []
        for row in order:
            score = float(per_image[row])
            if score <= 0.0:
                break
            image_id = int(self.unique_image_ids[row])
            hits.append(
                Hit(
                    id=image_id,
                    score=score,
                    payload=record_payload(self.corpus.by_image_id[image_id]),
                )
            )
        return hits
