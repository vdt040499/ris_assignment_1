"""Non-semantic baseline: BM25 keyword matching on captions.

This space does not live in Qdrant. It exists to answer a single question
in the report: how much better is semantic search than keyword matching?
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
    """Simple tokenization: lowercase, keep letters and digits, drop punctuation.

    Deliberately simple. A baseline must be easy to explain; if it beats CLIP
    anywhere, we want to be sure it's not because of some clever preprocessing step.
    """
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Bm25Retriever:
    """BM25 index over individual captions; each image's score is the highest score among its captions."""

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
        """Build the index from every caption in the corpus."""
        pairs = corpus.caption_pairs()
        documents = [tokenize(text) for _, _, text in pairs]
        return cls(
            bm25=BM25Okapi(documents),
            caption_image_ids=np.array([img for img, _, _ in pairs], dtype=np.int64),
            unique_image_ids=np.array(corpus.image_ids(), dtype=np.int64),
            corpus=corpus,
        )

    def save(self, path: Path) -> None:
        """Pickle the index portion. The corpus itself is not pickled — it is reloaded from corpus.jsonl."""
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
        """Load the pickled index and re-attach it to the corpus.

        :raises FileNotFoundError: if it hasn't been built yet. Callers should
            translate this error into the message "run python tasks.py build --space bm25-cap".
        """
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index found at {path}")
        with path.open("rb") as fh:
            state = pickle.load(fh)
        return cls(corpus=corpus, **state)

    def search(self, query: str, k: int, allowed_ids: set[int] | None = None) -> list[Hit]:
        """Top-k images by BM25 score.

        An image's score is the highest score among its captions (not the average):
        one caption saying exactly what the user is looking for is enough, and
        averaging would penalize images with many captions about other aspects.

        Images with a score of 0 (no token matched) are excluded rather than ranked
        last, so the results don't include completely irrelevant images.
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
