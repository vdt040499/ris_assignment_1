"""Read/write corpus.jsonl — the system's single source of truth.

After the ingest step, no component reads the original COCO JSON file again;
everything goes through this module.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

from app.errors import CorpusError


class CorpusRecord(BaseModel):
    """A single image in the corpus with its captions and category metadata."""

    image_id: int
    file_name: str
    width: int
    height: int
    captions: list[str]
    categories: list[str] = []
    supercategories: list[str] = []

    @field_validator("captions")
    @classmethod
    def _clean_captions(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("record must have at least one non-empty caption")
        return cleaned


@dataclass
class Corpus:
    """The entire corpus loaded in RAM, with a fast lookup index by image_id."""

    records: list[CorpusRecord]
    by_image_id: dict[int, CorpusRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_image_id = {r.image_id: r for r in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def image_ids(self) -> list[int]:
        return [r.image_id for r in self.records]

    def caption_pairs(self) -> list[tuple[int, int, str]]:
        """Flatten every caption into ``(image_id, caption_index, text)``.

        Used both for building the caption collection and for generating the
        text→image query set, so both sides always assign the same index to
        the same caption.
        """
        return [
            (rec.image_id, idx, text)
            for rec in self.records
            for idx, text in enumerate(rec.captions)
        ]


def record_payload(record: CorpusRecord) -> dict:
    """Payload attached to each Qdrant point. Keeps exactly the fields needed for filtering and display."""
    return {
        "image_id": record.image_id,
        "file_name": record.file_name,
        "width": record.width,
        "height": record.height,
        "captions": record.captions,
        "categories": record.categories,
        "supercategories": record.supercategories,
    }


CAPTION_ID_STRIDE = 100


def caption_point_id(image_id: int, caption_index: int) -> int:
    """Qdrant point id for a caption.

    Combines image_id with the caption index using a fixed stride, so ids in
    the caption collection never collide with ids in the image collection,
    and a caption id can always be traced back to its source image.

    :raises ValueError: if ``caption_index`` >= CAPTION_ID_STRIDE, since the
        formula would then lose its injectivity.
    """
    if not 0 <= caption_index < CAPTION_ID_STRIDE:
        raise ValueError(
            f"caption_index {caption_index} is out of range [0, {CAPTION_ID_STRIDE})"
        )
    return image_id * CAPTION_ID_STRIDE + caption_index


def caption_payload(record: CorpusRecord, caption_index: int) -> dict:
    """Payload for a caption point: the image's metadata + the caption itself."""
    return {
        **record_payload(record),
        "caption_index": caption_index,
        "caption": record.captions[caption_index],
    }


def write_corpus(records: Iterable[CorpusRecord], path: Path) -> int:
    """Write the corpus out as JSONL. Returns the number of records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
            count += 1
    return count


def load_corpus(path: Path) -> Corpus:
    """Load corpus.jsonl and validate each line.

    :raises CorpusError: the file doesn't exist, is empty, or has a line with
        an invalid schema — the message states the line number so it can be
        fixed right away.
    """
    if not path.exists():
        raise CorpusError(f"{path} not found. Run: python tasks.py ingest")
    records: list[CorpusRecord] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(CorpusRecord.model_validate_json(line))
            except ValidationError as exc:
                raise CorpusError(f"{path} has an invalid schema at line {lineno}: {exc}") from exc
    if not records:
        raise CorpusError(f"{path} is empty. Re-run: python tasks.py ingest")
    return Corpus(records=records)
