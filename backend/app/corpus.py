"""Đọc/ghi corpus.jsonl — nguồn sự thật duy nhất của hệ.

Sau bước ingest, không thành phần nào đọc lại file COCO JSON gốc; tất cả đi
qua module này.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator

from app.errors import CorpusError


class CorpusRecord(BaseModel):
    """Một ảnh trong corpus kèm caption và metadata category của nó."""

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
            raise ValueError("record phải có ít nhất một caption không rỗng")
        return cleaned


@dataclass
class Corpus:
    """Toàn bộ corpus nạp trong RAM, kèm chỉ mục tra nhanh theo image_id."""

    records: list[CorpusRecord]
    by_image_id: dict[int, CorpusRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.by_image_id = {r.image_id: r for r in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def image_ids(self) -> list[int]:
        return [r.image_id for r in self.records]

    def caption_pairs(self) -> list[tuple[int, int, str]]:
        """Dàn phẳng mọi caption thành ``(image_id, caption_index, text)``.

        Dùng cho cả việc build collection caption và việc sinh query set
        text→ảnh, nên hai bên luôn đánh cùng một chỉ số cho cùng một caption.
        """
        return [
            (rec.image_id, idx, text)
            for rec in self.records
            for idx, text in enumerate(rec.captions)
        ]


def record_payload(record: CorpusRecord) -> dict:
    """Payload gắn kèm mỗi point Qdrant. Giữ đúng các field cần để filter và hiển thị."""
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
    """Id point Qdrant cho một caption.

    Trộn image_id với chỉ số caption theo stride cố định, nên id trong
    collection caption không bao giờ đụng id trong collection ảnh và từ một id
    caption luôn suy lại được ảnh gốc.

    :raises ValueError: nếu ``caption_index`` >= CAPTION_ID_STRIDE, vì khi đó
        công thức mất tính đơn ánh.
    """
    if not 0 <= caption_index < CAPTION_ID_STRIDE:
        raise ValueError(
            f"caption_index {caption_index} ngoài khoảng [0, {CAPTION_ID_STRIDE})"
        )
    return image_id * CAPTION_ID_STRIDE + caption_index


def caption_payload(record: CorpusRecord, caption_index: int) -> dict:
    """Payload cho một point caption: metadata của ảnh + chính caption đó."""
    return {
        **record_payload(record),
        "caption_index": caption_index,
        "caption": record.captions[caption_index],
    }


def write_corpus(records: Iterable[CorpusRecord], path: Path) -> int:
    """Ghi corpus ra JSONL. Trả về số record đã ghi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
            count += 1
    return count


def load_corpus(path: Path) -> Corpus:
    """Nạp corpus.jsonl và validate từng dòng.

    :raises CorpusError: file không tồn tại, rỗng, hoặc có dòng sai schema —
        thông báo nêu rõ số dòng để sửa được ngay.
    """
    if not path.exists():
        raise CorpusError(f"Không thấy {path}. Chạy: python tasks.py ingest")
    records: list[CorpusRecord] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(CorpusRecord.model_validate_json(line))
            except ValidationError as exc:
                raise CorpusError(f"{path} sai schema ở dòng {lineno}: {exc}") from exc
    if not records:
        raise CorpusError(f"{path} rỗng. Chạy lại: python tasks.py ingest")
    return Corpus(records=records)
