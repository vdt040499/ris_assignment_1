"""Sinh hình minh hoạ cho `results/qualitative.md` (Task 18).

**Đây là script một lần (throwaway), không phải một phần của package `backend/`
đã review** — nó không được unit-test, không nằm trong `backend/cli/` (nơi
dành cho 4 subcommand production: ingest/build_index/make_querysets/evaluate),
và không được `tasks.py` biết tới. Nó sống ở `scripts/` để tách rõ "hạ tầng dự
án" khỏi "công cụ dựng report một lần rồi thôi".

**Vì sao không phải screenshot UI:** môi trường chạy agent này không có trình
duyệt / Playwright khả dụng. Script này đạt cùng mục đích bằng chứng cứ (cho
thấy ảnh thật được truy hồi cho query thật) bằng một đường khác: gọi thật
`POST /search/text` trên API đang chạy (`python tasks.py serve`), tải thumbnail
thật qua `GET /thumbs/{file_name}`, rồi ghép chúng vào một ảnh lưới bằng
Pillow. Đây là ảnh ghép từ kết quả tìm kiếm thật, KHÔNG phải screenshot của
frontend đang chạy — `results/qualitative.md` phải nói rõ điều này.

Không hardcode base URL: đọc `api_host`/`api_port` từ `app.config.get_settings()`
(cùng nguồn cấu hình mà `tasks.py serve` dùng), có thể override bằng `--base-url`.
Mọi tham số khác (space, k, số cột, thư mục ra) đều là CLI argument.

Cách chạy (yêu cầu `python tasks.py serve` đang chạy ở terminal khác):

    python scripts/generate_qualitative_figures.py

Với mỗi query, script ghi hai file vào `--out-dir` (mặc định `results/figures`):
- `<file>.png`  — ảnh lưới k thumbnail, mỗi ô có rank + score.
- `<file>.json` — nguyên văn response JSON của `/search/text`, để truy vết số
  liệu (đúng ý bao nhiêu / 10) trong `qualitative.md` về một file cụ thể thay
  vì gõ từ trí nhớ.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# Windows mặc định stdout/stderr theo codepage hệ thống (cp1252), làm print()
# tiếng Việt crash với UnicodeEncodeError. Cùng cách xử lý với tasks.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import get_settings  # noqa: E402


@dataclass(frozen=True)
class QuerySpec:
    group: str
    query: str
    file_name: str


# 10 query phân tích lỗi (Step 1 của task-18-brief.md), theo đúng bảng trong brief.
BAD_QUERIES: tuple[QuerySpec, ...] = (
    QuerySpec("Đếm số lượng", "three dogs", "bad_count_three_dogs.png"),
    QuerySpec("Đếm số lượng", "exactly two people sitting", "bad_count_two_people.png"),
    QuerySpec("Phủ định", "a street with no cars", "bad_negation_no_cars.png"),
    QuerySpec("Phủ định", "a plate without any vegetables", "bad_negation_no_vegetables.png"),
    QuerySpec("Quan hệ không gian", "a cup to the left of a laptop", "bad_spatial_cup_left.png"),
    QuerySpec("Quan hệ không gian", "a dog under a table", "bad_spatial_dog_under.png"),
    QuerySpec("Chữ trong ảnh", "a red sign that says STOP", "bad_ocr_stop_sign.png"),
    QuerySpec("Thuộc tính mịn", "a man in a striped blue shirt", "bad_attr_striped_shirt.png"),
    QuerySpec("Tổ hợp hiếm", "a person riding a zebra", "bad_rare_riding_zebra.png"),
    QuerySpec("Tổ hợp hiếm", "a cat wearing sunglasses", "bad_rare_cat_sunglasses.png"),
)

# 4 query tốt để đối chiếu.
GOOD_QUERIES: tuple[QuerySpec, ...] = (
    QuerySpec("Tốt", "a man riding a horse on the beach", "good_horse_beach.png"),
    QuerySpec("Tốt", "a plate of pizza on a wooden table", "good_pizza_table.png"),
    QuerySpec("Tốt", "a double decker bus on a city street", "good_bus_street.png"),
    QuerySpec("Tốt", "skiers on a snowy mountain slope", "good_skiers_snow.png"),
)

ALL_QUERIES: tuple[QuerySpec, ...] = BAD_QUERIES + GOOD_QUERIES

FONT_CANDIDATES = (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\consola.ttf")


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fetch_search(base_url: str, query: str, space: str, k: int) -> dict:
    """Gọi thật POST /search/text trên API đang chạy."""
    resp = requests.post(
        f"{base_url}/search/text",
        json={"query": query, "space": space, "k": k, "exact": True},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_thumb(base_url: str, thumb_url: str) -> Image.Image:
    """Tải thumbnail thật qua GET /thumbs/{file_name}."""
    resp = requests.get(f"{base_url}{thumb_url}", timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def compose_grid(
    query: str,
    space: str,
    results: list[dict],
    thumbs: list[Image.Image],
    cols: int,
    cell_size: int,
) -> Image.Image:
    """Ghép danh sách thumbnail thành một ảnh lưới, mỗi ô có nhãn rank + score.

    :param results: danh sách item của response `/search/text` (đã có rank, score).
    :param thumbs: ảnh thumbnail thật tương ứng theo cùng thứ tự với `results`.
    :param cols: số cột trong lưới (số hàng suy ra từ len(results)/cols).
    :param cell_size: cạnh ô vuông (thumbnail được resize/pad vào ô này).
    :return: một ảnh RGB duy nhất chứa toàn bộ lưới kèm tiêu đề query ở trên cùng.
    """
    n = len(results)
    rows = (n + cols - 1) // cols
    label_h = 26
    header_h = 40
    grid_w = cols * cell_size
    grid_h = header_h + rows * (cell_size + label_h)
    canvas = Image.new("RGB", (grid_w, grid_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    label_font = _load_font(14)
    draw.text((8, 8), f'query="{query}"  space={space}  k={n}', fill=(20, 20, 20), font=title_font)

    for i, (item, thumb) in enumerate(zip(results, thumbs)):
        row, col = divmod(i, cols)
        x0 = col * cell_size
        y0 = header_h + row * (cell_size + label_h)
        fitted = thumb.copy()
        fitted.thumbnail((cell_size, cell_size))
        paste_x = x0 + (cell_size - fitted.width) // 2
        paste_y = y0 + (cell_size - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))
        draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], outline=(200, 200, 200))
        label = f"#{item['rank']}  score={item['score']:.3f}"
        draw.text((x0 + 4, y0 + cell_size + 4), label, fill=(20, 20, 20), font=label_font)

    return canvas


def run(base_url: str, space: str, k: int, cols: int, cell_size: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in ALL_QUERIES:
        data = fetch_search(base_url, spec.query, space, k)
        thumbs = [fetch_thumb(base_url, r["thumb_url"]) for r in data["results"]]
        grid = compose_grid(spec.query, space, data["results"], thumbs, cols, cell_size)

        png_path = out_dir / spec.file_name
        json_path = out_dir / (Path(spec.file_name).stem + ".json")
        grid.save(png_path)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{spec.group}] {spec.query!r} -> {png_path.name} ({len(data['results'])} kết quả, "
              f"latency_ms={data['latency_ms']:.1f})")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=f"http://{settings.api_host}:{settings.api_port}",
        help="Base URL của API đang chạy (mặc định lấy từ app.config.get_settings()).",
    )
    parser.add_argument("--space", default="clip-b32", help="Space dùng để search (mặc định clip-b32).")
    parser.add_argument("--k", type=int, default=10, help="Số kết quả top-k (mặc định 10).")
    parser.add_argument("--cols", type=int, default=5, help="Số cột trong lưới ảnh (mặc định 5).")
    parser.add_argument("--cell-size", type=int, default=160, help="Cạnh ô vuông tính bằng pixel.")
    parser.add_argument(
        "--out-dir",
        default=str(settings.results_dir / "figures"),
        help="Thư mục ghi *.png + *.json (mặc định results/figures).",
    )
    args = parser.parse_args(argv)
    run(args.base_url, args.space, args.k, args.cols, args.cell_size, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
