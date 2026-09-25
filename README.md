# Tìm kiếm ngữ nghĩa đa phương thức trên MS-COCO

Tìm ảnh bằng câu chữ tự do hoặc bằng một tấm ảnh, trên 5.000 ảnh COCO val2017.
Backend FastAPI + Qdrant, frontend React. 5 không gian nhúng + 2 baseline, kèm
bộ số liệu ablation thật trong `results/` (không phải số ước lượng).

Xem `docs/report.md` để đọc phân tích đầy đủ, và `results/qualitative.md` để
xem 14 ví dụ tốt/xấu kèm hình.

## Yêu cầu

- Python 3.11, Node 18+
- Docker Desktop đang chạy (Qdrant)
- Khoảng 6GB đĩa trống (ảnh COCO + checkpoint model), không cần GPU

## Chạy lại từ đầu

```bash
docker compose up -d qdrant            # cần Docker Desktop đang chạy
pip install -r requirements.txt
cp .env.example .env

python tasks.py ingest                 # tải COCO, sinh corpus.jsonl + thumbnail
python tasks.py build --space all      # encode + nạp Qdrant, ~30-35 phút trên CPU
python tasks.py querysets              # sinh data/queryset_short.json (các bộ
                                        # queryset_vi.json / queryset_i2i.json
                                        # khác đã commit sẵn, xem ghi chú dưới)
python tasks.py eval --axis all        # sinh results/*.csv + *.md

python tasks.py serve                  # API tại http://localhost:8000
```

Terminal khác:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev                            # UI tại http://localhost:5173
```

**Ghi chú về query set.** `data/queryset_vi.json` (200 caption dịch tay sang
tiếng Việt, dùng cho trục ablation 5) và `data/queryset_i2i.json` (200
image_id chọn ngẫu nhiên, dùng đo ảnh→ảnh) đã được **commit sẵn vào repo** vì
bản dịch tiếng Việt cần review thủ công và không tái sinh giống hệt được mỗi
lần chạy. `python tasks.py querysets` sinh (hoặc — nếu đã tồn tại — bỏ qua trừ
khi có `--force`) `data/queryset_short.json`, bộ query ngắn kiểu từ khoá dùng
cho trục ablation 3(b).

**Idempotent.** `python tasks.py build --space <tên>` bỏ qua space đã build
sẵn; thêm `--force` để build lại. Có thể build từng space riêng lẻ, ví dụ
`python tasks.py build --space clip-b32`, thay vì `--space all` một lần —
hữu ích nếu máy yếu hoặc muốn build qua đêm.

## Test

```bash
python -m pytest                       # suite nhanh, không tải model, không cần Qdrant
python -m pytest -m integration        # test tải model thật (kiểm chứng mclip-b32
                                        # cùng không gian với clip-b32, §5.3 spec)
cd frontend && npm test && npm run typecheck
```

## Tài liệu

- Thiết kế: `docs/superpowers/specs/2026-09-23-multimodal-search-design.md`
- Báo cáo: `docs/report.md`
- Bảng số liệu ablation: `results/axis*.md` (+ `.csv` tương ứng), build metadata: `results/index_meta/*.json` (`data/index_meta/*.json` là bản sinh ra khi build lại, bị gitignore nên không có sẵn trên checkout mới)
- Phân tích tốt/xấu (14 ví dụ, kèm hình): `results/qualitative.md`, hình tại `results/figures/`
- Kịch bản quay demo (chưa quay, xem ghi chú trong file): `docs/demo-script.md`

## Xử lý sự cố thường gặp

- **`Không kết nối được Qdrant`** khi chạy `build`/`eval`/`serve`: chạy
  `docker compose up -d qdrant`, kiểm tra `curl http://localhost:6333/readyz`.
- **`Space chưa build index` (409)** khi gọi `/search/text` hoặc `/search/image`:
  chạy đúng lệnh API trả về, ví dụ `python tasks.py build --space siglip-b16`.
- **Build chậm hơn nhiều so với ước lượng**: build từng space độc lập
  (`--space <tên>` thay vì `--space all`) để không phải chạy lại từ đầu nếu
  gián đoạn giữa chừng — lệnh idempotent nên chạy lại an toàn.
