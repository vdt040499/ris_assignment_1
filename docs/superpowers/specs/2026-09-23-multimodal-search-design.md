# Hệ tìm kiếm ngữ nghĩa đa phương thức trên MS-COCO

**Ngày:** 2026-09-23
**Trạng thái:** Spec đã chốt, chờ implementation plan
**Loại:** Architectural (project mới)

---

## 1. Mục tiêu và tiêu chí thành công

Xây một hệ tìm kiếm ngữ nghĩa cho phép truy vấn bằng **câu chữ tự do** hoặc bằng **một tấm ảnh**, trả về top-k ảnh liên quan.

Đây là bài tập môn học có rubric. Vì vậy "thành công" được định nghĩa theo đúng các mục rubric, không phải theo độ phức tạp kỹ thuật:

| Mục rubric | Cách spec này đáp ứng |
|---|---|
| Chọn dataset | MS-COCO val2017: 5.000 ảnh, 25.014 caption người viết, metadata category |
| Hệ tìm kiếm ngữ nghĩa, input ảnh hoặc text | Hai chiều chính: text→ảnh, ảnh→ảnh. Chiều ảnh→text có trong API (xem §4.2) |
| Frontend gọi hàm tìm kiếm qua API, hiện top-k ảnh | React SPA tách biệt, gọi REST API của FastAPI |
| Phân tích kết quả tốt/xấu | Phân loại lỗi 5 nhóm, ~10 query minh hoạ kèm ảnh (§7.3) |
| Ablation về các model đã thực nghiệm | 5 không gian nhúng + 2 baseline, 5 trục ablation (§6) |
| Hệ chạy được + demo | Chạy local, một chuỗi lệnh; video demo 2-3 phút |

**Tiêu chí bao trùm:** người chấm phải chạy lại được hệ bằng các lệnh trong README, và mọi con số trong báo cáo phải truy được về một file trong `results/`.

### 1.1 Ràng buộc môi trường (đã kiểm chứng)

- Máy **không có GPU** (`torch 2.8.0+cpu`, không có NVIDIA). Mọi encode chạy trên CPU.
- RAM 16 GB, giới hạn số model nạp đồng thời.
- Windows 10, Python 3.11.9, Node 18.20.8.
- Có sẵn: `transformers 5.14.1`, `sentence-transformers 5.3.0`, `fastapi 0.124.4`, `uvicorn`, `datasets`, `hnswlib`, `numpy 2.2.6`, `Pillow`.
- Cần cài thêm: `qdrant-client`, `rank-bm25`, `pydantic-settings`, `pytest`.
- **Không cần** `faiss` (Qdrant lo phần index) và **không cần** `open_clip` (bản OpenCLIP LAION tải được qua `transformers`).
- Docker CLI 29.5.2 + Compose v5.1.3 đã cài, nhưng **daemon chưa chạy**, phải bật Docker Desktop.

Ngân sách CPU một lần cho toàn bộ index: khoảng 30-45 phút (§5.4).

---

## 2. Quyết định đã chốt

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Dataset | COCO val2017 (5k ảnh) | Có ground-truth caption thật nên đo được R@1/5/10, MRR. Đủ nhỏ cho CPU. Tải ~1 GB |
| Chiều truy vấn (UI) | text→ảnh, ảnh→ảnh | Theo yêu cầu "hiển thị top-k ảnh kết quả" |
| Chiều ảnh→text | Có trong API, ẩn trên UI | Cần index caption cho eval nên chiều này gần như miễn phí; giữ lại để không hụt mục "output ảnh hoặc text" của đề bài |
| Lưu trữ vector | **Qdrant qua Docker** | Người dùng chọn kiến trúc vector DB thật. Qdrant hỗ trợ `exact: true` tại query time nên vẫn lấy được số chính xác cho báo cáo |
| Tìm kiếm chính xác | `exact=true` làm chuẩn vàng cho eval | Với 5k vector, exact vừa đúng vừa nhanh; HNSW trở thành đối tượng thí nghiệm chứ không phải mặc định |
| Đa ngôn ngữ | Thêm model multilingual vào lineup | Demo gõ tiếng Việt chạy tốt, đồng thời thành một trục ablation thật |
| Backend | FastAPI | Cùng ngôn ngữ với phần model |
| Frontend | Vite + React + TS + Tailwind | Tách biệt frontend/API đúng yêu cầu đề bài; Tailwind theo chuẩn CLAUDE.md, không inline style |
| Giao demo | Chạy local + video | Không phụ thuộc hạ tầng ngoài, không lo quota |

### 2.1 Phương án dự phòng khi không có Docker

`qdrant-client` có chế độ embedded (`QdrantClient(path=...)`) không cần Docker. **Giới hạn đã biết:** chế độ embedded luôn chạy brute-force và bỏ qua cấu hình HNSW.

Hệ quả bắt buộc phải tôn trọng:

- Trục ablation "ANN vs exact" (§6.2, trục 4) **chỉ chạy được với Qdrant server qua Docker**.
- Chế độ embedded chỉ dùng để demo và để chạy test, không dùng để sinh số liệu ANN.
- `GET /health` phải báo rõ đang ở chế độ nào, và `tasks.py eval` phải **dừng với thông báo rõ ràng** nếu ai đó cố chạy trục 4 ở chế độ embedded.

---

## 3. Kiến trúc

Bốn thành phần, mỗi thành phần một nhiệm vụ, giao tiếp qua interface rõ ràng.

```
COCO val2017 ──ingest──> corpus.jsonl + thumbs/
                              │
                              ├──build_index(space)──> Qdrant collection (vector + payload)
                              │                              ▲         │
                              │                              │         │
      browser ──HTTP──> FastAPI ──encode query──────────────┘         │
         ▲                  │ ◄──────── top-k + payload ───────────────┘
         └── ảnh/thumbnail ─┘
                              │
                              └──eval CLI──> results/*.csv, *.md ──> báo cáo
```

### 3.1 `ingest` (offline, một lần)

Tải COCO val2017 images + `captions_val2017.json` + `instances_val2017.json`. Sinh thumbnail cạnh dài 256px. Xuất **một** file `data/corpus.jsonl`, mỗi dòng:

```json
{"image_id": 397133, "file_name": "000000397133.jpg", "width": 640, "height": 427,
 "captions": ["...", "...", "...", "...", "..."],
 "categories": ["person", "dining table"], "supercategories": ["person", "furniture"]}
```

**`corpus.jsonl` là nguồn sự thật duy nhất.** Sau bước này không thành phần nào đọc lại COCO JSON gốc. Ảnh có ít hơn 5 caption vẫn giữ nguyên số caption thực có; ảnh không có annotation category giữ mảng rỗng.

Bước này phải **resume được**: kiểm tra file đã tải, không tải lại, xác thực checksum.

### 3.2 `build_index` (offline, mỗi space một lần)

Nhận tên một space trong registry (§5), encode 5.000 ảnh theo batch trên CPU, upsert vào collection Qdrant kèm payload lấy từ `corpus.jsonl`. Ghi `data/index_meta/<space>.json`:

```json
{"space": "clip-b32", "hf_id": "openai/clip-vit-base-patch32", "dim": 512,
 "distance": "Cosine", "normalized": true, "n_points": 5000,
 "encode_seconds": 187.4, "batch_size": 32, "built_at": "2026-09-23T10:12:00",
 "git_commit": "abc1234"}
```

Báo cáo lấy thời gian build và số point từ file này, không chép tay.

Lệnh phải **idempotent**: chạy lại trên space đã build thì bỏ qua, trừ khi có `--force`.

### 3.3 `search-api` (FastAPI, online)

Nhận query text hoặc ảnh, chọn encoder theo `space` trong request, encode thành 1 vector, gọi Qdrant `query_points` với tuỳ chọn `exact` / `hnsw_ef` / filter payload, trả top-k kèm score và metadata. Cũng serve file ảnh và thumbnail.

Encoder nạp **lười**, cache LRU giữ tối đa `MODEL_CACHE_SIZE` (mặc định 2) model để RAM không vượt ~2 GB.

### 3.4 `eval` (offline CLI)

Đọc `corpus.jsonl`, encode query set (cache ra `data/cache/<space>_<queryset>.npy` để chạy lại tức thì), truy vấn **cùng một Qdrant mà demo đang dùng**, xuất `results/*.csv` và `results/*.md`.

Đây là lý do số liệu báo cáo và kết quả demo không thể lệch nhau: chúng đến từ cùng một index.

### 3.5 Ranh giới quan trọng nhất

**Registry là interface duy nhất giữa "phần model" và "phần hệ thống".** Mỗi space khai báo `{name, hf_id, dim, distance, normalized, collection, modes, languages, encode_image(), encode_text()}`.

Thêm model mới = thêm một entry trong registry. Không sửa API, không sửa frontend, không sửa eval. Đây là thứ làm ablation 5 model trở nên rẻ thay vì thành 5 lần copy-paste. `search.py` không biết CLIP hay SigLIP là gì; nó chỉ biết registry.

### 3.6 Xử lý lỗi

| Tình huống | Mã | Nội dung |
|---|---|---|
| Qdrant không kết nối được | 503 | Thông báo rõ + gợi ý `docker compose up -d qdrant`. Không trả stacktrace |
| Ảnh upload sai định dạng | 400 | Liệt kê định dạng cho phép (từ config `ALLOWED_IMAGE_TYPES`) |
| Ảnh upload quá lớn | 413 | Nêu giới hạn cụ thể (từ config `MAX_UPLOAD_MB`) |
| Space chưa build index | 409 | Kèm đúng lệnh cần chạy: `python tasks.py build --space <name>` |
| Query rỗng / `k` ngoài khoảng | 422 | Lỗi validation của pydantic |
| Space không tồn tại | 404 | Liệt kê các space hợp lệ |
| Space không hỗ trợ chiều truy vấn (vd `resnet50` với text) | 400 | Nêu các chiều mà space đó hỗ trợ |

---

## 4. API contract

Mọi mặc định đọc từ config. Không có hằng số nào viết cứng trong code (theo CLAUDE.md).

### 4.1 Endpoints

**`GET /health`**

```json
{"status": "ok", "qdrant": "ok", "qdrant_mode": "server",
 "spaces_ready": ["clip-b32", "laion-b32"], "spaces_missing": ["siglip-b16"]}
```

**`GET /spaces`** — registry + `index_meta`. Frontend dựng dropdown từ đây, nên thêm model không cần sửa frontend.

```json
[{"name": "clip-b32", "hf_id": "openai/clip-vit-base-patch32", "dim": 512,
  "modes": ["text2image", "image2image", "image2text"], "languages": ["en"],
  "ready": true, "n_points": 5000, "note": "baseline"}]
```

**`POST /search/text`**

```json
{"query": "a man riding a horse on the beach", "space": "clip-b32", "k": 20,
 "exact": true, "hnsw_ef": null, "target": "image",
 "filters": {"categories": ["horse"], "supercategories": []},
 "prompt_template": null}
```

`target`: `"image"` (mặc định) hoặc `"caption"`. `prompt_template` là `null` (dùng query thô) hoặc một chuỗi có `{}`, phục vụ trục ablation 3.

**`POST /search/image`** — luôn là `multipart/form-data`, không có biến thể JSON. Các field: `file` (ảnh upload) **hoặc** `image_id` (ảnh đã có trong corpus), cộng các field `space`, `k`, `exact`, `hnsw_ef`, `filters_json`. Phải có **đúng một** trong `file` / `image_id`; thiếu cả hai hoặc có cả hai thì trả 400. Một content type duy nhất cho một endpoint, để frontend và test không phải xử lý hai nhánh.

**`GET /examples`** — bộ query mẫu EN + VI cố định, để người chấm bấm một cái là thấy kết quả thay vì phải tự nghĩ query.

**`GET /thumbs/{file_name}`**, **`GET /images/{file_name}`** — serve ảnh, chỉ trong `DATA_DIR`, chặn path traversal.

### 4.2 Response chung

Cả hai chiều trả **một hình dạng duy nhất**, nghĩa là frontend có đúng một component grid, không phải hai:

```json
{"results": [{"image_id": 397133, "file_name": "000000397133.jpg",
              "thumb_url": "/thumbs/000000397133.jpg", "score": 0.3142,
              "captions": ["..."], "categories": ["person"], "rank": 1}],
 "latency_ms": 41.7, "space": "clip-b32", "exact": true,
 "total_searched": 5000, "query_echo": "a man riding a horse on the beach"}
```

Khi `target="caption"`, mỗi item thêm `matched_caption` và `caption_index`.

`latency_ms` đo thời gian encode + truy vấn + format phía server (không gồm thời gian mạng và render). Nó dùng để **hiển thị trên demo**.

Số latency **đưa vào báo cáo** thì do `eval` CLI đo, gọi Qdrant trực tiếp và tách riêng hai thành phần: `encode_ms` và `search_ms`. Lý do tách: trục ablation 4 so sánh ANN với exact, mà phần encode thì giống nhau ở cả hai, nên trộn chung sẽ làm loãng đúng cái hiệu ứng cần đo. Báo cáo ghi rõ mỗi bảng dùng con số nào.

### 4.3 Config (`.env`, pydantic-settings)

`QDRANT_MODE` (`server`|`embedded`), `QDRANT_URL`, `QDRANT_PATH`, `DATA_DIR`, `COLLECTION_PREFIX`, `TOP_K_DEFAULT` (20), `MAX_TOP_K` (100), `MAX_UPLOAD_MB` (10), `ALLOWED_IMAGE_TYPES`, `MODEL_CACHE_SIZE` (2), `DEVICE` (`cpu`), `BATCH_SIZE` (32), `THUMB_SIZE` (256), `HNSW_EF_DEFAULT`, `API_HOST`, `API_PORT`, `CORS_ORIGINS`.

---

## 5. Model lineup

### 5.1 Năm không gian nhúng + hai baseline

| Space | Model | Dim | Biến được thay đổi | Chiều hỗ trợ |
|---|---|---|---|---|
| `clip-b32` | `openai/clip-vit-base-patch32` | 512 | **baseline** | t2i, i2i, i2t |
| `clip-b16` | `openai/clip-vit-base-patch16` | 512 | kích thước patch (độ phân giải hiệu dụng) | t2i, i2i |
| `laion-b32` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 512 | **dữ liệu huấn luyện** (LAION-2B vs WIT-400M), kiến trúc y hệt baseline | t2i, i2i |
| `siglip-b16` | `google/siglip-base-patch16-224` | 768 | **hàm loss** (sigmoid vs softmax contrastive) | t2i, i2i |
| `mclip-b32` | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | 512 | **text tower đa ngôn ngữ**, dùng lại collection ảnh của `clip-b32` | t2i (đa ngữ) |
| `resnet50` | `microsoft/resnet-50` (ImageNet) | 2048 | baseline **không multimodal** | i2i |
| `bm25-cap` | BM25 trên toàn bộ caption | — | baseline **không ngữ nghĩa** | t2i |

`bm25-cap` là **space đặc biệt, không nằm trong Qdrant**: nó là một index từ khoá dựng từ `corpus.jsonl`, pickle ra `data/bm25_index.pkl` ở bước `build_index` và nạp lười như các model khác. Điểm của một ảnh là điểm BM25 cao nhất trong các caption của nó. Trong registry, space này có `dim: null`, `collection: null`, và cờ `backend: "bm25"` thay vì `"qdrant"`; `search.py` chọn nhánh theo cờ đó. Đây là ngoại lệ duy nhất trong registry và nó được khai báo tường minh chứ không xử lý bằng `if` rải rác.

### 5.2 Vì sao đúng bộ này

Mỗi cặp so sánh chỉ lệch **một** biến, nên bảng kết quả diễn giải được thay vì chỉ là một bảng xếp hạng:

- `clip-b32` vs `laion-b32` → cô lập ảnh hưởng của **dữ liệu huấn luyện** (cùng kiến trúc ViT-B/32).
- `clip-b32` vs `clip-b16` → cô lập ảnh hưởng của **kiến trúc/độ phân giải** (cùng dữ liệu OpenAI WIT).
- `clip-b16` vs `siglip-b16` → cô lập ảnh hưởng của **hàm loss** (cùng cỡ patch 16).
- `clip-b32` vs `resnet50` (i2i) → trả lời "CLIP có thật hơn feature ImageNet không?".
- `clip-b32` vs `bm25-cap` (t2i) → trả lời "ngữ nghĩa có thật hơn khớp từ khoá không?".
- `clip-b32` vs `mclip-b32` → đánh đổi đa ngôn ngữ.

### 5.3 Giả định phải kiểm chứng trước khi build

`mclip-b32` được distill để nằm **cùng không gian nhúng** với `clip-b32`, nên theo thiết kế nó dùng lại đúng collection ảnh `coco_clip_b32` và không cần build index riêng.

Giả định này **phải được kiểm chứng bằng test trước khi encode 25k vector**: encode cùng một caption tiếng Anh bằng cả hai model, cosine phải **> 0.9**.

Nếu test đỏ: giả định sai, `mclip-b32` phải có collection ảnh riêng, nghĩa là thêm một lần encode 5.000 ảnh (~3 phút) bằng image encoder đi kèm model đó. Xử lý ngay ở bước build, không để tới lúc viết báo cáo mới phát hiện.

### 5.4 Ngân sách CPU (ước lượng, một lần)

| Hạng mục | Nội dung encode | Ước lượng |
|---|---|---|
| `clip-b32` | 5.000 ảnh | ~3 phút |
| `laion-b32` | 5.000 ảnh | ~3 phút |
| `clip-b16` | 5.000 ảnh | ~12 phút |
| `siglip-b16` | 5.000 ảnh | ~12 phút |
| `resnet50` | 5.000 ảnh | ~3 phút |
| `coco_cap_clip_b32` | 25.014 caption | ~3 phút |
| `coco_clip_b32_raw` | dùng lại vector đã encode, chỉ upsert không normalize | < 1 phút |
| `bm25-cap` | dựng index từ khoá từ `corpus.jsonl`, không dùng model | < 1 phút |
| **Tổng** | | **~35 phút** |

Đây là ước lượng, không phải cam kết. Số thật ghi vào `index_meta` và đưa vào báo cáo.

---

## 6. Layout Qdrant và các trục ablation

### 6.1 Collection

Một collection cho mỗi space, `COSINE` trên vector đã L2-normalize:
`coco_clip_b32`, `coco_clip_b16`, `coco_laion_b32`, `coco_siglip_b16`, `coco_resnet50`.

Thêm hai collection đặc biệt:

- **`coco_clip_b32_raw`** — vector **chưa** normalize, distance `DOT`. Nhóm đối chứng cho ablation normalize.
- **`coco_cap_clip_b32`** — 25.014 vector caption, phục vụ chiều ảnh→text.

Payload mỗi point: `image_id`, `file_name`, `captions[]`, `categories[]`, `supercategories[]`, `width`, `height`. Tạo payload index trên `categories` và `supercategories` để filter chạy nhanh.

**ANN vs exact không cần collection riêng.** Qdrant nhận `params={"exact": true}` và `hnsw_ef` ngay tại query time trên cùng một collection. Eval lấy số exact làm chuẩn vàng rồi sweep `hnsw_ef`, cùng một index, không có nguy cơ hai index lệch nhau.

### 6.2 Năm trục ablation

| # | Trục | Cách đo | Điều cần trả lời |
|---|---|---|---|
| 1 | **Model** (5 space + 2 baseline) | R@1/5/10, MRR@10, thời gian build, latency p50/p95, RAM | Bảng kết quả chính |
| 2 | **Normalize** on/off | `coco_clip_b32` (cosine) vs `coco_clip_b32_raw` (dot) | Định lượng mức hỏng khi bỏ normalize |
| 3 | **Prompt template** | query thô vs `"a photo of {}"` vs `"a photo of {}, a type of scene"`, đo trên **hai bộ query riêng** (xem ghi chú dưới) | Prompt engineering đáng mấy điểm R@1, và đáng với loại query nào |
| 4 | **ANN vs exact** | sweep `hnsw_ef` ∈ {16, 64, 128, 256}; đo overlap@10 so với exact + p50/p95 latency | **Cần Docker server.** Dự đoán: với 5k vector, exact thắng cả về chính xác lẫn latency, dẫn tới kết luận "ANN đáng dùng từ quy mô nào" |
| 5 | **Ngôn ngữ** | 200 caption dịch tay sang tiếng Việt; `clip-b32` vs `mclip-b32` trên cả bộ EN và VI | Baseline sụp trên VI; multilingual mất vài điểm trên EN, đúng cái đánh đổi cần nêu |

**Ghi chú về trục 3.** Template `"a photo of {}"` vốn được thiết kế cho query ngắn kiểu tên lớp ("a dog"), nên bọc nó quanh một caption dài đầy đủ ("A man in a red shirt riding a horse along the beach") gần như chắc chắn không giúp gì, có khi còn hại. Đo mỗi trên caption dài rồi kết luận "prompt engineering vô dụng" sẽ là một kết luận sai vì chọn sai bộ query.

Nên trục 3 đo trên **hai bộ query**: (a) caption đầy đủ, và (b) `data/queryset_short.json` — 200 query ngắn kiểu từ khoá do tác giả viết cho các category COCO ("a zebra", "a stop sign", "pizza"), ground-truth là mọi ảnh có category tương ứng, dùng P@10 thay cho R@k. Kỳ vọng: template gần như không đổi trên (a) và giúp rõ trên (b). Đó mới là kết luận đúng và có ích.

**Các query set đều commit vào repo** để reproduce được:

| File | Nội dung | Dùng cho |
|---|---|---|
| `data/queryset_vi.json` | 200 caption chọn ngẫu nhiên (seed cố định) đã dịch tay sang tiếng Việt | Trục 5 |
| `data/queryset_i2i.json` | 200 image_id chọn ngẫu nhiên (seed cố định) | Đo ảnh→ảnh (§7.2) |
| `data/queryset_short.json` | 200 query ngắn kiểu từ khoá + category ground-truth | Trục 3 (b) |

Bộ tiếng Việt do tác giả dịch và review thủ công; báo cáo ghi rõ điều này.

---

## 7. Đo lường và phân tích

### 7.1 text→ảnh

Giao thức COCO 5k chuẩn: mỗi caption là một query, tìm trên 5.000 ảnh. Đo **R@1, R@5, R@10, MRR@10**. Một query tính là đúng nếu ảnh gốc của caption đó nằm trong top-k.

Con số 25.014 caption là số annotation thực có trong `captions_val2017.json`; `ingest` phải in ra con số đếm được thay vì giả định, và báo cáo dùng con số đó. Không phải ảnh nào cũng có đúng 5 caption.

Ghi rõ trong báo cáo: số sẽ *gần* nhưng không trùng khít số trong paper CLIP, vì val2017 khác Karpathy test split. Nêu điều này thay vì so sánh khập khiễng.

### 7.2 ảnh→ảnh

Không có ground-truth thật. Dùng proxy công khai:

- **P@k** = tỉ lệ trong top-k chia sẻ ít nhất một category COCO với ảnh query.
- **mAP theo supercategory**.

Báo cáo phải ghi rõ đây là proxy và nó thiên vị cái gì: **ảnh nhiều object dễ ăn điểm hơn**, vì xác suất trùng ít nhất một category cao hơn. Đây là điểm yếu duy nhất về mặt đo lường trong spec này; phương án tốt hơn (tự gán nhãn tay bộ query) bị loại vì chi phí.

Ảnh query cho i2i: 200 ảnh chọn ngẫu nhiên có seed cố định, lưu `data/queryset_i2i.json`, commit vào repo.

### 7.3 Phân tích tốt/xấu định tính

Phân loại lỗi thành 5 nhóm, mỗi nhóm 2 query minh hoạ kèm ảnh chụp kết quả và giải thích tại sao hỏng:

1. **Đếm số lượng** — "three dogs", "two people"
2. **Phủ định** — "a street with no cars"
3. **Quan hệ không gian** — "a cup to the left of a laptop"
4. **Thuộc tính mịn / chữ trong ảnh** — "a red sign that says STOP"
5. **Khái niệm tổ hợp hiếm** — "a person riding a zebra"

Nhóm phủ định và đếm số gần chắc chắn hỏng ở mọi model, đó là giới hạn đã biết của contrastive learning. Nêu được nó cho thấy hiểu giới hạn của phương pháp, không chỉ chạy được code. Kèm cả 3-4 ví dụ **tốt** để đối chiếu.

---

## 8. Frontend

Một trang, ba khu.

**Khu query** — ô nhập text và vùng kéo-thả ảnh (dán từ clipboard cũng được). Dropdown chọn model dựng từ `GET /spaces`. Panel "nâng cao" thu gọn: slider top-k, toggle `exact`, `hnsw_ef`, filter category.

**Khu kết quả** — grid ảnh responsive, badge điểm số trên mỗi ảnh, hover hiện caption. Bấm ảnh mở modal chi tiết có nút **"Tìm ảnh tương tự"** gọi `/search/image` với `image_id`. Nút này làm demo chiều ảnh→ảnh **không cần người chấm có file ảnh sẵn trong máy**, chi tiết nhỏ nhưng là khác biệt giữa demo mượt và demo bị kẹt.

**Chế độ so sánh** — cùng một query, chọn 2 model, kết quả hai cột cạnh nhau. Bảng số liệu nói "laion-b32 hơn 4 điểm R@1"; chế độ này cho người chấm *thấy* điều đó trên query của chính họ. Nó biến ablation từ một mục trong báo cáo thành phần demo được.

Chips query mẫu tiếng Việt hiển thị dưới ô nhập, để người chấm thử tiếng Việt ra kết quả tốt, nhưng chỉ khi chọn model multilingual, và đó chính là điều ta muốn họ tự phát hiện.

Vite + React + TS + Tailwind. Không inline style (CLAUDE.md). Hiện `latency_ms` và tên space trên mỗi lần tìm.

---

## 9. Cấu trúc project

```
assignment_1/
├── docker-compose.yml        # service qdrant
├── .env.example              # mọi tham số cấu hình
├── requirements.txt
├── tasks.py                  # CLI thay Makefile (Windows-friendly)
├── README.md                 # lệnh chạy lại từ đầu
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI routes
│   │   ├── config.py         # pydantic-settings
│   │   ├── schemas.py        # pydantic request/response
│   │   ├── registry.py       # interface duy nhất giữa model và hệ thống
│   │   ├── encoders/         # clip.py, siglip.py, mclip.py, resnet.py, bm25.py
│   │   ├── vectordb.py       # wrapper Qdrant (server + embedded fallback)
│   │   └── search.py         # điều phối: encode → query → format
│   ├── cli/{ingest,build_index,evaluate}.py
│   └── tests/
├── frontend/src/{App.tsx, components/, api/, hooks/}
├── data/                     # gitignore: ảnh COCO, thumbs, corpus.jsonl, cache/*.npy
│                             # commit: queryset_vi.json, queryset_i2i.json,
│                             #         queryset_short.json
├── results/                  # csv + bảng markdown do eval sinh ra
└── docs/report.md
```

Mỗi file một việc. `encoders/` tách riêng từng model vì đó là chỗ **duy nhất** có code phụ thuộc thư viện của model cụ thể.

---

## 10. Testing

Viết test trước (TDD). Chạy trên fixture 20 ảnh + Qdrant embedded in-memory nên toàn bộ suite xong trong vài giây.

**Unit**

- Registry trả đúng dim/distance/collection cho từng space.
- Hàm normalize cho vector có norm = 1 (sai số float chấp nhận được).
- Builder chuyển `filters` thành Qdrant `Filter` đúng cấu trúc.
- Validate schema `corpus.jsonl` (đủ field, kiểu đúng, caption không rỗng).
- Tokenizer BM25.

**Contract API** (FastAPI TestClient)

- Trả đúng `k` kết quả, score giảm dần, `rank` liên tục từ 1.
- **Mọi đường lỗi ở §3.6**: 400, 404, 409, 413, 422, 503. Đường lỗi được test vì đó chính là thứ hỏng lúc demo trước người chấm.
- Chặn path traversal ở `/images/{file_name}`.

**Kiểm chứng giả định** (đánh dấu `integration`, tải model thật)

- Cosine giữa `clip-b32` và `mclip-b32` trên cùng caption EN **> 0.9** (§5.3). Chạy **trước** khi build 25k vector.

**Tính tất định của eval**

- Cùng query set + cùng index → ra đúng cùng con số. Bảo vệ báo cáo khỏi việc số đổi giữa hai lần chạy.

**Smoke end-to-end**

- ingest fixture → build 1 index → search → eval, trên 20 ảnh.

---

## 11. Chạy lại từ đầu

```bash
docker compose up -d qdrant            # cần Docker Desktop đang chạy
pip install -r requirements.txt
cp .env.example .env
python tasks.py ingest                 # tải COCO, sinh corpus.jsonl + thumbs
python tasks.py build --space all      # ~35 phút CPU, một lần
python tasks.py eval --all             # sinh results/*.csv + *.md
python tasks.py serve                  # API
cd frontend && npm install && npm run dev
```

---

## 12. Báo cáo (`docs/report.md`, ~5 trang)

1. Bài toán và dataset
2. Kiến trúc hệ thống (kèm sơ đồ §3)
3. Phương pháp: lineup model và lý do chọn từng cặp so sánh (§5.2)
4. Kết quả định lượng: bảng chính + 5 bảng ablation
5. Phân tích tốt/xấu theo phân loại lỗi, kèm ảnh (§7.3)
6. Giới hạn và hướng mở rộng
7. Cách chạy lại

Mọi con số đọc từ `results/`, không chép tay.

---

## 13. Ngoài phạm vi

Những thứ **không** làm, để tránh phình scope:

- Deploy công khai (Hugging Face Spaces, Render). Có thể làm sau nếu muốn.
- Chiều text→text và hybrid BM25 + semantic cho corpus caption (BM25 chỉ dùng làm baseline t2i).
- Fine-tune model. Toàn bộ là zero-shot.
- Re-ranking bằng cross-encoder hay BLIP.
- Xác thực người dùng, lưu lịch sử tìm kiếm, phân tích hành vi.
- Dataset thứ hai (Flickr30k) để kiểm tra cross-domain.
- Dịch tự động VI→EN (đã chọn hướng model multilingual thay thế).

---

## 14. Rủi ro đã biết

| Rủi ro | Ảnh hưởng | Xử lý |
|---|---|---|
| Giả định `mclip-b32` chung không gian nhúng sai | Trục ablation 5 phải làm lại | Test kiểm chứng ở §5.3 chạy trước khi build |
| Docker Desktop không bật được trên máy chấm | Không chạy được trục ablation 4 | Fallback embedded cho demo; số liệu ANN đã sinh sẵn trong `results/` |
| Encode CPU chậm hơn ước lượng | Build index lâu hơn 35 phút | Build từng space độc lập và idempotent; có thể chạy qua đêm hoặc encode trên Colab rồi mang vector về |
| Proxy category cho i2i bị phản biện | Mất điểm phần đo lường | Nêu thẳng giới hạn và thiên vị của proxy trong báo cáo (§7.2) |
| Tải COCO val2017 chậm hoặc lỗi mạng | Chặn toàn bộ | `ingest` resume được, kiểm tra checksum, không tải lại file đã có |
