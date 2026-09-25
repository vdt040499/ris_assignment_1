# Báo cáo: Tìm kiếm ngữ nghĩa đa phương thức trên MS-COCO

**Ngày:** 2026-09-25
**Phạm vi:** Toàn bộ hệ thống (backend FastAPI + Qdrant, frontend React), chạy
thật trên corpus 5.000 ảnh COCO val2017, số liệu lấy từ `results/` và
`data/corpus.jsonl` — không có con số nào trong file này được gõ từ trí nhớ.

---

## 1. Bài toán và dataset

Bài toán: xây một hệ tìm kiếm ngữ nghĩa nhận vào **câu chữ tự do** hoặc **một
tấm ảnh**, trả về top-k ảnh liên quan nhất, cùng khả năng đánh giá định lượng
xem "liên quan nhất" đó tốt tới đâu.

Dataset: **MS-COCO val2017**. Số liệu đếm được trực tiếp từ
`data/corpus.jsonl` (nguồn sự thật duy nhất sau bước `ingest`, theo §3.1 của
spec thiết kế):

| Đại lượng | Giá trị đo được | Nguồn |
|---|---|---|
| Số ảnh | 5.000 | `data/corpus.jsonl` (5.000 dòng) |
| Số caption | 25.014 | tổng độ dài trường `captions` trên toàn bộ `data/corpus.jsonl` |
| Số ảnh không có category COCO nào | 48 | đếm số dòng có `categories: []` trong `data/corpus.jsonl` |

Con số 25.014 khớp với ghi chú ở §7.1 của spec: "không phải ảnh nào cũng có
đúng 5 caption" — annotation gốc của `captions_val2017.json` có ảnh với 4, 6,
7 caption thay vì đúng 5, nên tổng không tròn 25.000.

**Vì sao chọn dataset có ground-truth.** COCO val2017 gắn mỗi ảnh với 5 caption
người viết độc lập và nhãn category/supercategory chuẩn. Điều đó cho phép đo
định lượng thật (R@1/5/10, MRR@10 cho chiều text→ảnh dùng chính caption làm
ground-truth; P@k/mAP theo category cho chiều ảnh→ảnh) thay vì chỉ đánh giá
định tính bằng mắt. Một dataset không có nhãn nào sẽ không cho phép tính được
một con số Recall/MRR nào cả — toàn bộ trục ablation ở mục 4 sẽ không thực
hiện được.

Lưu ý phải nêu (và được nhắc lại ở mục 6): val2017 khác Karpathy test split
(split chuẩn dùng trong paper CLIP gốc), nên số liệu ở đây *gần* nhưng
**không so trực tiếp được** với số trong paper CLIP.

---

## 2. Kiến trúc hệ thống

Bốn thành phần, mỗi thành phần một nhiệm vụ, giao tiếp qua interface rõ ràng
(nguyên văn sơ đồ ở §3 của spec thiết kế,
`docs/superpowers/specs/2026-09-23-multimodal-search-design.md`):

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

- **`ingest`** (`backend/cli/ingest.py`, offline, một lần): tải COCO val2017,
  sinh thumbnail 256px, xuất một file `data/corpus.jsonl` duy nhất. Sau bước
  này không thành phần nào đọc lại COCO JSON gốc.
- **`build_index`** (`backend/cli/build_index.py`, offline, mỗi space một
  lần): encode ảnh theo batch trên CPU, upsert vào collection Qdrant, ghi
  `data/index_meta/<space>.json` — báo cáo lấy thời gian build và số point từ
  đúng file này (mục 4), không chép tay.
- **`search-api`** (`backend/app/main.py` + `search.py`, online): nhận query
  text hoặc ảnh, chọn encoder theo `space`, gọi Qdrant `query_points`, trả
  top-k kèm score và payload. Encoder nạp lười, cache LRU tối đa
  `MODEL_CACHE_SIZE` model.
- **`eval`** (`backend/cli/evaluate.py`, offline CLI): đọc `corpus.jsonl`,
  encode query set, truy vấn **cùng một Qdrant mà demo đang dùng**, xuất
  `results/*.csv` và `*.md`. Đây là lý do số liệu báo cáo và kết quả demo
  không lệch nhau — chúng đến từ cùng một index.

**Vai trò của registry** (`backend/app/registry.py`): đây là interface duy
nhất giữa "phần model" và "phần hệ thống". Mỗi space khai báo một `SpaceSpec`
(`name`, `hf_id`, `dim`, `distance`, `normalized`, `backend`,
`collection_suffix`, `modes`, `languages`, `encoder_key`, `builds_index`).
`search.py`, API và eval CLI chỉ biết tới `SpaceSpec`, không biết CLIP hay
SigLIP cụ thể là gì. Thêm một model = thêm một entry trong registry, không sửa
API/frontend/eval — đây là lý do ablation 5 model + 2 baseline trong mục 3, 4
rẻ về công sức thay vì phải copy-paste 5 lần.

---

## 3. Phương pháp

### 3.1 Năm không gian nhúng + hai baseline

| Space | Model | Dim | Biến được thay đổi | Chiều hỗ trợ |
|---|---|---|---|---|
| `clip-b32` | `openai/clip-vit-base-patch32` | 512 | **baseline** | t2i, i2i, i2t |
| `clip-b16` | `openai/clip-vit-base-patch16` | 512 | kích thước patch (độ phân giải hiệu dụng) | t2i, i2i |
| `laion-b32` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 512 | dữ liệu huấn luyện (LAION-2B vs WIT-400M) | t2i, i2i |
| `siglip-b16` | `google/siglip-base-patch16-224` | 768 | hàm loss (sigmoid vs softmax contrastive) | t2i, i2i |
| `mclip-b32` | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | 512 | text tower đa ngôn ngữ, dùng lại collection ảnh của `clip-b32` | t2i (đa ngữ) |
| `resnet50` | `microsoft/resnet-50` (ImageNet) | 2048 | baseline không multimodal | i2i |
| `bm25-cap` | BM25 trên toàn bộ caption | — | baseline không ngữ nghĩa | t2i |

(Registry còn một space kỹ thuật thứ tám, `clip-b32-raw` — cùng model
`clip-b32` nhưng vector không normalize, distance `Dot` — dùng riêng cho
ablation normalize ở mục 4.3, không phải một "model" độc lập nên không tính
vào bảng bảy dòng trên.)

### 3.2 Vì sao đúng bộ này — mỗi cặp cô lập đúng một biến

Đây là phần quan trọng nhất của thiết kế thực nghiệm (§5.2 spec): việc chọn
đúng các cặp so sánh khiến bảng kết quả ở mục 4 diễn giải được, thay vì chỉ là
một bảng xếp hạng vô nghĩa.

| Cặp so sánh | Biến cô lập | Câu hỏi trả lời |
|---|---|---|
| `clip-b32` vs `laion-b32` | dữ liệu huấn luyện (cùng kiến trúc ViT-B/32) | Dữ liệu huấn luyện lớn hơn/đa dạng hơn (LAION-2B) có giúp gì so với WIT-400M gốc của OpenAI không? |
| `clip-b32` vs `clip-b16` | kiến trúc/độ phân giải (cùng dữ liệu WIT) | Patch nhỏ hơn (độ phân giải hiệu dụng cao hơn) đáng giá bao nhiêu? |
| `clip-b16` vs `siglip-b16` | hàm loss (cùng cỡ patch 16) | Sigmoid loss (SigLIP) có thật sự tốt hơn softmax contrastive (CLIP) không? |
| `clip-b32` vs `resnet50` (i2i) | multimodal contrastive vs supervised ImageNet feature | CLIP có thật "hơn" feature ImageNet thuần cho tác vụ tìm ảnh tương tự không? |
| `clip-b32` vs `bm25-cap` (t2i) | biểu diễn ngữ nghĩa (dense) vs khớp từ khoá (sparse) | Ngữ nghĩa có thật hơn khớp từ khoá không, và hơn bao nhiêu? |
| `clip-b32` vs `mclip-b32` | text tower đơn ngữ vs đa ngôn ngữ | Đánh đổi gì khi thêm khả năng đa ngôn ngữ? |

**Giả định phải kiểm chứng trước khi build:** `mclip-b32` được distill để nằm
cùng không gian nhúng với `clip-b32`, nên nó **dùng lại đúng collection ảnh**
`coco_clip_b32` thay vì encode lại 5.000 ảnh — registry khai báo
`builds_index=False` cho space này (`backend/app/registry.py`). Giả định đó
được kiểm chứng bằng test tích hợp
`backend/tests/test_encoders_integration.py::test_multilingual_shares_the_clip_space`
(ngưỡng chặn: cosine > 0.9 giữa vector text của `clip-b32` và `mclip-b32` trên
cùng câu `"a man riding a horse on the beach"`). Chạy lại phép đo này (bằng
đúng hai encoder thật, không phải số ước lượng) cho **cosine đo được ≈
0.9753** — vượt xa ngưỡng 0.9, xác nhận giả định đúng. Test tương ứng
(`pytest -m integration`) chạy PASS. *(Con số 0.9753 không nằm trong một file
`results/*.md` sẵn có vì đây là một phép kiểm chứng giả định thiết kế — không
phải một chỉ số ablation — nhưng nó được đo lại trực tiếp bằng đúng code
encoder của dự án, không gõ từ trí nhớ hay ước lượng.)*

---

## 4. Kết quả định lượng

Toàn bộ bảng dưới đây là **nguyên văn** từ `results/axis*.md`, sinh bởi
`python tasks.py eval --axis all` chạy thật trên Qdrant server với dữ liệu đầy
đủ 5.000 ảnh / 25.014 caption.

### 4.1 Trục 1a — Model, text→ảnh (`results/axis1_model_t2i.md`)

5.000 caption làm query, tìm trên 5.000 ảnh (giao thức COCO 5k chuẩn).

| space | n_queries | exact | hnsw_ef | prompt_template | R@1 | R@5 | R@10 | MRR@10 | search_ms_p50 | search_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 | 5000 | True | None |  | 0.3138 | 0.5536 | 0.659 | 0.4152 | 3.367 | 49.237 |
| clip-b16 | 5000 | True | None |  | 0.328 | 0.5786 | 0.6802 | 0.4356 | 3.347 | 49.45 |
| laion-b32 | 5000 | True | None |  | 0.3948 | 0.6488 | 0.7544 | 0.5041 | 3.512 | 49.383 |
| siglip-b16 | 5000 | True | None |  | 0.479 | 0.7262 | 0.8112 | 0.5837 | 3.907 | 49.699 |
| bm25-cap | 5000 | True | None |  | 0.9874 | 0.9988 | 0.9998 | 0.9922 | 42.975 | 62.145 |

**Đọc bảng.** SigLIP-B16 vượt xa cả ba biến thể CLIP (R@1 0.479 so với 0.3138
của baseline, +16.5 điểm), xác nhận hàm loss sigmoid contrastive học biểu diễn
text-ảnh khớp tốt hơn softmax contrastive ở cùng thang huấn luyện. LAION-B32
hơn CLIP-B32 gần 8 điểm R@1 (0.3948 vs 0.3138) dù kiến trúc y hệt — hiệu ứng
thuần của việc huấn luyện trên LAION-2B thay vì WIT-400M. Con số BM25 (R@1
0.9874) trông "thắng" mọi model ngữ nghĩa nhưng đây là baseline **không công
bằng theo thiết kế**: BM25 tìm trực tiếp trên chính 25.014 caption gốc (chính
là ground-truth), nên gần như luôn tìm lại đúng câu query — nó đo khả năng
khớp từ khoá trên văn bản, không đo khả năng nối ảnh với ngôn ngữ tự nhiên như
CLIP-family, hai con số không nên bị đọc như hai lựa chọn ngang hàng cho cùng
một bài toán.

### 4.2 Trục 1b — Model, ảnh→ảnh, proxy category (`results/axis1_model_i2i.md`)

200 ảnh query (seed cố định, `data/queryset_i2i.json`), P@10/mAP@10 tính theo
proxy "chia sẻ ít nhất một category COCO".

| space | n_queries | P@10 | mAP@10 | self_hits | search_ms_p50 | search_ms_p95 |
|---|---|---|---|---|---|---|
| clip-b32 | 200 | 0.8805 | 0.9071 | 200 | 4.153 | 7.097 |
| clip-b16 | 200 | 0.878 | 0.9116 | 200 | 4.977 | 8.353 |
| laion-b32 | 200 | 0.8855 | 0.9152 | 200 | 3.998 | 9.377 |
| siglip-b16 | 200 | 0.923 | 0.941 | 200 | 4.969 | 7.703 |
| resnet50 | 200 | 0.8845 | 0.9078 | 200 | 8.167 | 11.524 |

**Đọc bảng.** SigLIP-B16 vẫn dẫn đầu cả ở chiều ảnh→ảnh (P@10 0.923). Điểm
đáng chú ý nhất: **ResNet-50 thuần supervised-ImageNet (P@10 0.8845) gần như
ngang bằng CLIP-B32 multimodal (0.8805)** trên chính proxy đo lường này — tức
là với thước đo "chia sẻ category", một feature ImageNet cổ điển không hề thua
kém CLIP. Đây không có nghĩa CLIP "vô dụng" cho i2i; nó có nghĩa proxy category
thiên vị đặc điểm bố cục/vật thể nổi bật mà cả hai loại feature đều nắm được
tốt — phần Giới hạn (mục 6) nêu rõ điểm yếu đo lường này. `self_hits = 200` ở
mọi space (đúng bằng `n_queries`) xác nhận mỗi ảnh query luôn là láng giềng
gần nhất tuyệt đối của chính nó trong ranking thô, tức cache vector dùng để
tạo query khớp đúng với vector đã nạp vào collection. Giá trị lành mạnh của
cột này luôn là `n_queries`, không phải 0 — một giá trị gần 0 mới là dấu hiệu
cache và collection lệch nhau, cần kiểm tra trước khi tin P@k/mAP@k.

### 4.3 Trục 2 — Normalize on/off (`results/axis2_normalize.md`)

`clip-b32` (cosine trên vector đã L2-normalize) vs `clip-b32-raw` (dot trên
vector thô).

| space | n_queries | exact | hnsw_ef | prompt_template | R@1 | R@5 | R@10 | MRR@10 | search_ms_p50 | search_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-b32 | 5000 | True | None |  | 0.3138 | 0.5536 | 0.659 | 0.4152 | 3.357 | 49.089 |
| clip-b32-raw | 5000 | True | None |  | 0.2636 | 0.493 | 0.603 | 0.3618 | 3.353 | 48.159 |

**Đọc bảng.** Bỏ normalize làm R@1 giảm 5 điểm tuyệt đối (0.3138 → 0.2636,
tương đương giảm ~16% tương đối) và MRR@10 giảm từ 0.4152 xuống 0.3618. Vector
CLIP thô không có norm đồng nhất giữa các ảnh, nên dot product bị trộn lẫn
giữa "độ tương đồng ngữ nghĩa thật" và "độ lớn vector" — cosine (tương đương
dot trên vector đã chuẩn hoá) loại bỏ đúng nhiễu này. Kết luận: normalize
không phải chi tiết kỹ thuật phụ, nó đóng góp một phần đáng kể vào chất lượng
tìm kiếm.

### 4.4 Trục 3 — Prompt template (`results/axis3_prompt.md`)

Đo trên hai bộ query khác nhau theo đúng lưu ý ở §6.2 spec: (a) 5.000 caption
đầy đủ (R@k/MRR), và (b) 80 query ngắn kiểu từ khoá (`data/queryset_short.json`
lấy mẫu từ các category có mặt trong tập dữ liệu, dùng P@10).

| space | n_queries | prompt_template | R@1 | R@5 | R@10 | MRR@10 | search_ms_p50 | P@10 |
|---|---|---|---|---|---|---|---|---|
| clip-b32 | 5000 | (query thô) | 0.3138 | 0.5536 | 0.659 | 0.4152 | 3.382 |  |
| clip-b32 | 80 | (query thô) |  |  |  |  | 3.435 | 0.8012 |
| clip-b32 | 5000 | `a photo of {}` | 0.309 | 0.5528 | 0.6626 | 0.4145 | 3.425 |  |
| clip-b32 | 80 | `a photo of {}` |  |  |  |  | 3.416 | 0.7962 |
| clip-b32 | 5000 | `a photo of {}, a type of scene` | 0.2866 | 0.5304 | 0.6438 | 0.3905 | 3.434 |  |
| clip-b32 | 80 | `a photo of {}, a type of scene` |  |  |  |  | 3.571 | 0.7812 |

**Đọc bảng.** Trên caption dài (5.000 query), bọc prompt template gần như
không đổi R@1 (0.3138 → 0.309, giảm nhẹ) và làm giảm rõ khi thêm cụm dài hơn
(`"a type of scene"`: R@1 xuống 0.2866). Trên query ngắn (80 query), xu hướng
là **template thuần túy không giúp mà còn hại nhẹ**: P@10 giảm dần từ 0.8012
(query thô) → 0.7962 (`a photo of {}`) → 0.7812 (template dài nhất). Điều này
khác với kỳ vọng gốc của prompt engineering trong CLIP-family (thường được
thiết kế cho zero-shot classification trên tên lớp rất ngắn như "dog"/"cat"
đơn từ, không phải các cụm 2-3 từ như "stop sign"/"pizza" mà `queryset_short`
sử dụng) — kết luận đúng ở đây là **prompt template không đáng dùng cho corpus
và bộ query cụ thể này**, không phải "prompt engineering luôn vô dụng" nói
chung.

### 4.5 Trục 4 — ANN (HNSW) vs exact (`results/axis4_ann.md`)

**Sự cố đã sửa trước khi đọc bảng này.** Bản trước của mục này đo trên một
collection mà Qdrant **chưa từng build HNSW thật**: `indexed_vectors_count`
trên server sống là `0` cho cả 7 collection (`optimizer_config.indexing_threshold`
mặc định là 20.000 KB, trong khi 5.000 vector 512-chiều chia trên 8 segment chỉ
~1,25MB/segment — không bao giờ chạm ngưỡng), nên mọi truy vấn gắn nhãn "ANN"
thực chất là full-scan so với chính nó (exact đội lốt ANN). Đã sửa theo hai
bước:
1. `VectorStore.ensure_collection` (`backend/app/vectordb.py`) nay tạo
   collection mới với `optimizers_config=OptimizersConfigDiff(indexing_threshold=1)`.
   **Lưu ý:** giá trị đúng là `1`, không phải `0` — theo đúng docstring của
   chính field này trong `qdrant-client` ("To disable vector indexing, set to
   0"), `0` là sentinel **tắt hẳn** indexing, xác nhận thực nghiệm bằng cách đặt
   0 rồi đợi 10+ phút: `indexed_vectors_count` đứng yên ở 0 và
   `/telemetry` báo `optimizations.count = 0` cho mọi collection. `1` (KB) nhỏ
   hơn một vector 512-chiều nên mọi segment có dữ liệu đều vượt ngưỡng ngay.
2. 7 collection cũ (build trước khi có default trên) được ép build lại HNSW
   bằng `client.update_collection(name, optimizer_config=OptimizersConfigDiff(indexing_threshold=1))`,
   xác nhận xong khi `indexed_vectors_count == points_count` cho cả 7 (bao gồm
   `coco_cap_clip_b32`, 25.014 point) — kiểm bằng `GET /collections/<name>` và
   bằng chính có tồn tại thư mục `vector_index/` (chứa `hnsw_config.json`)
   trong từng segment trên đĩa.
3. `eval_ann_sweep` (`backend/cli/evaluate.py`) nay raise `RuntimeError` ngay
   nếu `indexed_vectors_count != points_count` của collection — lỗi này không
   còn có thể tái diễn một cách âm thầm. Guard này đã được xác nhận bắt đúng
   lỗi trên chính server thật, ngay trước khi sửa xong bước 2 ở trên (log lỗi
   thật, không phải giả lập):
   `RuntimeError: Collection 'coco_clip_b32' chưa build xong HNSW (indexed_vectors_count=0/5000)...`

Sweep `hnsw_ef` ∈ {16, 64, 128, 256} trên 5.000 vector thật đã index xong
(`indexed_vectors_count = 5000 = points_count`), `k=10`, so với baseline exact.

| space | hnsw_ef | overlap@10 | R@1 | search_ms_p50 | search_ms_p95 |
|---|---|---|---|---|---|
| clip-b32 | 16 | 1.0 | 0.3138 | 3.303 | 49.228 |
| clip-b32 | 64 | 1.0 | 0.3138 | 3.347 | 49.578 |
| clip-b32 | 128 | 1.0 | 0.3138 | 3.341 | 49.317 |
| clip-b32 | 256 | 1.0 | 0.3138 | 3.412 | 49.373 |

**Đọc bảng — số liệu không đổi so với trước, nhưng giờ có ý nghĩa thật.**
`overlap@10 = 1.0` ở mọi `hnsw_ef` và latency không đổi theo `ef`, **giống hệt**
kết quả đo được khi collection còn chưa index — nhưng lần này con số phản ánh
đúng cái nó tuyên bố đo: HNSW *đã tồn tại thật* (`indexed_vectors_count =
points_count`, xác nhận độc lập bằng file `vector_index/hnsw_config.json` trên
đĩa) và truy vấn với `exact=False` vẫn cho kết quả giống hệt exact. Lý do kỹ
thuật xác định được, không phải phỏng đoán: collection này có 8 segment, mỗi
segment ~625 point (5.000/8), trong khi
`hnsw_config.full_scan_threshold = 10.000` (mặc định) — tức Qdrant tự chọn
full-scan cho bất kỳ segment nào có ít hơn 10.000 point, **bất kể** cờ
`exact`/`hnsw_ef` truyền vào. Với 625 « 10.000, cả 8 segment đều rơi vào
nhánh full-scan này, nên dù đồ thị HNSW có tồn tại, nó không được dùng để trả
lời truy vấn ở quy mô hiện tại. Đây là một tham số **khác** với
`indexing_threshold` đã sửa ở trên (một cái quyết định có *build* đồ thị hay
không, cái kia quyết định có *dùng* đồ thị đã build khi tìm hay không) —
không nằm trong phạm vi sửa của C1 (thay đổi nó sẽ ảnh hưởng hành vi search
mặc định của toàn hệ, ngoài phạm vi finding này).

Kết luận **thực chất không đổi so với bản trước, nhưng nay được xác minh đúng
cách thay vì dựa trên một index chưa từng tồn tại**: ở quy mô 5.000 vector
chia trên 8 segment nhỏ như thiết lập hiện tại của dự án, ANN không mang lại
khác biệt nào so với exact — không phải vì "corpus nhỏ nên hai thuật toán tình
cờ giống nhau", mà vì cơ chế heuristic full-scan-cho-segment-nhỏ của chính
Qdrant khiến engine dùng full-scan cho cả hai trường hợp. ANN sẽ có lý do tồn
tại ở quy mô lớn hơn nhiều (hàng triệu vector, đủ để mỗi segment vượt
`full_scan_threshold`), ngoài phạm vi corpus này.

### 4.6 Trục 5 — Ngôn ngữ, Anh vs Việt (`results/axis5_language.md`)

200 nội dung giống hệt nhau, một bản tiếng Anh gốc và một bản dịch tay sang
tiếng Việt (`data/queryset_vi.json`), chạy trên cả `clip-b32` (đơn ngữ) và
`mclip-b32` (đa ngôn ngữ).

| space | n_queries | R@1 | R@5 | R@10 | MRR@10 | language |
|---|---|---|---|---|---|---|
| clip-b32 | 200 | 0.255 | 0.51 | 0.67 | 0.3663 | en |
| clip-b32 | 200 | 0.0 | 0.025 | 0.03 | 0.0093 | vi |
| mclip-b32 | 200 | 0.195 | 0.435 | 0.62 | 0.3045 | en |
| mclip-b32 | 200 | 0.13 | 0.34 | 0.445 | 0.2134 | vi |

**Đọc bảng.** `clip-b32` **sụp hoàn toàn** trên tiếng Việt: R@1 rơi từ 0.255
(EN) xuống **0.0** (VI), R@10 chỉ còn 0.03 — text tower của CLIP gốc chỉ được
huấn luyện trên tiếng Anh nên gần như không mã hoá được ngữ nghĩa tiếng Việt.
`mclip-b32` giữ được phần lớn khả năng trên tiếng Việt (R@1 0.13, R@10 0.445 —
vẫn thấp hơn EN nhưng khác một trời một vực so với baseline sụp đổ), đổi lại
**mất khoảng 6 điểm R@1 trên chính tiếng Anh** so với `clip-b32` (0.195 vs
0.255) — đây chính là đánh đổi đa ngôn ngữ mà thiết kế dự đoán trước ở §6.2
spec: "baseline sụp trên VI; multilingual mất vài điểm trên EN".

### 4.7 Thời gian build index (`data/index_meta/*.json`)

| Space | encode_seconds | n_points | Ghi chú |
|---|---|---|---|
| `clip-b32-raw` | 236.62s (~3.9 phút) | 5.000 | encode **từ đầu** cho kiến trúc CLIP-B32 trên CPU (image encoder chạy trước, xem ghi chú dưới) |
| `clip-b32` | 9.69s | 5.000 | **không phải encode từ đầu** — build order (`DERIVED_FROM` trong `backend/cli/build_index.py`) chạy `clip-b32-raw` trước, sau đó `clip-b32` tái sử dụng đúng vector ảnh đã encode (chỉ L2-normalize + upsert lại) |
| `clip-b16` | 709.11s (~11.8 phút) | 5.000 | encode từ đầu, patch nhỏ hơn → chậm hơn rõ rệt so với patch32 |
| `laion-b32` | 270.42s (~4.5 phút) | 5.000 | encode từ đầu, cùng kiến trúc patch32 nên thời gian gần với `clip-b32-raw` |
| `siglip-b16` | 682.41s (~11.4 phút) | 5.000 | encode từ đầu |
| `resnet50` | 21.04s | 5.000 | **không dùng được để so sánh** — xem ghi chú riêng bên dưới |
| `bm25-cap` | 0.22s | 5.000 | không encode neural, chỉ dựng inverted index từ khoá |

**Ghi chú quan trọng về `resnet50`.** Con số 21.04s trong
`data/index_meta/resnet50.json` là kết quả của một lần build lại bằng
`--force` **tái sử dụng vector cache đã có sẵn từ file**, không phải thời gian
encode 5.000 ảnh từ đầu bằng ResNet-50. Lần build gốc (encode thật từ đầu) đã
crash giữa chừng lúc upsert do sự cố hết dung lượng đĩa thật trong quá trình
phát triển dự án, nên không còn metadata hợp lệ của lần chạy from-scratch đó.
Vì vậy con số 21.04s **không được dùng để so sánh tốc độ encode giữa các
model** trong báo cáo này — nó chỉ xác nhận `resnet50` đã build xong với đủ
5.000 point.

**Tổng thời gian encode from-scratch có thể so sánh được** (không tính
`clip-b32` vì là dẫn xuất, không tính `resnet50` vì không hợp lệ):
236.62 + 709.11 + 270.42 + 682.41 + 0.22 ≈ **1.899 giây (~31.6 phút)**, khớp
tương đối với ước lượng ~35 phút ở §5.4 spec (ước lượng đó gồm cả phần encode
caption cho collection ảnh→text, không tách riêng trong `index_meta` hiện có).

---

## 5. Phân tích tốt/xấu

Phân tích định tính đầy đủ (14 query — 10 lỗi + 4 tốt, mỗi query kèm hình và
giải thích cơ chế hỏng/thành công dựa trên caption/category thật trả về từ
API) nằm ở **`results/qualitative.md`**. Tóm tắt bảng 5 nhóm lỗi:

| Nhóm lỗi | Trung bình đúng ý /10 |
|---|---|
| Đếm số lượng | 2.0 |
| Phủ định | 1.0 |
| Quan hệ không gian | 0.5 |
| Thuộc tính mịn / chữ trong ảnh | 4.5 (bị kéo lên bởi may mắn ở "STOP sign", xem giải thích dưới) |
| Tổ hợp hiếm | 0.0 |

Đối chiếu, 4 query tốt trung bình 5.75/10.

Bốn hình tiêu biểu (xem đầy đủ 14 hình tại `results/figures/`):

![three dogs](../results/figures/bad_count_three_dogs.png)
*"three dogs" — 2/10 đúng ý. Toàn bộ 10 ảnh đều có chó, nhưng số lượng caption
xác nhận dao động 1–5+ con. CLIP không có tín hiệu contrastive nào buộc phân
biệt "three" với "two" hay "a pack of" — các câu này gần như cùng một vùng
vector.*

![a street with no cars](../results/figures/bad_negation_no_cars.png)
*"a street with no cars" — 0/10 đúng ý. Mọi kết quả trả về đều **có** category
`car`/`bus`/`truck` thật — phủ định "no" bị tokenizer/encoder xử lý gần như bỏ
qua, nên top-10 lại chính là các ảnh có nhiều xe nhất về mặt thị giác.*

![a person riding a zebra](../results/figures/bad_rare_riding_zebra.png)
*"a person riding a zebra" — 0/10 đúng ý. Tổ hợp này không tồn tại ảnh thật
tương ứng trong 5.000 ảnh COCO; mô hình bỏ qua động từ "riding" và trả về ảnh
gần nhất theo danh từ nổi bật nhất ("zebra").*

![a red sign that says STOP](../results/figures/bad_ocr_stop_sign.png)
*"a red sign that says STOP" — 9/10 đúng ý, nhưng **không phải nhờ đọc được
chữ**: CLIP nhận diện đúng hình bát giác đỏ đặc trưng của biển stop (một đặc
điểm thị giác, không phải OCR), trùng khớp với category `stop sign` của COCO.*

**Nhận định quan trọng nhất** (chi tiết đầy đủ ở cuối `results/qualitative.md`):
nhóm **Tổ hợp hiếm** hỏng tuyệt đối (0.0/10) vì bản thân COCO không có ảnh
đúng nghĩa cho các tổ hợp này; nhóm **Quan hệ không gian** (0.5/10) đáng lo
hơn về mặt phương pháp vì các vật thể liên quan đều tồn tại đơn lẻ trong
corpus — cái hỏng là do CLIP không mã hoá quan hệ không gian trong vector toàn
cục, một giới hạn kiến trúc thật sự chứ không phải do thiếu dữ liệu.

---

## 6. Giới hạn và hướng mở rộng

**Giới hạn đo lường ảnh→ảnh (proxy category).** P@k/mAP ở mục 4.2 dùng proxy
"chia sẻ ít nhất một category COCO" vì không có ground-truth "ảnh nào giống ảnh
nào" thật. Proxy này **thiên vị ảnh nhiều object**: một ảnh có 8-9 category
(như nhiều ví dụ trong `results/qualitative.md`) có xác suất trùng ít nhất một
category với ảnh khác cao hơn hẳn một ảnh chỉ có 1 category, bất kể mức độ
giống nhau thật về mặt thị giác/ngữ nghĩa. Đây là lý do ResNet-50 (mục 4.2) đo
được gần ngang CLIP dù trực giác cho rằng CLIP nên "hiểu ảnh" tốt hơn nhiều.

**Giới hạn lấy mẫu text→ảnh.** Trục 1 (t2i) dùng toàn bộ 25.014 caption của
5.000 ảnh này — tức là chỉ so khớp trong phạm vi tập dữ liệu đã ingest, không
phải toàn bộ phân bố ảnh COCO/Internet. Con số R@k đo được là "khả năng tìm
lại đúng ảnh trong một tập 5.000 ảnh đã biết trước", không nên suy rộng thành
"khả năng tìm ảnh đúng trong một kho ảnh mở bất kỳ".

**val2017 khác Karpathy test split.** Paper CLIP gốc và nhiều benchmark
retrieval chuẩn dùng Karpathy test split (một tập con 5.000 ảnh khác, chọn
theo cách khác từ COCO). Số liệu trong báo cáo này **không so sánh trực tiếp
được** với số trong paper CLIP dù cùng đơn vị đo (R@1/5/10) — khác tập ảnh,
khác phân bố caption.

**Bộ query ngắn chỉ phủ 80 category, không phải 200.** `data/queryset_short.json`
dùng cho trục 3(b) lấy mẫu 80 category thực có mặt trong 5.000 ảnh corpus,
không phải 200 category đầy đủ của COCO — một số category hiếm không xuất
hiện đủ trong tập 5.000 ảnh để tạo query có ý nghĩa.

**Giới hạn phương pháp lộ ra ở mục 5.** Đếm số lượng, phủ định và quan hệ
không gian đều hỏng nặng — đây là giới hạn đã biết của contrastive learning
trên vector toàn cục (không phải lỗi cài đặt), vì hàm mục tiêu huấn luyện chỉ
tối ưu độ tương đồng ảnh-caption tổng thể, không có tín hiệu nào buộc mô hình
mã hoá số lượng, phủ định hay quan hệ không gian tương đối giữa các vật thể.

**Hướng mở rộng:**
1. **Re-ranking bằng cross-encoder** (hoặc BLIP-2/GIT) trên top-50 kết quả sơ
   bộ từ CLIP-family, đặc biệt để cải thiện nhóm quan hệ không gian và thuộc
   tính hợp thành — cross-encoder có thể "nhìn" cả ảnh và câu cùng lúc thay vì
   so hai vector độc lập.
2. **Fine-tune trên domain cụ thể** nếu ứng dụng thực tế hẹp hơn COCO tổng
   quát (vd chỉ ảnh sản phẩm, chỉ ảnh y tế) — zero-shot CLIP không tối ưu cho
   domain hẹp.
3. **Dataset thứ hai để kiểm tra cross-domain** (vd Flickr30k, đã bị loại khỏi
   phạm vi ở §13 spec vì chi phí) — xác nhận các kết luận ở mục 4 (đặc biệt
   thứ hạng model) có giữ nguyên trên phân bố ảnh khác COCO hay không.

---

## 7. Cách chạy lại

Xem `README.md` ở thư mục gốc — mục "Chạy lại từ đầu" liệt kê đầy đủ chuỗi
lệnh từ `docker compose up -d qdrant` tới `npm run dev`, đã được xác minh khớp
với `tasks.py`/`docker-compose.yml`/`frontend/package.json` hiện tại của dự án.
