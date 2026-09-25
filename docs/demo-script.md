# Kịch bản quay video demo (2-3 phút)

**Đây là một kịch bản/shot-list cho người thật tự quay, KHÔNG PHẢI một video
đã quay sẵn.** Môi trường thực hiện Task 18 không có công cụ ghi màn hình, và
việc điều khiển một UI trực tiếp trong 2-3 phút không phải việc một agent văn
bản có thể làm. Không có file video nào tồn tại kèm theo repo này. File này
thay thế bước quay bằng một kịch bản chi tiết, đúng 7 nhịp theo yêu cầu của
brief Task 18, để người dùng có thể tự quay bằng phần mềm ghi màn hình bất kỳ
(OBS Studio, phần mềm ghi màn hình có sẵn của Windows Game Bar `Win+G`, v.v.)
mà không phải tự nghĩ lại kịch bản.

**Chuẩn bị trước khi bấm ghi:**

1. `docker compose up -d qdrant` — chờ `curl http://localhost:6333/readyz`
   trả `all shards are ready`.
2. Terminal 1: `python tasks.py serve` — chờ log xác nhận `Uvicorn running on
   http://127.0.0.1:8000`.
3. Terminal 2: `cd frontend && npm run dev` — chờ log xác nhận
   `Local: http://localhost:5173/`.
4. Mở `http://localhost:5173` trong trình duyệt, để cửa sổ đủ lớn để grid kết
   quả hiện rõ khi quay.
5. Chuẩn bị sẵn một file ảnh bất kỳ trên máy (ví dụ một ảnh chụp màn hình) để
   dùng ở nhịp 3 (kéo-thả ảnh).
6. Bắt đầu ghi màn hình + audio (giọng nói) trước khi làm nhịp 1.

---

## Nhịp 1 — Query text tiếng Anh → kết quả tốt (~15 giây)

- Gõ vào ô tìm kiếm chính: **`a man riding a horse on the beach`**
- Đảm bảo dropdown model đang là **`clip-b32`** (mặc định).
- Bấm nút tìm kiếm (hoặc Enter).
- Nói: "Đây là một query tiếng Anh mô tả cảnh thông thường. Hệ trả về ảnh
  người cưỡi ngựa trên bãi biển, đúng ý gần như toàn bộ top-10."
- Chỉ vào badge `latency_ms` và tên space hiển thị trên kết quả.

## Nhịp 2 — Tiếng Việt trên `mclip-b32`, rồi lặp lại trên `clip-b32` để thấy sụp (~30 giây)

- Đổi dropdown model sang **`mclip-b32`**.
- Xoá ô tìm kiếm, gõ: **`một con mèo đang ngủ trên giường`** (có sẵn trong
  chip query mẫu tiếng Việt — có thể bấm thẳng vào chip thay vì gõ tay).
- Bấm tìm kiếm. Nói: "Với model đa ngôn ngữ, câu tiếng Việt này trả về đúng
  cảnh mèo ngủ trên giường."
- Đổi dropdown model sang **`clip-b32`**, KHÔNG đổi nội dung ô tìm kiếm.
- Bấm tìm kiếm lại. Nói: "Cùng một câu tiếng Việt, đổi sang model chỉ huấn
  luyện tiếng Anh — kết quả sụp hoàn toàn, không còn liên quan gì tới mèo hay
  giường nữa. Số liệu ablation trong `results/axis5_language.md` xác nhận
  điều này: R@1 của `clip-b32` trên 200 query tiếng Việt là 0.0."

## Nhịp 3 — Kéo một ảnh từ máy vào → ảnh→ảnh (~20 giây)

- Đổi dropdown model về lại **`clip-b32`**.
- Kéo file ảnh đã chuẩn bị sẵn (bước 5 phần Chuẩn bị) vào vùng kéo-thả ảnh của
  UI (hoặc dùng nút chọn file nếu kéo-thả không tiện khi quay).
- Bấm tìm kiếm. Nói: "Đây là chiều ảnh→ảnh — thay vì gõ chữ, tôi đưa thẳng một
  tấm ảnh, hệ trả về những ảnh có nội dung thị giác gần nhất trong 5.000 ảnh
  COCO."

## Nhịp 4 — Bấm một ảnh kết quả → modal → "Tìm ảnh tương tự" (~20 giây)

- Trong grid kết quả đang hiện (từ nhịp 3 hoặc nhịp 1), bấm vào một ảnh bất kỳ
  để mở modal chi tiết.
- Chỉ vào caption và category hiện trong modal.
- Bấm nút **"Tìm ảnh tương tự"**.
- Nói: "Nút này gọi `/search/image` bằng chính `image_id` của ảnh đang xem —
  không cần người xem có sẵn file ảnh nào trên máy, chỉ cần bấm một cái."

## Nhịp 5 — Chế độ so sánh `clip-b32` vs `laion-b32` (~25 giây)

- Chuyển sang chế độ so sánh (compare view) trong UI.
- Chọn cột 1 = **`clip-b32`**, cột 2 = **`laion-b32`**.
- Gõ một query chung, ví dụ: **`a plate of pizza on a wooden table`**.
- Bấm tìm kiếm. Nói: "Cùng một query, hai model cạnh nhau. Bảng ablation
  `results/axis1_model_t2i.md` nói `laion-b32` hơn `clip-b32` gần 8 điểm R@1
  (0.3948 so với 0.3138) — ở đây có thể thấy trực tiếp trên chính query của
  mình, không cần đọc bảng số."

## Nhịp 6 — Tắt `exact`, đặt `hnsw_ef=16`, so sánh latency/kết quả (~20 giây)

- Mở panel "nâng cao" (advanced).
- Tắt toggle **`exact`**.
- Đặt **`hnsw_ef = 16`**.
- Bấm tìm kiếm lại với cùng query ở nhịp 5 (hoặc một query bất kỳ).
- Nói: "Tắt exact và hạ `hnsw_ef` xuống mức thấp nhất bật chế độ tìm kiếm gần
  đúng (ANN/HNSW) — trên một collection đã thật sự build xong HNSW
  (`indexed_vectors_count = points_count`, xác nhận lại trong đợt sửa lỗi gần
  nhất, xem `docs/report.md` §4.5). Bảng `results/axis4_ann.md` cho thấy
  `overlap@10 = 1.0` ở mọi mức `hnsw_ef` kể cả 16 — kết quả không đổi và độ trễ
  cũng không khác biệt đáng kể so với exact search. Lý do xác định được: mỗi
  segment của collection này chỉ có khoảng 625 vector, dưới
  `full_scan_threshold` mặc định của Qdrant (10.000), nên engine tự chọn
  full-scan cho cả hai trường hợp — HNSW dù tồn tại thật vẫn không được dùng
  ở quy mô này. Ở quy mô này, ANN không mang lại lợi ích rõ rệt."
- (Nếu muốn nhấn mạnh bằng số cụ thể: đọc to `latency_ms` hiển thị trên UI ở
  cả hai lần chạy — exact và ANN — để người xem tự so sánh trực tiếp.)

## Nhịp 7 — Một query lỗi đã biết, nói thẳng vì sao hỏng (~20 giây)

- Bật lại `exact`, đổi model về `clip-b32`.
- Gõ một trong các query lỗi đã phân tích trong `results/qualitative.md`, ví
  dụ: **`a person riding a zebra`**.
- Bấm tìm kiếm.
- Nói: "Đây là một tổ hợp khái niệm hiếm — cưỡi ngựa vằn — mà bản thân 5.000
  ảnh COCO không có ảnh nào khớp thật. Toàn bộ top-10 chỉ là ảnh ngựa vằn đơn
  lẻ, không có người cưỡi. Mô hình bỏ qua động từ 'riding' và trả về ảnh gần
  nhất theo danh từ nổi bật nhất trong câu. Đây là giới hạn đã biết của
  contrastive learning trên vector toàn cục, không phải lỗi cài đặt — phân
  tích đầy đủ nằm trong `results/qualitative.md`, nhóm 'Tổ hợp hiếm', điểm
  trung bình đúng ý 0/10 trên cả hai query của nhóm này."

---

## Ghi chú thời lượng

Tổng thời lượng theo timing từng nhịp: 15 + 30 + 20 + 20 + 25 + 20 + 20 = 150
giây (2 phút 30 giây), nằm trong khung 2-3 phút mà brief yêu cầu. Có thể co
giãn nhẹ từng nhịp tuỳ tốc độ nói thật khi quay.
