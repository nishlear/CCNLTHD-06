# Kiểm chứng chạy thật JD Matching — 24/10/2026

Đợt chạy buổi tối, ghi nhận lúc 21:22 (Asia/Saigon). Đây là lượt kiểm chứng mới, tách khỏi log và ba ảnh chụp buổi sáng trong báo cáo. Mã nguồn đang làm việc được đối chiếu bằng SHA-256 trong các snapshot; không dùng riêng HEAD để đại diện cho thay đổi chưa commit.

## 1. Kết quả kiểm thử và khởi chạy

| Hạng mục | Kết quả |
|---|---|
| Đồng bộ môi trường | `uv sync --locked --extra research --link-mode copy --offline` thành công |
| Backend tests | 115 passed; 15 subtests passed; 13,53 giây |
| Frontend tests | 4 passed; 0 failed; 0 skipped |
| Backend thực tế | `GET /health` trả HTTP 200, `status=ok` |
| Truy cập lượt đã hoàn tất | http://127.0.0.1:8000/#search=13dbc9cc7fad4f29a96bdf686fd3d9e5 |

Các provider trong regression tests được thay tại boundary để kiểm tra logic và nhánh lỗi. Chỉ các bằng chứng model ở mục 2–5 bên dưới được tính là chạy model thật.

Lỗi môi trường phát hiện trong đợt chạy: SDK chưa có `socksio` trong khi máy dùng SOCKS proxy. Đã khai báo `httpx[socks]>=0.28` trong pyproject và cập nhật uv.lock; sau đó SDK gọi Gemini thành công. Không tắt hoặc bỏ qua proxy của ứng dụng. Lần thử chạy thêm backend bị từ chối vì cổng 8000 đã được backend trước sử dụng; các API trên backend đang chạy vẫn trả kết quả thành công.

## 2. Model thực tế

| Tác vụ | Model và bằng chứng |
|---|---|
| Kiểm tra cloud | `gemini-3.5-flash-lite`; HTTP thành công; output `MODEL_OK`; response ID `Qbzcat_hCemp4t4Pl-mKuQw` |
| Extract pilot mới | `gemini-3.5-flash-lite` cho cả 20 hồ sơ |
| Embed pilot | `gemini-embedding-2`, một request `batchEmbedContents` HTTP 200 cho 20 hồ sơ |
| Hiểu query, giải thích, sinh câu hỏi | `gemini-3.5-flash-lite` qua backend thật |
| Local | `qwen2.5:3b` warmup thành công; gọi extract riêng một CV thật, JSON qua schema, 30,147 giây |

Request `MODEL_OK` riêng có 7 input, 3 output, 69 thinking, tổng 79 token; thời gian 17.489,543 ms. Đây không phải token hoặc độ trễ của toàn bộ pilot.

Kiểm tra Qwen dùng `format=json`, `temperature=0`, `keep_alive=30m`, `num_ctx=4096`, cùng prompt/schema extraction của ứng dụng. CV nguồn thuộc tập pilot, ID b9b88a17; tên rỗng là phù hợp với nguồn ẩn danh. Kết quả local chỉ lưu làm bằng chứng riêng, không ghi đè hồ sơ đã extract bằng Gemini. Không có lỗi 429 tự nhiên trong pilot, nên không gọi kiểm tra local trực tiếp này là một lần failover thực tế từ cloud.

## 3. Pilot 20 CV mới và kiểm tra nạp lại

Tập vào được chọn từ `sample-100000.jsonl`: 20 CV ID đầu tiên chưa xuất hiện trong app.db tại lúc chuẩn bị. SHA-256 tập con: `948e2ce992531c727d1875bb00c3bc6535bec9c861c72b6b5b7a6302d00e8ae4`.

| Đại lượng | Lượt mới | Nạp lại cùng tập |
|---|---:|---:|
| Requested | 20 | 20 |
| Processed | 20 | 0 |
| Ready mới | 20 | 0 |
| Failed | 0 | 0 |
| Already ready | 0 | 20 |
| Thời gian | 121,124 giây | 0,609 giây |

Log thực tế ghi 20 phản hồi generateContent HTTP 200, 20 lần lưu extraction cache và một batchEmbedContents HTTP 200. Timestamp của toàn bộ cache đã hoàn tất trước timestamp bắt đầu pha embedding. Các profile cô đọng có 34–70 từ, đều dưới 300 từ. Không gửi raw CV vào embedding.

Log AFC của SDK có khoảng cách nhỏ nhất 4,981 giây và lớn nhất 7,221 giây. Đây là timestamp log trong SDK, không phải timestamp cấp quyền SQLite hay timestamp trên đường truyền; không dùng các số này để khẳng định đã đo được giới hạn tuyệt đối 5 giây trên mạng. Pacer 12 RPM và phối hợp giữa process được kiểm tra ở cấp code/regression tests.

Sau pilot, app.db có **40 ready** và **60 failed lịch sử**. 60 bản ghi lỗi có từ trước, không phải lỗi của pilot mới. Không xóa hay sửa lịch sử để cải thiện số liệu.

## 4. Tìm và duyệt đúng một ứng viên

Query gửi qua API thật: `Software engineer with at least 3 years of commercial experience, skilled in Akka, Netty and Kafka Streams`; `top_k=1`.

- Search ID: `13dbc9cc7fad4f29a96bdf686fd3d9e5`.
- Ứng viên: Anonymous candidate **3f56345b**, Software Engineer.
- Kinh nghiệm lưu: **5 năm**; đoạn nguồn ghi **“5+ years of commercial experience”**.
- Kỹ năng khớp: Akka, Netty, Kafka Streams; không thiếu kỹ năng trong query.
- Điểm mới: **0,977572**. Điểm là công cụ xếp hạng, không phải xác suất phù hợp.

| Thời điểm | SQLite/API | Checkpoint | Câu hỏi |
|---|---|---|---:|
| Trước duyệt | `awaiting_review` | `pending`; `next=[human_review]`; errors rỗng | 0 |
| Sau duyệt 1 người | `completed`; 1 approved | `approved`; next rỗng; errors rỗng | 6 |

Sáu câu hỏi đã được lưu, gồm 2 technical, 2 experience và 2 behavioral. API tìm kiếm, review và health đều trả HTTP 200. Snapshot trước và sau được đọc bởi tiến trình Python riêng, xác nhận dữ liệu bền vững trên đĩa. Không tuyên bố đây là một benchmark chủ động tắt/khởi động lại server.

Lượt này dùng API của hệ thống; chưa có ảnh UI mới. Ba ảnh người dùng lúc 08:29–08:30 và hai ảnh Top 1 cũ vẫn thuộc các lượt buổi sáng, không dùng làm ảnh của search ID mới.

## 5. Kiểm tra Chroma

- Collection: `cv_gemini_embedding_2_768_v1`; metadata ghi đúng Google Gemini / gemini-embedding-2 / 768.
- **40 vector, mỗi vector 768 chiều**, toàn bộ hữu hạn và có marker `compact-v2`.
- Norm nhỏ nhất 0,9999999916; lớn nhất 1,0000000540.
- Không có hồ sơ ready nào thiếu vector.

## 6. Chỉ mục bằng chứng

Tất cả file nằm trong `data/benchmarks/proof/live-validation-20261024/`.

| Tệp | Nội dung kiểm chứng |
|---|---|
| `pytest-final.log`, `pytest-final.xml` | Bộ kiểm thử cuối sau đồng bộ dependency |
| `frontend-tests.log` | Bốn test frontend |
| `model-check-sdk.json` | Inference Gemini thật, model trả về, response ID, usage |
| `pilot-run.json`, `pilot.log` | 20 CV mới, nguồn extract, hai pha, request embedding thật |
| `resume-run.json`, `resume.log` | Cùng 20 CV được bỏ qua qua already_ready |
| `before-review.json` | Checkpoint pending, 0 câu hỏi, hồ sơ và trích đoạn kinh nghiệm |
| `after-review.json` | Completed, approved 1 người, 6 câu hỏi, errors rỗng, vector và hash code |
| `search-request.json`, `search-created.json` | Query thật và response shortlist |
| `review-request.json`, `review-response.json` | Quyết định duyệt và câu hỏi thật |
| `qwen-real-extract.json`, `ollama-ps.txt` | Output local thật và trạng thái model/context |
| `health-final.json`, `server.log` | Backend còn phục vụ; log node và API |
| `summary.json`, `evidence-manifest.json` | Tổng hợp và SHA-256 từng tệp đóng gói |

Gói ZIP đi kèm không chứa `.env`, API key, database đầy đủ, profile trình duyệt hoặc raw CV đầu vào. Script đọc lại dữ liệu: `scripts/capture_report_state.py`.

```powershell
.\.venv\Scripts\python.exe -m scripts.capture_report_state --search-id 13dbc9cc7fad4f29a96bdf686fd3d9e5 --expect completed --report data/benchmarks/proof/live-validation-20261024/recheck.json
```

## 7. Giới hạn của đợt kiểm chứng

Không xuất hiện 429/timeout tự nhiên trong pilot; failover và backoff lỗi đã được kiểm tra bằng regression tests, không giả làm bằng chứng lỗi provider thật. Chưa đo Precision/Recall, chất lượng extraction bằng bộ nhãn, tải đồng thời hoặc toàn bộ hóa đơn provider.

Chưa chụp thêm ảnh giao diện: Edge cục bộ không khởi chạy được trong môi trường thực thi; tải Chromium bị HTTP 403 do domain ngoài allowlist. Bằng chứng mới vì vậy là log/API/SQLite/checkpoint/Chroma. Không tạo ảnh mô phỏng để thay thế.
