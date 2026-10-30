# Kết quả kiểm chứng bổ sung

Lượt kiểm chứng bổ sung thực hiện trên backend thật, sử dụng dữ liệu CV từ corpus đã chuẩn bị và cấu hình model cố định của hệ thống. Bằng chứng gồm phản hồi API, log gọi model, dữ liệu nghiệp vụ SQLite, trạng thái checkpoint và vector Chroma. Các kết quả dưới đây thuộc lượt mới, được phân biệt với pilot và ảnh minh họa đã trình bày trước đó.

## Kiểm thử và xác nhận model

| Hạng mục | Kết quả | Minh chứng |
|---|---|---|
| Kiểm thử backend | 115 tests và 15 subtests passed, trong 13,53 giây | pytest-final.log, pytest-final.xml |
| Kiểm thử frontend | 4 passed, 0 failed, 0 skipped | frontend-tests.log |
| Kiểm tra backend | GET /health trả HTTP 200, status=ok | health-final.json |
| Inference cloud | gemini-3.5-flash-lite trả MODEL_OK, finish_reason=STOP | model-check-sdk.json |

Bộ kiểm thử kiểm tra logic và các nhánh lỗi bằng cách thay lời gọi provider tại boundary; kết quả test không đại diện cho độ chính xác của model. Phép MODEL_OK là inference Gemini thật, có response ID Qbzcat_hCemp4t4Pl-mKuQw. Request này dùng 7 input token, 3 output token và 69 thinking token, tổng 79 token; thời gian 17.489,543 ms. Đây là số đo của riêng request kiểm tra, không phải tổng usage hay thời gian của pilot CV.

## Extract trước và embedding sau trên 20 CV mới

Tập kiểm chứng gồm 20 CV ID chưa xuất hiện trong database tại thời điểm chọn. Cả 20 hồ sơ được trích xuất bằng gemini-3.5-flash-lite; sau đó profile cô đọng được vector hóa bằng gemini-embedding-2. Chưa có hồ sơ trong pilot phải chuyển sang Qwen.

| Chỉ số | Lượt xử lý mới | Nạp lại cùng tập |
|---|---:|---:|
| Hồ sơ yêu cầu | 20 | 20 |
| Hồ sơ xử lý | 20 | 0 |
| Ready mới | 20 | 0 |
| Failed | 0 | 0 |
| Already ready | 0 | 20 |
| Thời gian | 121,124 giây | 0,609 giây |

Log ghi nhận 20 phản hồi generateContent HTTP 200 và 20 lần ghi extraction cache. Tất cả cache được lưu trước khi bắt đầu pha embedding. Pha sau chỉ gửi profile dài 34–70 từ, đều dưới 300 từ; một request batchEmbedContents trả HTTP 200 cho 20 hồ sơ. Kết quả nạp lại xác nhận trạng thái ready và vector hợp lệ được tái sử dụng, không phát sinh extraction mới cho cùng tập CV.

Sau pilot, database có 40 hồ sơ ready và Chroma có 40 vector. Database còn 60 hàng failed lịch sử; đây không phải lỗi của pilot mới. Số liệu này cập nhật quy mô chỉ mục sau lượt bổ sung, còn số liệu pilot trước được giữ theo ngữ cảnh riêng.

Minh chứng: pilot-run.json, pilot.log, resume-run.json và resume.log.

## Tìm ứng viên có từ 3 năm kinh nghiệm và duyệt một người

Truy vấn sử dụng: “Software engineer with at least 3 years of commercial experience, skilled in Akka, Netty and Kafka Streams”, top_k=1. Search ID của lượt này là 13dbc9cc7fad4f29a96bdf686fd3d9e5.

| Thuộc tính | Kết quả |
|---|---|
| Ứng viên | Anonymous candidate 3f56345b |
| Vai trò | Software Engineer |
| Kinh nghiệm lưu trữ | 5 năm |
| Bằng chứng trong CV nguồn | “5+ years of commercial experience” |
| Kỹ năng khớp | Akka, Netty, Kafka Streams |
| Kỹ năng thiếu trong truy vấn | Không có |
| Điểm xếp hạng | 0,977572 |

Bằng chứng nguồn hỗ trợ yêu cầu ít nhất 3 năm kinh nghiệm thương mại đối với ứng viên này. Điểm xếp hạng không được diễn giải thành xác suất tuyển dụng hay thay thế việc kiểm tra CV.

| Trạng thái | Trước duyệt | Sau duyệt một ứng viên |
|---|---|---|
| API và SQLite | awaiting_review | completed, 1 người approved |
| approval_status của graph | pending | approved |
| Node chờ trong checkpoint | human_review | Không còn |
| Câu hỏi đã lưu | 0 | 6 |
| Lỗi trong graph | Không có | Không có |

Sáu câu hỏi gồm 2 technical, 2 experience và 2 behavioral. API tìm kiếm và review đều trả HTTP 200. Snapshot trước/sau được đọc bằng tiến trình Python riêng, xác nhận checkpoint và dữ liệu nghiệp vụ tồn tại trên đĩa. Phép kiểm chứng này không bao gồm benchmark chủ động khởi động lại server.

Lượt bổ sung được thực hiện qua API của hệ thống; không dùng các ảnh của lượt trước để đại diện cho search ID mới. Có thể mở lại kết quả đã lưu tại http://127.0.0.1:8000/#search=13dbc9cc7fad4f29a96bdf686fd3d9e5 trên máy chạy ứng dụng.

Minh chứng: search-request.json, search-created.json, review-request.json, review-response.json, before-review.json và after-review.json.

## Vector và kiểm tra Qwen thực tế

Collection cv_gemini_embedding_2_768_v1 có 40 vector, mỗi vector 768 chiều. Toàn bộ giá trị hữu hạn, có marker compact-v2 và norm xấp xỉ 1; không có hồ sơ ready nào thiếu vector. Metadata xác nhận model gemini-embedding-2. Minh chứng: after-review.json.

Qwen2.5:3b cũng đã được gọi trực tiếp để extract riêng một CV thật từ tập pilot. Kết quả JSON qua schema chung, sử dụng format=json, temperature=0, keep_alive=30m và num_ctx=4096; thời gian 30,147 giây. Phép thử này xác nhận local model trả được dữ liệu theo hợp đồng extraction, không phải bằng chứng một lỗi 429 thực tế đã kích hoạt failover. Output Qwen được lưu riêng, không ghi đè hồ sơ Gemini. Minh chứng: qwen-real-extract.json và ollama-ps.txt.

## Phạm vi kết luận và truy vết

Bằng chứng mới xác nhận ingest hai pha, tái sử dụng hồ sơ ready, lưu vector, interrupt trước duyệt và sinh câu hỏi sau duyệt với dữ liệu thật. Pilot không xuất hiện lỗi 429 hoặc timeout tự nhiên; nhánh failover và backoff được kiểm tra bằng regression tests. Cấu hình pacer 12 RPM đã được đối chiếu ở cấp mã nguồn và test, chưa có phép đo timestamp trên đường truyền để chứng minh khoảng cách tuyệt đối giữa mọi request.

Chưa có bộ nhãn CV–JD để kết luận Precision, Recall hoặc độ chính xác extraction; cũng chưa đo toàn bộ chi phí provider. Những hạn chế về lọc cứng kinh nghiệm và hiểu nghề nghiệp đã nêu trong báo cáo vẫn còn hiệu lực.

Gói bằng chứng gồm 24 tệp. summary.json tổng hợp kết quả; evidence-manifest.json ghi SHA-256 của các tệp đóng gói. Snapshot lưu thêm hash các file code để đối chiếu với bản mã nguồn đã chạy. Gói chia sẻ không chứa API key, raw CV đầu vào hoặc toàn bộ database.
