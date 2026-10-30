# Kiểm tra quy tắc ingest hai pha

Ngày kiểm tra: 24/10/2026. Phạm vi: code ứng dụng và runner trong repository.

| Quy tắc | Trước thay đổi | Trạng thái sau bổ sung |
|---|---|---|
| 1.1 Cố định model | Config đã chặn đổi model chính; adapter vẫn hỗ trợ embedding-001 và cho đổi collection | Chỉ Gemini Flash-Lite, Qwen 3B, Embedding 2; cố định 768 chiều và collection; checker cũng chặn model khác |
| 1.2 Giữ local model | Có tham số mặc định nhưng cho đổi context/keep_alive | Chỉ chấp nhận context 4096 và keep_alive 30m hoặc -1; tuần tự hóa local generation; runner warmup đúng Qwen |
| 1.3 Hợp đồng chung | Chung Pydantic, Ollama chưa nhận JSON Schema trong prompt | Cả hai nhận cùng system/user prompt chứa cùng schema, cùng bước kiểm tra và cùng mapping sang bản ghi |
| 2.1 Regex trước LLM | Có email, URL, số quốc tế, dọn khoảng trắng | Bổ sung số nội địa và nhãn SĐT; tất cả luồng ingest tiền xử lý trước extract; redaction version contacts-v2 |
| 2.2 JSON bắt buộc | Đã có JSON MIME và JSON mode, temperature 0 | Giữ nguyên, thêm schema vào prompt local; kiểm tra JSON object và Pydantic trước cache |
| 2.3 Khóa ngắn | Đã có name, role, skills, exp_yrs, edu, summary | Giữ nguyên; tên trường lưu trữ không ảnh hưởng schema gửi model |
| 2.4 Nén trước embed | Đã embed profile cô đọng nhưng có thể đủ 300 từ | Chỉ role/skills/kinh nghiệm/summary; dưới 300 từ, giữ chỗ cho từng phần, không gửi raw CV |
| 3.1 Tối đa 12 RPM | Pacer chỉ trong process, cho cấu hình đến 60 RPM | SQLite điều tiết chung giữa process dùng cùng app.db; chỉ cho 1..12 RPM, chờ tối thiểu 5 giây giữa các lượt cấp quyền gọi chat |
| 3.2 Failover attempt 2 | Có chuyển local, nhưng graph còn có thể quay lại cloud sau lỗi/thiếu dữ liệu | Tối đa một call cloud; 429 hoặc timeout chuyển ngay một lần sang local, không retry cloud; dữ liệu hỏng/thiếu trường làm failed riêng CV |
| 3.3 Backoff embedding | Đã có backoff cho mỗi văn bản | Áp dụng cho batch CV và embedding truy vấn, chờ 10/20/40 giây + jitter 1..3 giây, không đổi model embedding |
| 4.1 Kiểm tra dict | Đã có kiểm tra ở node từng CV | Kiểm tra từng CV trước batch indexing; đầu vào rỗng/sai kiểu ở save_and_index được ghi failed ngay; kiểm tra thêm schema và trường bắt buộc |
| 4.2 Truy xuất an toàn | Một số state vẫn dùng khóa trực tiếp | Node ingest và dữ liệu extract dùng .get với mặc định đúng kiểu; JSON object được xác minh trước khi dùng |
| 4.3 Cô lập lỗi | Provider lỗi được bắt, cache/graph lỗi còn có thể lan ra batch | Bọc từng graph con; phần tử đầu vào/batch sai kiểu, cache/JSON/provider lỗi thành failed riêng CV; dữ liệu extract sai kiểu trường không cản ghi failed; bulk upsert lỗi thì thử từng CV với vector đã tạo |
| 4.4 Checkpoint từng CV | Có commit ready từng CV, chưa có cache extract độc lập | Cache commit ngay khi extract hợp lệ; upsert xong commit ready từng CV; lần sau bỏ qua ready có vector hợp lệ hoặc dùng cache khi embedding chưa xong |
| Hai pha và batch | Extract xong thì embed ngay từng CV | Pha 1 hoàn tất tất cả nhánh extract/cache trước; pha 2 gom tối đa 20 CV/request và bulk upsert; khóa OS tránh xen pha giữa các lượt ingest cùng app.db |

## Luồng thực thi

Mỗi nhóm input của runner (mặc định 20) thực hiện hai pha riêng. Với `-Limit 50`, runner mặc định xử lý các nhóm 20, 20, 10. Việc chia nhóm giữ lượng state trong RAM có giới hạn.

1. Đọc PDF/text, che contact và dọn khoảng trắng bằng code thuần.
2. Tìm cache theo CV ID và SHA-256 của phiên bản hợp đồng, input đã tiền xử lý, tên hiển thị. Cache bị thay đổi input hoặc schema hỏng sẽ không được dùng.
3. Nếu chưa có cache: gọi Gemini qua pacer chung. SDK chat chỉ được một attempt; lỗi 429 hoặc timeout chuyển sang Qwen. Các lỗi 503 overload cũng đủ điều kiện chuyển local. Không ngủ backoff cloud ở bước failover.
4. Kiểm tra schema, tên hiển thị và skills; commit JSON hợp lệ vào bảng `extraction_cache` của app.db.
5. Sau khi mọi CV trong nhóm xong pha 1, ghép profile theo mẫu `role | Skills: ... | Experience: ... years | summary` và gửi từng batch tối đa 20 `Content` độc lập. SDK tạo một `batchEmbedContents`, không gộp nhiều CV thành một vector.
6. Kiểm tra số vector, số chiều, giá trị hữu hạn và chuẩn hóa. Upsert Chroma theo batch; commit `ready` từng CV ngay sau đó.

Cache tồn tại độc lập với trạng thái failed khi embedding chưa hoàn tất. Nếu process dừng sau upsert nhưng trước ready, lần sau dùng cache và upsert lại cùng CV ID; không tạo bản sao. Xóa CV qua API đồng thời xóa cache của CV đó.

## Xác minh và giới hạn

Các kiểm thử dùng SQLite, LangGraph, SDK Google thật ở phần tạo request, Chroma và PDF parser; mọi provider request được thay bằng phản hồi cục bộ. Đã kiểm tra rào cản hai pha, chia batch 20+3, ánh xạ một CV/một vector, backoff đủ ba mức, lỗi từng CV, tái sử dụng cache, input đổi thì cache mất hiệu lực, checkpoint từng CV, pacing và khóa pipeline giữa các process.

Lượt rà soát bổ sung ngày 24/10/2026 đã chạy toàn bộ suite: **115 passed, 15 subtests passed trong 13,77 giây**. Các ca mới kiểm tra ghi failed ngay khi extraction rỗng, phần tử không phải dict trong đầu vào/batch, dữ liệu sai kiểu trường, giữ thứ tự kết quả, embedding truy vấn gặp 429 và phản hồi Ollama báo lỗi dù HTTP thành công.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Test SDK tạo request được cô lập khỏi biến proxy của máy vì HTTP boundary đã được thay thế trong test; không tắt proxy của ứng dụng. Không có request Gemini hoặc CV mới được gửi trong đợt rà soát này.

Kiểm tra Ollama thực tế: máy đã có đúng `qwen2.5:3b`; `ollama_warmup()` trả `true`. Ngay sau warmup, `ollama ps` ghi context **4096**, model dùng khoảng **2,2 GB**, và còn khoảng **29 phút** lưu trú. Warmup và generation dùng chung khóa trong process, cùng keep_alive/options. Đây là kiểm tra nạp model, chưa phải phép đo chất lượng extraction của Qwen hay thử failover với một lỗi 429 thật từ Gemini. keep_alive yêu cầu Ollama giữ model, không bảo đảm hệ điều hành không swap hoặc client khác không nạp model khác.

Quota chat và embedding không tự chuyển cho nhau. Giảm request chat không bảo đảm embedding hết lỗi quota. Cơ chế chung chỉ phối hợp các process dùng cùng app.db trên một host; project/key dùng từ ứng dụng khác hoặc database khác cần điều tiết riêng.
