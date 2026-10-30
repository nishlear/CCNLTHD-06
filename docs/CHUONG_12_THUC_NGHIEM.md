# Chương 12 Triển khai tích hợp và minh chứng kết quả

## 12.1 Hiện thực hóa hệ thống và kiến trúc thực thi

Hệ thống JD Matching được hiện thực bằng FastAPI, LangGraph, SQLite và ChromaDB. Graph nạp CV thực hiện tiền xử lý, trích xuất, kiểm tra hợp đồng dữ liệu, lưu cache và lập chỉ mục; graph tìm kiếm phân tích yêu cầu, truy hồi, chấm điểm, giải thích, chờ HR duyệt rồi sinh câu hỏi phỏng vấn. Luồng điều khiển được xác định bởi node, conditional edge và Send; bind_tools/ToolNode chỉ xuất hiện trong phần lab tham khảo.

Cấu hình sinh văn bản dùng gemini-3.5-flash-lite làm nguồn chính và qwen2.5:3b qua Ollama làm nguồn dự phòng. Tại extract, một lần gọi cloud được chuyển sang Qwen khi gặp 429, timeout hoặc 503; hai nguồn dùng chung prompt và JSON Schema. Vector được tạo bằng gemini-embedding-2 ở 768 chiều. SharedPacingLimiter dùng SQLite để điều tiết cloud ở mức 12 RPM, còn concurrency=4 giới hạn số nhánh ingest đang hoạt động.

Ingest gồm hai pha: các nhánh process_cv trích xuất và commit extraction_cache; barrier chờ đủ nhánh trước khi save_and_index xử lý danh sách prepared để embed/upsert. Bản ghi được cập nhật ready sau khi lưu và lập chỉ mục thành công. Khi chạy lại, importer tái sử dụng CV đã ready và còn vector tương ứng.

**Bảng 12.1 – Ma trận phủ kỹ thuật**

| Yêu cầu | Node / state | Cơ chế thực thi | Minh chứng |
| --- | --- | --- | --- |
| Tiền xử lý và schema | preprocess, extract, validate; clean_text, extracted | Cache hợp lệ đi validate; chưa có cache đi extract; lỗi dữ liệu đi mark_failed. | test_workflows.py; test_gemini_services.py |
| Reducer | prepared, processed; results, explanations, questions, errors | operator.add nối các phần cập nhật danh sách từ nhánh song song; rank sắp xếp kết quả. | test_ingest_two_phase.py; lab state_checkpoint |
| Barrier hai pha | process_cv → save_and_index; prepared | Superstep chờ toàn bộ nhánh trích xuất/cache trước pha embed/upsert. | pilot.log; test_ingest_two_phase.py |
| Pacing và fallback | extract; SharedPacingLimiter | SQLite điều tiết cloud; test provider boundary kiểm tra chuyển Qwen khi 429/timeout/503. | test_gemini_services.py; test_ingest_two_phase.py |
| Human review | human_review; approved_cv_ids | interrupt lưu điểm dừng; Command(resume) tiếp tục cùng thread_id. | before-review.json; after-review.json; test_workflows.py |
| Chịu gọi lặp | claim_review, resume_search, save_review | Claim nguyên tử; khôi phục cùng quyết định; transaction thay tập câu hỏi của search. | test_database.py; test_workflows.py |
| Phân loại kết quả | outcome_fields; errors, incomplete_cv_ids | UI, API và lịch sử dùng cùng tiêu chí đầu ra hoàn chỉnh. | test_outcomes.py; frontend.test.cjs |

Ma trận liên hệ cơ chế trong mã nguồn với test hoặc output có thể kiểm tra trực tiếp.

Sản phẩm giữ ý nghĩa xếp hạng ưu tiên có HR duyệt: ranking_mode=soft_priority, hard_filters_applied=false. API phân biệt status của workflow với outcome của đầu ra, nhờ đó một graph kết thúc vẫn được báo rõ nếu sinh câu hỏi thiếu hoặc gặp lỗi.

**Bảng 12.2 – Quy tắc ánh xạ status và outcome**

| status | outcome | Tiêu chí |
| --- | --- | --- |
| running / awaiting_review | pending | Đang xử lý hoặc chờ HR quyết định. |
| completed | succeeded | Không có errors; mỗi CV được duyệt có đúng 2 câu technical, 2 experience và 2 behavioral. Không chọn CV thì không yêu cầu câu hỏi. |
| completed | completed_with_errors | Graph đã kết thúc nhưng có errors hoặc số câu hỏi theo loại chưa đầy đủ. |
| failed | failed | Workflow bị dừng do lỗi không thể tiếp tục, chẳng hạn retrieval. |

API trả thêm errors, approved_count, question_count, expected_question_count và incomplete_cv_ids để giao diện và thống kê diễn giải thống nhất.

## 12.2 Môi trường, nguồn dữ liệu và tiền xử lý thông tin liên hệ

Nguồn thực nghiệm là tập Djinni tiếng Anh gồm 210.250 dòng. Bước chuẩn bị loại hồ sơ thiếu dữ liệu, ngoài khoảng độ dài hoặc trùng theo ID/nội dung chuẩn hóa, rồi lấy mẫu Reservoir Sampling 100.000 hồ sơ từ 209.638 bản ghi đủ điều kiện; 612 dòng không đạt độ dài. Mỗi hồ sơ có ID nguồn và tên hiển thị Anonymous candidate để truy vết bằng định danh nghiên cứu.

**Bảng 12.3 – Môi trường và cấu hình thực nghiệm**

| Hạng mục | Giá trị |
| --- | --- |
| Máy đo | Windows 10; Intel64 Family 6 Model 183; 28 CPU logic; RAM 15,84 GiB |
| Runtime và lưu trữ | Python 3.11.9; SQLite 3.45.1; uv.lock |
| Thư viện khóa | langgraph 1.2.16; chromadb 1.5.9; google-genai 2.33.0; langchain-google-genai 4.4.2 |
| Corpus nguồn / mẫu | 210.250 dòng / 100.000 CV; reservoir seed 20261023 |
| Văn bản trong mẫu | 100–7.406 ký tự/CV; trung vị 786 ký tự; tổng 98.021.342 ký tự |
| Tệp nguồn / JSONL | Parquet 237.415.736 byte; JSONL 153.764.041 byte; chuẩn bị mẫu 8,220 giây |
| Pilot cloud | 20 CV mỗi lượt; tách xử lý trong lượt với already_ready |
| Cloud / local chat | gemini-3.5-flash-lite / qwen2.5:3b; temperature=0 |
| Embedding và collection | gemini-embedding-2; 768 chiều; cv_gemini_embedding_2_768_v1 |
| Điều tiết và bộ nhớ local | 12 RPM; concurrency=4; keep_alive=30m; num_ctx=4096 |
| Hợp đồng dữ liệu | JSON với skills, exp_yrs, role, summary; nội dung embed compact-v2 dưới 300 từ |

Hai quy mô được dùng cho hai phép đo: corpus 100.000 CV cho baseline từ khóa cục bộ và pilot 20 CV cho pipeline có model.

Tiền xử lý dùng regex/hàm chuỗi để che email, URL, liên kết hồ sơ và số điện thoại, sau đó chuẩn hóa khoảng trắng trước extract. Văn bản đưa vào embedding được ghép từ các trường đã trích xuất theo mẫu role | Skills: skills | Experience: exp_yrs years | summary. JSONL nguồn được giữ để đối chiếu; nội dung đã lọc đi vào chỉ mục hoặc biên gọi model.

Benchmark FTS5 đã lưu dùng bộ lọc contacts-v1. Pipeline hiện hành dùng contacts-v2 trong text_cleaning.py, được gọi từ importer và preprocess. Bảng 12.4 giữ nguyên kết quả của phiên bản đã đo.

**Bảng 12.4 – Kết quả bộ lọc contacts-v1 trên 100.000 CV**

| Chỉ số | Kết quả |
| --- | --- |
| Hồ sơ có ít nhất một thay thế | 1.277 |
| URL được thay thế | 2.811 |
| Liên kết hồ sơ | 16 |
| Số điện thoại | 8 |
| Định danh messenger | 6 |
| Thời gian lọc | 17,916 giây |

Số thay thế đếm theo lần xuất hiện, nên một hồ sơ có thể đóng góp nhiều mục liên hệ.

## 12.3 Kết quả thực nghiệm tổng hợp

### 12.3.1 Kiểm thử tự động

Bộ kiểm thử kiểm tra định tuyến graph, hợp đồng JSON, bảo vệ dữ liệu, hai pha ingest, checkpoint/human review, persistence và cách UI/API diễn giải kết quả. Các lỗi 429, timeout và 503 được tạo tại biên provider để kiểm tra nhánh xử lý; kết quả gọi model thật được báo cáo riêng ở 12.3.3.

**Bảng 12.5 – Kết quả kiểm thử mã nguồn**

| Bộ kiểm thử | Kết quả | Thời gian |
| --- | --- | --- |
| Python pytest | 119 passed; 15 subtests passed | 20,45 giây |
| Frontend Node test | 5 passed; 0 failed | 107,064 ms |

Đây là kết quả của bộ mã nguồn bàn giao; các mốc test cũ được giữ trong artifact lịch sử.

### 12.3.2 Baseline tìm kiếm từ khóa trên 100.000 CV

Baseline lập chỉ mục văn bản đã lọc liên hệ bằng SQLite FTS5 và xếp hạng BM25. Thời gian tạo chỉ mục là 26,305 giây, gồm bước lọc 17,916 giây; tệp SQLite chiếm 189.288.448 byte. Mỗi truy vấn chạy 50 lần với cache ấm.

**Bảng 12.6 – Độ trễ truy vấn SQLite FTS5 BM25**

| Truy vấn AND | Số khớp | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- |
| python + fastapi | 494 | 0,940 | 2,486 |
| java + spring | 4.035 | 3,411 | 4,314 |
| react + typescript | 4.380 | 3,748 | 5,091 |
| docker + kubernetes | 1.665 | 2,227 | 4,133 |
| sql + data | 4.363 | 6,534 | 8,062 |

Độ trễ trung vị nằm trong khoảng 0,940–6,534 ms; phép đo này sử dụng truy vấn từ khóa cục bộ, với 0 request model và 0 request embedding.

### 12.3.3 Pipeline Gemini trên pilot 20 CV

Phép kiểm tra SDK xác nhận đường gọi inference hoạt động. Các pilot sau đó ghi riêng số CV xử lý trong lượt, số ready được tái sử dụng và thời gian runner. Pilot A/B đều ghi nguồn extract là Gemini; kết quả review được đối chiếu theo từng search.

**Bảng 12.7 – Tổng hợp các lượt chạy Gemini**

| Phép thử | CV xử lý / tái sử dụng | Thời gian | Kết quả và liên hệ review |
| --- | --- | --- | --- |
| SDK MODEL_OK | 0 CV; 1 request | 1.915,83 ms | succeeded, STOP; output MODEL_OK; 7 input, 3 output nhìn thấy, 55 thinking, 65 token tổng. |
| pilot-0-20 | 5 / 15 | 7,658 s | 5 ready, 0 failed; search edd6a99b dừng awaiting_review, Top 10. |
| pilot-A-20 | 19 / 1 | 99,424 s | 19 ready, 0 failed; runner tạo search 154131c9. Search riêng a91025e0 có Top-10/60 câu hỏi. |
| pilot-B-20 | 20 / 0 | 121,124 s | 20 ready, 0 failed; runner chỉ ingest. Search sau đó 13dbc9cc có Top-1/6 câu hỏi. |
| pilot-B-resume | 0 / 20 | 0,609 s | Bỏ qua cả 20 CV đã ready và có vector tương ứng. |

Thời gian của runner và của review là hai pha đo riêng; 60/6 câu hỏi được ghi theo search, không cộng vào thời gian ingest trong bảng. MODEL_OK là request độc lập, không đại diện usage của pilot.

Kiểm tra lưu trữ sau pilot B ghi nhận 40 vector compact-v2 trong collection, đều 768 chiều, hữu hạn và có chuẩn xấp xỉ 1; toàn bộ CV ready được đối chiếu đều có vector.

## 12.4 Phân tích cơ chế điều phối và nghiên cứu tình huống

### 12.4.1 Đồng thời, pacing và barrier hai pha

Mục tiêu phân tích là xác định thời điểm pha embedding bắt đầu so với các nhánh trích xuất. Concurrency=4 cho phép nhiều nhánh hoạt động, trong khi SharedPacingLimiter điều tiết lượt gọi cloud theo nhịp khoảng 5 giây.

**Bảng 12.8 – Timeline thực thi pilot-B-20**

| Mốc tương đối | Sự kiện trong log | Ý nghĩa |
| --- | --- | --- |
| t+0,000 s | SDK bắt đầu xử lý chat đầu | Mốc quy chiếu của pilot-B-20. |
| t+5,022 s; t+10,003 s | Hai lượt chat kế tiếp | Các nhánh đồng thời vẫn đi qua pacer. |
| t+13,957 s | Một nhánh commit extraction_cache | Kết quả pha 1 được lưu khi các nhánh khác còn chạy. |
| t+108,953 s | Nhánh cuối commit cache | Hoàn tất 20 cache writes. |
| t+109,164 s | phase=embed_upsert; batch_size=20 | Pha 2 bắt đầu sau toàn bộ cache. |
| t+118,628 s | batchEmbedContents HTTP 200 | Nhận response embedding theo batch. |

Timeline cho thấy barrier giữ đúng thứ tự extract/cache → embed/upsert. Reducer operator.add ghép các cập nhật prepared/processed và results/explanations/questions/errors; barrier chờ nhánh, còn rank sắp xếp theo score và CV ID. Ba cơ chế đảm nhiệm ba chức năng khác nhau.

### 12.4.2 Minh chứng tính điểm và giải thích

Case study dùng cv-2bc3301a trong search a91025e0 với truy vấn “dev 3 năm”. Hồ sơ có similarity=0,8190341294, vượt MIN_SIMILARITY=0,55 tại retrieval. Query có skills rỗng nên skill_fit=1; dữ liệu đã extract lưu 2 năm và min_years=3.

**Phân rã điểm của cv-2bc3301a**

| Thành phần | Phép tính | Đóng góp |
| --- | --- | --- |
| Kỹ năng | 0,5 × 1 | 0,500000 |
| Tương đồng vector | 0,3 × 0,8190341294 | 0,245710 |
| Kinh nghiệm | 0,2 × (2/3) | 0,133333 |
| Tổng | Làm tròn sau khi cộng | 0,879044; hạng 9 |

Score 0,879044 vượt EXPLAIN_THRESHOLD=0,50 nên hồ sơ đi qua explain và được UI chọn sẵn. MIN_SIMILARITY áp dụng cho tương đồng vector, còn EXPLAIN_THRESHOLD áp dụng cho điểm tổng; vì vậy hai ngưỡng phục vụ hai bước khác nhau.

Ví dụ thể hiện đúng thiết kế điểm mềm: dù giá trị kinh nghiệm đang lưu thấp hơn yêu cầu, ứng viên vẫn nằm trong shortlist để HR xem xét.

### 12.4.3 Lỗi trích xuất LLM và độ nhạy thứ hạng

Mục tiêu là xác định thông tin nguồn nào có thể làm sai thành phần kinh nghiệm. Hai hồ sơ đứng đầu cùng được Gemini trích xuất 3,0 năm, nhưng câu nguồn cần cách diễn giải thận trọng hơn.

**Bảng 12.9 – Văn bản nguồn và độ nhạy thứ hạng**

| CV | Đoạn nguồn | Extract | Đối chiếu văn bản | Điểm / hạng |
| --- | --- | --- | --- | --- |
| 62c4d272 | “I worked with a dating project in a product company for almost 3 years” | 3,0 | Dưới 3 năm; số chính xác chưa được xác lập. | Quan sát: 0,958391 / 1. Tạm unknown: 0,758391 / 9. |
| 700fef76 | “I have been learning programming for more than 3 years” | 3,0 | Thời gian học; số năm làm việc thương mại chưa xác định. | Quan sát: 0,956896 / 2. Tạm unknown: 0,756896 / 9. |

Phép thử thay kinh nghiệm của từng CV bằng unknown riêng lẻ và giữ nguyên chín điểm còn lại trong Top 10 đã lưu. Mỗi trường hợp giảm 0,2 điểm và về hạng 9; cách so sánh này cô lập ảnh hưởng của thành phần kinh nghiệm.

Lỗi “almost 3” và việc đồng nhất thời gian học với thời gian làm có thể lan sang lời giải thích và câu hỏi phỏng vấn. Node human_review tạo điểm kiểm tra thông tin nguồn trước khi HR duyệt.

### 12.4.4 Resume và tính lũy đẳng

Mục tiêu là kiểm tra một lượt tìm kiếm có thể dừng chờ HR và tiếp tục đúng trạng thái. search_id được dùng làm thread_id của SqliteSaver; khi có quyết định, Command(resume=...) tiếp tục thread đó. Các search được giữ tách biệt trong Bảng 12.10.

**Bảng 12.10 – Minh chứng human review theo search**

| Search rút gọn | Kết quả lưu | Minh chứng |
| --- | --- | --- |
| a91025e0 | 10 CV duyệt; 60 câu hỏi | report-audit-20261024.json; ba ảnh giao diện Top 10. |
| c3adbe3a | 1 CV duyệt; 6 câu hỏi | report-audit-20261024.json; ảnh top1-awaiting-review và top1-completed. |
| 13dbc9cc | pending/0 câu → approved/6 câu | before-review.json và after-review.json; next từ human_review thành rỗng. |
| lab-review-1 | output [] → [A,B]; replay giữ [A,B] | lab-start.json, lab-resume.json, lab-replay.json; process mới dùng cùng thread. |

human_review không ghi nghiệp vụ trước interrupt. Khi resume, node chạy lại từ đầu; claim_review nhận quyền xử lý nguyên tử, resume_search đối chiếu quyết định và save_review thay tập câu hỏi trong transaction thay vì nối thêm bản sao. Test checkpoint chạy ở process mới xác nhận cùng thread khôi phục được, thread sai bị từ chối và replay giữ kết quả.

Bằng chứng cơ chế nằm trong test_workflows.py, test_database.py và test_lab_checkpoint.py; các giới hạn về khởi động lại web và số lần gọi model được tập trung ở 12.5.1.

## 12.5 Giới hạn kết luận, sự cố lịch sử và khả năng tái lập

### 12.5.1 Phạm vi và giới hạn thực nghiệm

Quy mô và thước đo. Baseline FTS5 đo truy hồi từ khóa trên 100.000 CV; mỗi pilot cloud chỉ chọn 20 CV. Số ready hoặc vector quan sát được không chứng minh đã index toàn corpus. Chưa có bộ nhãn HR độc lập để tính Precision@k, Recall@k hay nDCG@k; unit/integration tests xác nhận logic trong phạm vi test, không đo độ đúng tuyển dụng của LLM.

Ý nghĩa điểm. Score là tổng trọng số để ưu tiên, không phải xác suất tuyển dụng hoặc bảo đảm đạt mọi điều kiện. min_years là điểm mềm; skills rỗng cho skill_fit=1 và cộng 0,5. Ví dụ cv-2bc3301a tính trên giá trị extract, chưa xác nhận 2 năm là tổng kinh nghiệm đúng. Phép unknown ở Bảng 12.9 là phân tích độ nhạy trong Top 10 cố định, không phải ground truth, rerun retrieval hoặc kết quả sửa đồng thời cả hai CV.

Tiền xử lý. contacts-v1 trong benchmark và contacts-v2 hiện hành là hai phiên bản khác nhau; chạy lại bằng v2 có thể đổi số thay thế và thời gian. Regex chỉ che thông tin nhận diện được; tên trong văn bản, công ty, địa điểm và liên hệ viết né vẫn có thể còn. Tên hiển thị ẩn danh không đồng nghĩa JSONL nguồn đã được ẩn danh hoàn toàn.

Đồng thời và model. Timeline dùng timestamp log SDK, không phải timestamp request trên mạng; khoảng cách nhỏ nhất đã lưu là 4,981 giây nên không dùng log đó để khẳng định giới hạn tuyệt đối 5 giây. Pilot A/B dùng cloud; warm-up Qwen và phép extract Qwen trực tiếp là bằng chứng riêng, chưa chứng minh failover tự nhiên trong pilot. Test mock 429/timeout/503 xác nhận logic chuyển model; embedding giữ cùng Gemini và retry backoff 10/20/40 giây cộng jitter 1–3 giây.

Khôi phục và chi phí. Đọc checkpoint hoặc resume lab ở process độc lập chưa tương đương thử restart/resume qua web. Transaction ngăn nhân đôi bản ghi nghiệp vụ, nhưng chưa bảo đảm model chỉ được gọi đúng một lần nếu process dừng sau response và trước checkpoint. Usage MODEL_OK không được cộng vào pilot; chưa có phép đo chi phí toàn pilot đầy đủ. Auth, OCR và quyết định tuyển dụng tự động nằm ngoài phạm vi bàn giao.

### 12.5.2 Sự cố môi trường và dữ liệu lịch sử

Các phép thử OpenAI ban đầu bị chặn trước xử lý do thiếu dependency/DNS 11001, với 0/100 CV được xử lý; những lượt thử Gemini cũ gặp DNS/timeout hoặc HTTP 504 cũng không tạo bằng chứng suy luận thành công. Các log được giữ riêng trong artifact sự cố, còn kết quả thành công được tổng hợp tại 12.3.3.

SQLite có 60 CV failed lịch sử với lý do lưu “Thiếu full_name, skills sau 2 lần thử.” Các hàng thiếu run ID gốc được gắn legacy-unattributed trong evidence-review.json và không cộng vào mẫu số pilot A/B. Một search lịch sử 154131c9 có status=completed nhưng 0 câu hỏi và ConnectError tại generate_questions; đây là trường hợp phân biệt workflow kết thúc với đầu ra đủ dùng theo Bảng 12.2.

### 12.5.3 Lệnh tái lập và danh mục tệp minh chứng

Các đường dẫn dưới đây tính từ thư mục gốc repository. Artifact gốc, JSON và log đầy đủ được giữ nguyên; bảng trong phần chính chỉ trình bày số liệu và hành vi cần phân tích.

**Bảng 12.11 – Danh mục minh chứng và mục sử dụng**

| Nhóm | Tệp / thư mục | Dùng tại |
| --- | --- | --- |
| Nguồn và môi trường | data/datasets/djinni/sample-100000.manifest.json; data/benchmarks/environment_20261024.json | 12.2 |
| Baseline contacts-v1 | data/benchmarks/djinni_100k_filtered_20261024.json; experiments_20261024.json | 12.2; 12.3.2 |
| SDK độc lập | data/benchmarks/gemini-flash-lite-check.json | 12.3.3 |
| Pilot 0 / A | data/benchmarks/pilot-20-results-20261024.json; pilot-20-verification-20261024.json; gemini-dataset-run.json | 12.3.3 |
| Pilot B và review | data/benchmarks/proof/live-validation-20261024/: pilot-run.json, resume-run.json, pilot.log, before-review.json, after-review.json, summary.json, qwen-real-extract.json | 12.3.3; 12.4 |
| Search, ảnh và lỗi nguồn | data/benchmarks/report-audit-20261024.json; proof/report-audit-20261024/; proof/review-followup/evidence-review.json | 12.4; 12.5.2 |
| Tests và checkpoint | data/benchmarks/proof/review-followup/: tests.log, tests.xml, frontend-tests.log, lab-*.json, lab-checkpoints-v2.db, handoff-checkpoints.db | 12.3.1; 12.4.4 |
| Sự cố cũ | data/benchmarks/pilot_100_20261024.json; gemini-execution-20261024.json; gemini-deadline-investigation-20261024.json; experiments_20261024.json | 12.5.2 |
| Tái lập mã | pyproject.toml, uv.lock; src/backend/graphs/; labs/state_checkpoint/; scripts/package_handoff.py | 12.1; Phần IV |
| Gói bàn giao | data/benchmarks/proof/: jd-matching-handoff.zip, review-followup/handoff-manifest.json; working-tree.patch trong ZIP | 12.5.3 |

Trừ các đường dẫn bắt đầu bằng data/, các tên nối tiếp trong cùng ô thuộc thư mục đã nêu ở đầu ô. Các snapshot lịch sử giữ số test và cấu hình tại thời điểm của chính chúng.

**Bảng 12.12 – Định danh nguồn và truy vết**

| Đối tượng | Định danh đầy đủ |
| --- | --- |
| Commit mã ứng dụng / lab | 8861d52ba723b7a5f6f9a625118d23cc48e43db3 |
| Dataset revision | f4d04f624126e27a17770d06953dce7e914ce214 |
| SHA-256 Parquet nguồn | a7b1378e0b231507c377659bf2c80fa4ba60c4e31e1107b13425d103879ecc3b |
| SHA-256 JSONL 100k | e446c3fa164a3003f453f28ad6130fbda79cd2f31f8b86620edabcf338984f8d |
| SHA-256 pilot B 20 CV | 948e2ce992531c727d1875bb00c3bc6535bec9c861c72b6b5b7a6302d00e8ae4 |
| Search / thread_id | edd6a99b380d426292266df10f7f0167 (pilot 0)<br>154131c9b3fd4c64b33aaf918e82d827 (runner A)<br>a91025e0f20f489db58dda8d4a739efe (Top 10)<br>c3adbe3a46d348a1be2e0577737d9b4f (Top 1)<br>13dbc9cc7fad4f29a96bdf686fd3d9e5 (Top 1 bổ sung) |
| CV case studies | djinni-2bc3301a-5066-5dfc-ac78-1dfc3ee9b5fd<br>djinni-62c4d272-3685-5392-8539-69b59d802888<br>djinni-700fef76-d28b-5075-bc93-4d3a63c5ea15 |

Manifest ghi HEAD thực tế, working_tree_dirty và SHA-256 từng tệp trong gói. Các chỉnh lý tài liệu sau commit đi kèm source và patch; HEAD không được dùng để đại diện riêng cho phần chưa commit. Mọi mã băm nguồn cũ vẫn được giữ trong artifact gốc và bản lưu chương trước biên tập.

Lệnh tái lập được tách theo tác động. Hai nhóm đầu chỉ chạy test hoặc baseline cục bộ; nhóm cloud bên dưới phát sinh request thật và vẫn giới hạn 20 CV.

```powershell
uv sync --locked --extra research
uv run --no-sync python -m pytest -q
node --test tests/frontend.test.cjs
uv run --no-sync python -m scripts.benchmark_corpus --repeats 50 --redact-contacts --database data/benchmarks/fts-recheck.sqlite --report data/benchmarks/fts-recheck.json
```

Baseline chạy lại dùng bộ lọc hiện hành; Bảng 12.4 và 12.6 dẫn kết quả contacts-v1 đã lưu. Dùng tên output mới để bảo toàn báo cáo gốc.

```powershell
uv run --no-sync python -m scripts.check_gemini_model --transport sdk --model gemini-3.5-flash-lite --timeout 60 --report data/benchmarks/model-recheck.json
.\scripts\run-real-data.ps1 -Limit 20
uv run --no-sync python -m scripts.run_dataset --input data/benchmarks/proof/live-validation-20261024/pilot-input.jsonl --limit 20 --batch-size 20 --report data/benchmarks/pilot-recheck.json
```

already_ready phụ thuộc trạng thái SQLite/Chroma hiện có; do đó lệnh chạy lại có thể tái sử dụng hồ sơ thay vì tạo một lượt ingest lạnh. Mỗi lần chạy lưu báo cáo riêng và đối chiếu đúng search/thread.

```powershell
uv run --no-sync python -m labs.state_checkpoint.solution --db data/lab/recheck.db --phase start
uv run --no-sync python -m labs.state_checkpoint.solution --db data/lab/recheck.db --phase resume
uv run --no-sync python -m scripts.package_handoff
```

Nguồn tham khảo: LangGraph Interrupts — https://docs.langchain.com/oss/python/langgraph/interrupts ; Djinni Dataset — https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english .

