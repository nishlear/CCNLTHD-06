## 12.10 Đối chiếu cơ chế và tiêu chí bàn giao

### 12.10.1 Quyết định về phạm vi và điểm số

Phiên bản bàn giao giữ ý nghĩa xếp hạng ưu tiên có HR duyệt. Công thức 0,5 × skill_fit + 0,3 × similarity + 0,2 × experience_fit không đổi; chưa thêm hard filter theo kinh nghiệm, nghề hoặc cấp bậc. CV dưới yêu cầu vẫn được xếp hạng là giới hạn sản phẩm đã công bố, không phải sai khác giữa code và thiết kế. UI đã cảnh báo điểm cao chưa bảo đảm đạt mọi yêu cầu; API công bố ranking_mode=soft_priority, hard_filters_applied=false.

MIN_SIMILARITY=0,55 áp dụng tại retrieve trên similarity vector. EXPLAIN_THRESHOLD=0,5 áp dụng trên score tổng trước explain; UI cũng chọn sẵn từ score 0,5. Hai đại lượng khác nhau nên không cần ép hai ngưỡng bằng nhau. HR phải kiểm tra điều kiện nguồn trước khi duyệt.

### 12.10.2 Liên hệ yêu cầu với state node edge và bằng chứng

| Yêu cầu | Node và state | Điều phối thực tế | Minh chứng |
|---|---|---|---|
| Schema và tiền xử lý | preprocess/extract/validate; clean_text/extracted/source | Cache hợp lệ đi validate; chưa có cache đi extract; thiếu trường đi mark_failed | test_workflows.py; test_gemini_services.py |
| Reducer | IngestState.prepared/processed; SearchState.results/explanations/questions/errors | operator.add nối cập nhật danh sách từ các nhánh; không tự xếp hạng hay loại trùng | test_ingest_two_phase.py; lab collected=[A,B] |
| Barrier hai pha | process_cv → save_and_index; prepared | Superstep chờ đủ nhánh process_cv rồi mới embed/upsert | 20 cache writes trước embed; pilot.log; test_ingest_two_phase.py |
| Fallback và pacing | extract, SharedPacingLimiter; source/api_pacing | Một cloud attempt; 429/timeout/503 sang Qwen; schema lỗi failed; SQLite điều tiết 12 RPM | test_gemini_services.py; test_ingest_two_phase.py; ollama-ps.txt |
| Human review | human_review; approval_status/approved_cv_ids/questions | interrupt; search_id=thread_id; Command(resume) cùng thread | before-review.json; after-review.json; test_workflows.py |
| Chịu gọi lặp | claim_review/resume_search/save_review | Claim nguyên tử; quyết định khớp checkpoint; thay câu hỏi trong transaction, không append bản sao | test_workflows.py; test_database.py |
| Trạng thái kết thúc | outcome_fields; errors/question_count/incomplete_cv_ids | Phân biệt kết thúc graph với đầu ra dùng được trong UI/API/lịch sử | test_outcomes.py; frontend-tests.log |

Bảng phủ xác nhận cơ chế có code và output để giải thích, không phải đánh giá như lab cũ hoặc chứng minh model tự lập kế hoạch. Môi trường dùng Python 3.11, uv.lock và cấu hình model cố định. Không bổ sung bind_tools/ToolNode trước hạn.

### 12.10.3 Phân biệt corpus pilot và lịch sử lỗi

| Nhãn run đối chiếu | Requested | Processed / ready / failed / already_ready | Artifact |
|---|---:|---|---|
| pilot-A-20 | 20 | 19 / 19 / 0 / 1 | gemini-dataset-run.json; 99,424 giây |
| pilot-B-20 | 20 | 20 / 20 / 0 / 0 | pilot-run.json; 121,124 giây |
| pilot-B-resume | 20 | 0 / 0 / 0 / 20 | resume-run.json; 0,609 giây |
| legacy-unattributed | 60 hàng failed lịch sử | Không cộng vào mẫu số A hoặc B | evidence-review.json |

Đây là nhãn audit bổ sung để liên hệ artifact, không giả nhận là run_id được runner gốc ghi. Một already_ready của A là tái sử dụng hàng ready có vector, chưa đủ để gọi là extraction-cache hit. Số 19 là số xử lý trong lượt, không chứng minh đó là lần đầu các CV xuất hiện.

Kho đã chuẩn bị có 100.000 hồ sơ; từng pilot chỉ xét 20. Sau hai lượt, chỉ mục được kiểm tra có 40 vector. Không suy ra đã index toàn kho, không ingest thêm corpus trong đợt hoàn thiện này. Gói bàn giao chỉ chứa tập pilot 20 CV.

Cả 60 hàng failed lịch sử lưu lý do “Thiếu full_name, skills sau 2 lần thử.” Bản ghi chưa có run_id gốc; audit lưu từng CV ID, thời điểm lưu, lý do và đánh dấu chưa xác định run. Không gom chúng vào pilot hoặc suy đoán nguồn chỉ từ timestamp.

Pilot A và B đều extract bằng Gemini. Warmup Qwen chỉ xác nhận nạp model; phép extract Qwen trực tiếp ở mục 12.9 là output thật riêng, không phải failover tự nhiên. Test 429/timeout/503 được gắn nhãn thay provider boundary. Embedding retry cùng model Gemini với backoff 10/20/40 giây + jitter 1–3 giây; lỗi schema extraction làm failed riêng CV. MODEL_OK là request riêng, không cộng token hay latency của nó vào toàn pilot.

### 12.10.4 Timeline và ý nghĩa đồng thời

| Mốc tương đối trong log | Sự kiện | Ý nghĩa |
|---|---|---|
| t+0,000 s | SDK bắt đầu xử lý chat đầu | Mốc quy chiếu pilot-B-20 |
| t+5,022 s và t+10,003 s | Hai lời gọi chat kế tiếp | Concurrency 4 không đồng nghĩa bốn lượt cloud khởi phát đồng thời |
| t+13,957 s | Một nhánh commit extraction_cache | Các nhánh khác còn xử lý; chưa embed riêng CV này |
| t+108,953 s | Nhánh cuối commit cache | Log có đủ 20 cache writes |
| t+109,164 s | phase=embed_upsert, batch_size=20 | Pha 2 bắt đầu sau toàn bộ cache |
| t+118,628 s | batchEmbedContents HTTP 200 | Một response batch, vector riêng từng CV |

Reducer nối các phần cập nhật; barrier đồng bộ các nhánh; rank sắp xếp ổn định theo score và CV ID. Đây là ba nhiệm vụ khác nhau. Pacer cấp lượt qua SQLite, concurrency giới hạn nhánh đang hoạt động. Mốc SDK không phải timestamp trên đường truyền; khoảng cách log nhỏ nhất 4,981 s không chứng minh giới hạn tuyệt đối 5 s giữa mọi request mạng.

### 12.10.5 Ví dụ qua retrieval score explain và chọn sẵn

Trong search a91025e0f20f489db58dda8d4a739efe, CV 2bc3301a có similarity=0,8190341294, vượt ngưỡng retrieval 0,55. Query có skills rỗng nên skill_fit=1, đóng góp 0,5. Theo giá trị đã extract là 2 năm và min_years=3, experience_fit=2/3 đóng góp 0,133333. Similarity đóng góp 0,245710; tổng làm tròn là 0,879044, hạng 9 trong shortlist lưu trữ.

Score từ 0,5 dẫn tới explain và UI chọn sẵn dù số năm đã lưu chưa đạt 3. Điểm cao không xác nhận mọi điều kiện bắt buộc. CV nguồn đề cập thời gian từng dự án nên tổng kinh nghiệm cần xác minh độc lập; ví dụ này giải thích phép tính trên dữ liệu extract, không xác nhận 2 năm là ground truth.

### 12.10.6 Phân tích lỗi trích xuất và ảnh hưởng thứ hạng

| CV | Đoạn nguồn | Extract | Đối chiếu văn bản | Độ nhạy của điểm và thứ hạng |
|---|---|---:|---|---|
| 62c4d272 | “I worked with a dating project in a product company for almost 3 years” | 3,0 | Dưới 3, chưa có số chính xác; không tự gán 2,9 | Hạng 1, score 0,958391; tạm unknown: 0,758391, hạng 9 |
| 700fef76 | “I have been learning programming for more than 3 years” | 3,0 | Số năm làm việc thương mại chưa xác định; thời gian học không phải kinh nghiệm làm việc | Hạng 2, score 0,956896; tạm unknown: 0,756896, hạng 9 |

Đây là đối chiếu nguồn, chưa phải bộ nhãn HR. Phép độ nhạy thay số năm của từng CV bằng unknown riêng lẻ, giữ điểm của chín CV còn lại và chỉ xét Top 10 đã lưu. Không phải rerun retrieval, không phải ground truth được sửa, không phải thứ hạng khi sửa cả hai đồng thời. Không sửa đè dữ liệu lịch sử. evidence-review.json lưu nguồn, giá trị extract, đánh giá đối chiếu và phép tính.

### 12.10.7 Resume và tiêu chí hoàn tất

Top-10/60 câu hỏi thuộc a91025e0f20f489db58dda8d4a739efe. Top-1/6 câu hỏi thuộc c3adbe3a46d348a1be2e0577737d9b4f. Lượt Top-1 bổ sung 13dbc9cc7fad4f29a96bdf686fd3d9e5 cũng có 6 câu hỏi riêng. Không ghép ba search thành một lượt.

Theo tài liệu LangGraph, node chứa interrupt chạy lại từ đầu khi resume bằng cùng thread_id. human_review không ghi nghiệp vụ trước interrupt. Claim review được nhận nguyên tử; save_review thay câu hỏi của search trong transaction nên replay không append bản sao. Nếu model đã sinh kết quả nhưng process dừng trước checkpoint, chưa có bảo đảm gọi LLM đúng một lần. Nguồn: https://docs.langchain.com/oss/python/langgraph/interrupts

| status | outcome dùng cho UI API và thống kê | Tiêu chí |
|---|---|---|
| running / awaiting_review | pending | Chưa kết thúc, có thể đang chờ HR |
| completed | succeeded | Không errors; mỗi CV approved có đúng 2 câu mỗi loại; không chọn CV thì không yêu cầu câu hỏi |
| completed | completed_with_errors | Graph kết thúc nhưng có errors hoặc câu hỏi thiếu/sai số lượng |
| failed | failed | Search bị abort, chẳng hạn lỗi retrieval |

API thêm errors, approved_count, question_count, expected_question_count, incomplete_cv_ids. UI và thống kê lịch sử dùng cùng outcome; không coi mọi completed là thành công. Lịch sử không đọc được checkpoint lỗi được đánh dấu không xác minh được thay vì mặc định thành công. Đọc snapshot ở process mới và lab resume ở process mới không được gọi là thử chủ động web restart/resume.

# PHẦN IV TỰ ĐÁNH GIÁ VÀ BÀN GIAO

## 13.1 Tự đánh giá và phạm vi hoàn thiện

Những cơ chế có thể trình diễn gồm TypedDict, reducer, fan-out, barrier, schema chung, cache/vector và checkpoint/human review. Chất lượng xếp hạng còn phụ thuộc extraction và trọng số chưa hiệu chỉnh bằng bộ nhãn. Hệ thống hỗ trợ HR trong phạm vi thử nghiệm, chưa có auth, OCR hoặc quyết định tuyển dụng tự động. Tests xác nhận logic và các nhánh đã kiểm tra, không đo độ đúng tuyển dụng hoặc LLM.

Ưu tiên tuần 8 là chốt ý nghĩa điểm mềm và diễn giải điều kiện chưa lọc cứng; dẫn output resume với cùng thread_id, giữ pilot 20 CV. Chốt tuần 9 là đóng gói corpus pilot, cấu hình/model, checkpoint mẫu, tests, lab và tự đánh giá/AI/nhật kí. Không tự đặt thêm ngày nộp ngoài mốc tuần 9 đã được phản hồi.

## 13.2 Lab state reducer và checkpoint

labs/state_checkpoint có starter.py, solution.py và README. Đây là chương trình xác định dùng LangGraph/SQLite thật, không phải output model. Starter yêu cầu hàm collect trả cập nhật danh sách một phần tử. Hai nhánh collected=[A] và [B] được reducer hợp nhất thành [A,B]; barrier cho review đọc đủ dữ liệu.

Chạy python -m labs.state_checkpoint.solution --db data/lab/checkpoints.db --phase start: next=[review], output rỗng. Chạy process mới với --phase resume và cùng thread_id=lab-review-1: output=[A,B], next rỗng. Resume lần nữa không nhân đôi; thread khác bị từ chối vì không có checkpoint. lab-start.json, lab-resume.json, lab-replay.json và SQLite mẫu nằm trong snapshot.

## 13.3 Sử dụng AI và lỗi có bằng chứng

Trợ lý AI hỗ trợ rà soát, soạn bổ sung, đề xuất test và viết lab. Nhóm cần tự chạy, giải thích và xác nhận đóng góp khi bảo vệ; không mặc định coi nội dung AI là đã kiểm chứng.

Hai lỗi Gemini có nguồn ở mục 12.10.6 là almost 3 thành 3 và thời gian học bị coi là kinh nghiệm. Sai lệch có thể lan sang giải thích/câu hỏi; văn bản trôi chảy không chứng minh dữ liệu đúng.

Mã lab do AI soạn ban đầu dùng SQLite với check_same_thread=True, gây ProgrammingError khi checkpoint chạy ở background thread. Đã sửa thành check_same_thread=False như ứng dụng rồi chạy lại start/resume/replay và regression tests. Không mô tả lỗi kỹ thuật này thành LangGraph tự mất state. Bộ test mới đạt 119 passed và 15 subtests; frontend đạt 5 tests. Đây là lượt test mã nguồn bổ sung, không thay số 107 + 12 trong lượt thực nghiệm trước.

## 13.4 Nhật kí kỹ thuật và xác nhận đóng góp

| Mục | Kết quả truy vết | Căn cứ |
|---|---|---|
| J1 Kiến trúc nền | Graph, cấu hình model, SQLite và ingest hai pha | Git history và source snapshot |
| J2 Pilot A | 19 xử lý + 1 already_ready; Top-10/60 câu hỏi | pilot-A-20 và search a91025e0... |
| J3 Pilot bổ sung | 20 xử lý; nạp lại bỏ qua 20; Top-1/6 câu hỏi | pilot-B-20, pilot-B-resume, search 13dbc9cc... |
| J4 Hoàn thiện bàn giao | Điểm mềm/outcome, phân tích lỗi, lab, test hồi quy | Source, tests.log, evidence-review.json và lab output |

Đây là nhật kí kỹ thuật tổng hợp từ artifact, không tự nhận là nhật kí cá nhân được viết theo từng ngày. Chưa có xác nhận phân công của thành viên; không suy từ tài khoản commit để gán công lao.

| Thành viên trên bìa | Thông tin cần tự xác nhận trước nộp |
|---|---|
| Dư Nguyễn Đăng Khoa | Đầu việc, file/artifact và mức đóng góp |
| Trần Đăng Khoa | Đầu việc, file/artifact và mức đóng góp |
| Mai Vũ Trung Tín | Đầu việc, file/artifact và mức đóng góp |

## 13.5 Bản bàn giao và khả năng tái lập

Manifest snapshot ghi HEAD thực tế, trạng thái working tree và SHA-256 source/artifact. HEAD chỉ đại diện nội dung đã commit; nếu source chưa commit, phải kèm patch và ghi rõ, không dùng HEAD cũ giả đại diện mã mới. Gói gồm toàn bộ graph, frontend, config, uv.lock, tests, lab, báo cáo, corpus pilot 20 CV, checkpoint mẫu và bằng chứng được chọn; không chứa .env, API key, virtualenv hay toàn corpus.

Lượt hoàn thiện dùng dữ liệu đã có và test xác định, không gửi request cloud hoặc mở ingest mới. Phần chưa thể xác nhận thay nhóm là nhật kí cá nhân và đóng góp; provenance của lỗi lịch sử vẫn được đánh dấu thiếu. Không tự mở rộng auth/OCR/hard filter vào phạm vi.

Snapshot bàn giao ghi HEAD nền bd29fa7db06ce698b2c5caea1c2d7889d4d0242e và working_tree_dirty=true. Không tạo được commit mới do quyền ghi .git/index.lock bị từ chối. Vì vậy HEAD này không đại diện riêng cho code mới; source đầy đủ, working-tree.patch và manifest hash trong gói snapshot mô tả đúng trạng thái bàn giao.
