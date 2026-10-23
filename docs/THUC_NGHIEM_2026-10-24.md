# **Chương 12 Triển khai tích hợp và minh chứng kết quả** {#chương-12-triển-khai-tích-hợp-và-minh-chứng-kết-quả}

Cập nhật thực nghiệm ngày 24/10/2026 (Asia/Saigon). Phần kết quả ở chương này sử dụng 100.000 hồ sơ ứng viên thật, thay cho bộ số liệu ba CV mẫu của bản trước. Các kiểm thử logic có fixture ở những chương trước vẫn là kiểm thử có kiểm soát; chúng không được coi là bằng chứng gọi model thật.

## **12.1 Hiện thực hóa trong repository và phạm vi kiểm chứng** {#12.1-hiện-thực-hóa-trong-repository-và-phạm-vi-kiểm-chứng}

Mã nguồn cục bộ đã thay hai graph stub bằng luồng ingest/search, bổ sung client OpenAI, Chroma và cơ chế HR duyệt bằng SQLite checkpoint. Đây là mô tả hiện thực đã viết trong repository; lượt chạy hiện tại chưa xác minh tích hợp LangGraph/Chroma/OpenAI vì chưa cài được phụ thuộc. Các thay đổi chưa commit; HEAD bf98fa0f74aa xác định nền mã ban đầu, không đại diện cho các thay đổi đang thử nghiệm.

| Hạng mục | Hiện thực trong mã nguồn | Bằng chứng của lượt này |
| :---- | :---- | :---- |
| Ingest | Send theo CV; đọc PDF hoặc văn bản corpus; extract–validate tối đa hai lần; lưu/index hoặc failed | Đã kiểm tra mã nguồn; chưa chạy luồng model thật |
| Search và review | Parse tiêu chí; truy xuất Chroma; score/rank; explain; interrupt human\_review và Command resume; chỉ sinh câu hỏi cho ID đã duyệt | Đã viết mã; pytest tích hợp chưa chạy |
| OpenAI | gpt-4o-mini cho tác vụ cấu trúc; text-embedding-3-small cho embedding; timeout 30 giây, client retry 1 | Đã có cấu hình khóa; chưa xác thực key hoặc gọi model thành công |
| Dữ liệu và giao diện | Danh sách CV phân trang; importer bỏ qua hàng ready có cùng văn bản đã lọc; lọc liên hệ trước ingest | Kiểm thử SQLite, bộ lọc, lấy mẫu và giao diện đạt như mục 12.5 |

Công thức nghiệp vụ giữ 0.5 × skill\_fit \+ 0.3 × similarity \+ 0.2 × experience\_fit; đây không phải xác suất phù hợp tuyển dụng. Kiểm thử công thức và dữ liệu thiếu được thực thi riêng, không dùng kết quả fixture làm đầu ra OpenAI.

## **12.2 Nguồn dữ liệu thật và môi trường đo** {#12.2-nguồn-dữ-liệu-thật-và-môi-trường-đo}

| Thành phần | Quan sát / số lượng |
| :---- | :---- |
| Môi trường | Windows; Python 3.11.9; SQLite 3.45.1; CPU logic 28; RAM vật lý 15.84 GiB |
| CPU do runtime báo cáo | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| Dataset | lang-uk/recruitment-dataset-candidate-profiles-english; dataset card khai báo MIT |
| Dòng Parquet nguồn | 210,250 |
| Hồ sơ tiếng Anh hợp lệ, duy nhất | 209,638 |
| Loại do độ dài CV | 612 |
| Mẫu dùng trong thực nghiệm | 100,000; reservoir sampling; seed 20261023 |
| Kích thước JSONL | 153,764,041 byte |

Bảng 12.1 Nguồn dữ liệu và môi trường thực nghiệm ngày 24/10/2026. Dataset là văn bản hồ sơ ứng viên đã được nguồn công bố là ẩn danh; không phải 100.000 PDF gốc và không phải CV sinh tổng hợp. Quét toàn bộ nguồn rồi lấy mẫu trên các hồ sơ đủ điều kiện; loại trùng theo source ID hoặc SHA-256 của nội dung đã casefold và chuẩn hóa khoảng trắng. Chỉ nhận CV tiếng Anh dài 80–24.000 ký tự trước khi thêm vị trí. Tên Anonymous candidate chỉ là ID hiển thị; kinh nghiệm nguồn không được dùng làm nhãn extraction.

Revision nguồn: f4d04f624126e27a17770d06953dce7e914ce214. SHA-256 JSONL: e446c3fa164a3003f453f28ad6130fbda79cd2f31f8b86620edabcf338984f8d. Không có nhãn mức phù hợp CV–JD để tính Precision, Recall hoặc nDCG.

## **12.3 Lọc thông tin liên hệ trước xử lý** {#12.3-lọc-thông-tin-liên-hệ-trước-xử-lý}

Bộ lọc contacts-v1 được áp dụng trước khi văn bản đi vào graph, extraction, embedding và cơ sở dữ liệu ứng dụng. File nguồn vẫn giữ nguyên cục bộ để đối chiếu; báo cáo chỉ lưu số đếm tổng hợp. Hàng ready cũ chứa văn bản chưa lọc sẽ được xử lý lại.

| Kết quả trên 100.000 hồ sơ | Giá trị |
| :---- | :---- |
| Hồ sơ có thay thế | 1,277 |
| URL được thay thế | 2,811 |
| Liên kết hồ sơ không có giao thức | 16 |
| Chuỗi giống số điện thoại | 8 |
| Liên hệ messenger có nhãn | 6 |
| Email khớp mẫu bộ lọc | 0 |
| Thời gian riêng cho bộ lọc | 17.916 giây |

Bảng 12.2 Kết quả bộ lọc theo mẫu. Các chuỗi phát hiện có thể bị nhận nhầm; đây chưa phải bảo đảm ẩn danh hoàn toàn. Tên, công ty, địa điểm, chi tiết dự án và thông tin liên hệ viết biến thể vẫn có thể nhận diện gián tiếp một người. Không đưa nội dung CV gốc, thông tin liên hệ hoặc khóa API vào báo cáo.

## **12.4 Baseline tìm kiếm từ khóa trên 100.000 hồ sơ đã lọc** {#12.4-baseline-tìm-kiếm-từ-khóa-trên-100.000-hồ-sơ-đã-lọc}

Đã tạo mới chỉ mục SQLite FTS5, tokenizer unicode61, truy vấn AND và xếp hạng BM25, lấy top 10\. Thời gian đọc JSONL, lọc liên hệ, chèn chỉ mục và commit: 26.305 giây, bao gồm 17.916 giây lọc. File SQLite sau checkpoint: 189,288,448 byte (180.52 MiB).

Mỗi truy vấn được đo 50 lần; tổng 250 lần đo cho 5 nhóm truy vấn. Số đếm hồ sơ khớp được tính trước lượt đo, nên kết quả là warm-cache. p50 là trung vị; p95 tính bằng nearest-rank.

| Truy vấn AND | Hồ sơ khớp | p50 (ms) | p95 (ms) |
| :---- | :---- | :---- | :---- |
| Python AND FastAPI | 494 | 0.940 | 2.486 |
| Java AND Spring | 4,035 | 3.411 | 4.314 |
| React AND TypeScript | 4,380 | 3.748 | 5.091 |
| Docker AND Kubernetes | 1,665 | 2.227 | 4.133 |
| SQL AND data | 4,363 | 6.534 | 8.062 |

Bảng 12.3 Số đo thực tế trên dữ liệu thật đã lọc. Phép đo này là tìm kiếm từ khóa cục bộ; thời gian truy vấn không gồm tạo chỉ mục, lọc, HTTP, LangGraph, Chroma, embedding hoặc LLM. Không có request OpenAI/embedding trong baseline. Số liệu không đo tốc độ hay chất lượng ANN, không suy ra cold-start hoặc tải nhiều người dùng, và không chứng minh độ phù hợp tuyển dụng.

## **12.5 Kết quả kiểm thử mã nguồn đã thực thi** {#12.5-kết-quả-kiểm-thử-mã-nguồn-đã-thực-thi}

| Nhóm kiểm thử | Đạt / tổng |
| :---- | :---- |
| Công thức score và dữ liệu thiếu | 5 / 5 |
| SQLite: giao dịch, review, phân trang | 7 / 7 |
| Lấy mẫu và loại trùng dữ liệu | 2 / 2 |
| Lọc liên hệ và biên dữ liệu ingest | 5 / 5 |
| Giao diện bằng Node test, DOM giả lập | 4 / 4 |
| Tổng đã thực thi | 23 / 23 |

Bảng 12.4 Các ca kiểm thử đã chạy trong repository. 19 kiểm thử Python unittest hoàn tất trong 0.453 giây; 4 kiểm thử Node hoàn tất trong 83.026 ms. Kiểm thử giao diện dùng DOM giả lập, chưa phải browser end-to-end. Bộ pytest cho API/LangGraph/Chroma/checkpoint chưa chạy được vì thiếu phụ thuộc. Tỷ lệ đạt chỉ phản ánh các assert của bộ kiểm thử này, không phải độ chính xác của model.

## **12.6 Kết quả thử chạy OpenAI với 100 hồ sơ** {#12.6-kết-quả-thử-chạy-openai-với-100-hồ-sơ}

Đã thực thi .\\scripts\\run-real-data.ps1 \-Limit 100\. Bước uv sync \--upgrade \--extra research dừng khi truy cập PyPI do lỗi DNS 11001\. Terminal thực thi tự động cũng không phân giải được api.openai.com. PowerShell thông thường của người dùng đã phân giải được cả hai tên miền; hai môi trường chưa có cùng kết quả kết nối, và DNS thành công chưa xác thực được API key.

| Hạng mục | Kết quả quan sát |
| :---- | :---- |
| API key có cấu hình | Có; giá trị khóa không được ghi lại |
| Cập nhật thư viện / uv.lock | Chưa hoàn tất do DNS 11001 |
| Pilot OpenAI | Bị chặn trước bước xử lý CV |
| Hồ sơ đã xử lý / yêu cầu | 0 / 100 |
| Ready / failed phát sinh | 0 / 0; chưa bắt đầu, không phải tỷ lệ thành công |
| Request model / embedding trong pilot | 0 / 0 |
| Token, chi phí, độ trễ và chất lượng model | Chưa đo |
| Chạy 100.000 hồ sơ qua OpenAI | Chưa thực hiện |

Bảng 12.5 Trạng thái pilot thực tế. Runner trực tiếp trả status=blocked, exit code=2, thiếu dotenv, langgraph, langchain\_openai, chromadb và fastapi. Chưa có phản hồi model, usage từ provider hoặc trace API thành công để xác nhận sử dụng model thật. Việc có khóa và mã gọi API không thay cho bằng chứng thực thi. Kết quả fixture ở Chương 4–5 và tab Minh chứng tuần 7 vẫn giữ đúng phạm vi kiểm thử có kiểm soát.

## **12.7 Tái lập, minh chứng và giới hạn kết luận** {#12.7-tái-lập,-minh-chứng-và-giới-hạn-kết-luận}

Lệnh baseline đã chạy:

python \-m scripts.benchmark\_corpus \--redact-contacts \--database data/benchmarks/djinni\_100k\_filtered\_20261024.sqlite \--report data/benchmarks/djinni\_100k\_filtered\_20261024.json \--repeats 50

Script không ghi đè database đã có; lần chạy lại cần chọn đường dẫn database và report mới. Khi terminal truy cập được PyPI/OpenAI, lệnh .\\scripts\\run-real-data.ps1 \-Limit 100 sẽ cập nhật phụ thuộc, chạy pytest, kiểm tra API rồi xử lý pilot. Sau khi có số liệu ready/failed, lỗi, usage và chi phí mới mở rộng tới \-Limit 100000; chưa mở rộng trong lượt thực nghiệm này.

Minh chứng nằm trong repository: docs/THUC\_NGHIEM\_2026-10-24.md; data/benchmarks/experiments\_20261024.json; data/benchmarks/djinni\_100k\_filtered\_20261024.json; data/benchmarks/pilot\_100\_20261024.json; data/benchmarks/runtime-check.json. JSON tổng hợp lưu SHA-256 của ba script benchmark, bộ lọc và importer. Chưa có commit triển khai, kết quả CI hoặc Docker build cho lần cập nhật này.

Cần bổ sung phản hồi model thật, model/snapshot, input/output tokens, số request, độ trễ, lỗi, tỷ lệ CV thành công và trace search/review/restart. Precision@k, Recall@k, nDCG@k cần nhãn CV–JD độc lập. Giữ riêng ba loại bằng chứng: baseline dữ liệu thật, kiểm thử logic có fixture và thực nghiệm model thật. Hiện đã có hai loại đầu; loại thứ ba còn bị chặn.

Nhật ký thực nghiệm ngày 24/10/2026: thẻ Thực nghiệm 100k CV — 24-10-2026 trong cùng tài liệu. Nhật ký fixture tuần 7 được giữ để đối chiếu lịch sử, không được diễn giải lại thành phản hồi OpenAI.

## **12.8 Nguồn tham khảo** {#12.8-nguồn-tham-khảo}

Dataset: https\://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english

Drushchak & Romanyshyn (2024), Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings. https\://aclanthology.org/2024.unlp-1.2/

### 

## **12.9 Chuyển nhà cung cấp API sang Google Gemini** {#12.9-chuyển-nhà-cung-cấp-api-sang-google-gemini}

Cập nhật ngày 24/10/2026 sau lượt thực nghiệm trên: cấu hình hiện hành đã chuyển từ OpenAI sang Gemini. Mục này thay thế thông tin nhà cung cấp hiện hành ở mục 12.1. Các kết quả FTS5 và nhật ký pilot OpenAI ở mục 12.6 được giữ theo đúng lượt đo cũ, không được diễn giải lại thành kết quả Gemini.

| Thành phần | Cấu hình / trạng thái mới |
| :---- | :---- |
| Model tác vụ cấu trúc | gemini-3.8-flash; Gemini Developer API; native JSON schema |
| Embedding | gemini-embedding-2; 768 chiều; chuẩn hóa vector; tiền tố riêng cho query và document |
| Thư viện khai báo | langchain-google-genai \>=4.4.2,\<5 và google-genai \>=2.20,\<3; chưa cài thành công trong terminal này |
| Xác thực | GEMINI\_API\_KEY hoặc GOOGLE\_API\_KEY từ Google AI Studio; không dùng OpenAI key |
| Collection mặc định | cv\_gemini\_embedding\_2\_768\_v1; kiểm tra provider/model/số chiều/định dạng; không trộn vector cũ |
| Kiểm tra model | python \-m scripts.check\_gemini\_model; lưu data/benchmarks/gemini-model-check.json |
| Kết quả kiểm thử | 32 unittest Python và 4 kiểm thử Node đạt; các biên API dùng test double |
| Trạng thái gọi Gemini thật | missing\_api\_key; 0 request được gửi trong lần kiểm tra này |
| Cập nhật phụ thuộc / uv.lock | Đã thử uv sync; còn bị chặn bởi DNS 11001 khi truy cập PyPI |

Importer giữ bước lọc liên hệ trước khi gửi CV tới Gemini. Hàng ready chỉ được tái sử dụng nếu văn bản khớp và có vector trong collection hiện tại; vì vậy các hàng đã index bằng nhà cung cấp cũ không bị bỏ qua nhầm. Ngưỡng similarity 0.55 vẫn là tham số thực nghiệm và cần hiệu chỉnh trên embedding Gemini.

Cần điền Google key vào .env, chạy kiểm tra model, rồi chạy scripts/run-real-data.ps1 \-Limit 100 để thu phản hồi, usage và độ trễ thật. Chưa có bằng chứng extraction, embedding hoặc workflow CV với Gemini thật; pytest tích hợp cũng chưa chạy do thiếu phụ thuộc. Nhật ký kiểm chứng chuyển đổi: data/benchmarks/gemini-migration-checks.json.

Nguồn: https\://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash ; https\://ai.google.dev/gemini-api/docs/embeddings ; https\://pypi.org/project/langchain-google-genai/4.4.2/