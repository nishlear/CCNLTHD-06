**ỦY BAN NHÂN DÂN TP HỒ CHÍ MINH**  
**TRƯỜNG ĐẠI HỌC SÀI GÒN**  
**KHOA CÔNG NGHỆ THÔNG TIN**

|  |  |
| ----- | :---: |
|  |  |

   
   
![][image1] 

**TÌM HIỂU VỀ CÔNG NGHỆ LANGGRAPH**

   
                                    	  
**Họ và tên sinh viên :**   
**Dư Nguyễn Đăng Khoa \- 3122410183**  
**Trần Đăng Khoa \- 3121410275**  
**Mai Vũ Trung Tín \- 3121410506**

**GVHD: Phạm Thi Vương**

   
 

***TP. Hồ Chí Minh, tháng 9 năm 2026***

# **MỤC LỤC**

Lời mở đầu

Phần 1 Tổng quan và phân tích công nghệ

Chương 1 Giới thiệu về LangGraph

Chương 2 Vị thế công nghệ và phạm vi ứng dụng

Phần 2 Kiến thức và thực hành trên ứng dụng

Chương 3 Môi trường và cấu hình thực thi

Chương 4 State node và điều phối hai pha

Chương 5 Checkpoint và sự phê duyệt của con người

Phần 3 Ứng dụng JD Matching

Chương 11 Kiến trúc và hợp đồng dữ liệu hiện hành

Chương 12 Triển khai tích hợp và minh chứng kết quả

12.1 Hiện thực hóa hệ thống và kiến trúc thực thi

12.2 Môi trường, nguồn dữ liệu và tiền xử lý thông tin liên hệ

12.3 Kết quả thực nghiệm tổng hợp

12.4 Phân tích cơ chế điều phối và nghiên cứu tình huống

12.5 Giới hạn kết luận, sự cố lịch sử và khả năng tái lập

Phần IV Tự đánh giá và bàn giao

Tài liệu tham khảo

# **LỜI MỞ ĐẦU**

Báo cáo trình bày cách dùng LangGraph để điều phối hệ thống JD Matching từ nạp CV, trích xuất thông tin, tạo chỉ mục vector đến tìm kiếm, HR duyệt và sinh câu hỏi phỏng vấn. Bản cập nhật ngày 24/10/2026 đối chiếu mã nguồn với giao diện đang chạy, dữ liệu SQLite, Chroma và lịch sử checkpoint. Hệ thống đã hoàn thành luồng tìm kiếm và duyệt bằng Gemini thật; chất lượng sàng lọc vẫn cần cải thiện ở cách hiểu kinh nghiệm và điều kiện lọc.

## **1 Lý do chọn đề tài**

CV có thể thiếu thông tin, file PDF có thể lỗi, tác vụ gọi API có thể quá hạn và kết quả phải chờ con người duyệt. Một lời gọi LLM đơn lẻ không xử lý được toàn bộ các tình huống này. Nhóm chọn bài toán CV để nghiên cứu State, node, cạnh điều kiện, xử lý song song và checkpoint trong một ứng dụng có đầu vào, đầu ra và trạng thái nghiệp vụ cụ thể.

LangGraph được dùng làm tầng điều phối; Gemini cung cấp extraction và sinh văn bản; Chroma lưu vector; SQLite lưu dữ liệu nghiệp vụ và checkpoint. Phạm vi không bao gồm huấn luyện mô hình, OCR cho PDF ảnh scan, tài khoản người dùng hoặc quyết định tuyển dụng tự động.

## **2 Mục tiêu báo cáo**

Các mục tiêu đã có minh chứng gồm tổ chức graph bằng TypedDict và reducer, xử lý CV theo hai pha, tìm kiếm có giải thích, dừng tại human\_review, tiếp tục bằng Command và lưu câu hỏi cho các CV được duyệt. Kết quả được kiểm tra bằng dữ liệu thật đã nạp và đầu ra ứng dụng, không thay phản hồi model bằng chuỗi viết sẵn.

Các mục tiêu còn cần đánh giá thêm gồm độ chính xác extraction, khả năng hiểu điều kiện nghề nghiệp, lọc cứng kinh nghiệm, Precision hoặc Recall với nhãn CV–JD và khả năng chịu tải. Không suy ra các chỉ số này từ số ca kiểm thử mã nguồn hoặc từ một lượt demo.

## **3 Phạm vi nghiên cứu**

Ứng dụng sử dụng Python 3.11, FastAPI, LangGraph, SDK Google Gen AI, Ollama, Chroma và SQLite. Cấu hình chỉ cho phép một cloud chat model, một local fallback và một embedding model. Corpus đã chuẩn bị có 100.000 hồ sơ; pilot model thật trong báo cáo chỉ xét 20 hồ sơ được chọn, không phải 100.000 hồ sơ đã vector hóa.

## **4 Tiến độ và hiện trạng**

| Hạng mục | Hiện trạng ngày 24/10/2026 |
| ----- | ----- |
| Nền tảng API và giao diện | Khởi chạy trên 127.0.0.1:8000; tải CV, tìm kiếm, lịch sử và duyệt hoạt động |
| Graph ingest | Đã triển khai extraction/cache rồi embedding/upsert; có checkpoint ready từng CV |
| Graph search và review | Đã xác nhận interrupt, duyệt 10 CV/60 câu hỏi và chạy lại Top 1/6 câu hỏi |
| Kiểm thử | Terminal người dùng ghi 107 passed và 12 subtests passed trong 7,46 giây |
| Đánh giá chất lượng | Đã ghi nhận các hạn chế; chưa có bộ nhãn CV–JD và benchmark chất lượng truy hồi |

# **PHẦN 1: TỔNG QUAN VÀ PHÂN TÍCH CÔNG NGHỆ**

# **Chương 1: Giới thiệu về LangGraph**

### **1.1. Bối cảnh và lịch sử**

LangGraph được phát triển bởi công ty LangChain (đội ngũ đứng sau framework LangChain nổi tiếng). Công nghệ này được giới thiệu và ra mắt chính thức vào đầu năm 2024\. LangGraph cho phép biểu diễn các bước xử lý và điều kiện chuyển bước bằng đồ thị có trạng thái. Trong bài toán CV, vòng lặp kiểm tra dữ liệu và điểm dừng chờ HR là hai lý do sử dụng tầng điều phối; khả năng kiểm soát phụ thuộc vào schema, router và giới hạn được người phát triển triển khai.

### **1.2. Kiến trúc và nguyên lý hoạt động**

**Theo tài liệu chính thức, LangGraph là framework điều phối ở mức thấp đồng thời là runtime cho các agent chạy lâu và có trạng thái. Graph có thể kết hợp hàm Python xác định với bước dùng LLM; không bắt buộc dùng LangChain hoặc LLM cho mọi node. Nguồn: LangGraph overview, https\://docs.langchain.com/oss/python/langgraph/overview (truy cập 17/10/2026).** **Mô hình hoạt động và thành phần chính:** LangGraph hoạt động dựa trên cấu trúc đồ thị trạng thái (State Graph). Các thành phần cốt lõi bao gồm:

\* **State (Trạng thái):** Một cấu trúc dữ liệu trung tâm được cập nhật liên tục qua mỗi bước, giúp toàn bộ hệ thống chia sẻ bối cảnh chung.

\* **Nodes (Đỉnh):** Các hàm thực thi logic (ví dụ: gọi LLM, gọi Tool, phân tích dữ liệu). Mỗi node nhận State hiện tại, xử lý và trả về dữ liệu để cập nhật State.

\* **Edges (Cạnh):** Định hướng luồng đi từ Node này sang Node khác. Điểm mạnh của LangGraph là "Conditional Edges" giúp rẽ nhánh động (ví dụ trong JD Matching: preprocess đi validate khi có cache, đi extract khi chưa có cache; validate đi mark_failed nếu thiếu trường). Ứng dụng dùng router Python và cạnh đã khai báo, không dùng bind_tools/ToolNode để model tự chọn công cụ.

\* **Checkpointer:** Hệ thống lưu vết bộ nhớ, cho phép tạm dừng luồng chạy để chờ con người phê duyệt (Human-in-the-loop) hoặc quay lại các trạng thái trong quá khứ (Time travel).

## **1.3. So sánh và đánh giá**

Bảng 1.1 So sánh mô hình điều phối theo tài liệu chính thức; không phải benchmark hiệu năng hoặc xếp hạng chất lượng agent.

| Lựa chọn | Cách tổ chức | Cân nhắc cho hệ thống CV |
| ----- | ----- | ----- |
| LangGraph | State, node, cạnh, checkpoint và interrupt | Phù hợp minh chứng vòng lặp và HR duyệt; cần tự thiết kế router/schema |
| LangChain / LCEL | RunnableSequence và RunnableParallel để ghép các bước | Có thể dùng cho chuỗi gọi model; việc lưu trạng thái và vòng lặp nghiệp vụ cần thiết kế thêm |
| LlamaIndex Workflows | Các step giao tiếp bằng event có kiểu | Là lựa chọn khác cho workflow nhiều bước và dữ liệu; cần đánh giá tích hợp với ứng dụng |
| CrewAI Flows | Luồng theo sự kiện, state, rẽ nhánh và persistence | Có cơ chế điều phối; không kết luận “cứng nhắc” nếu chưa thử trong cùng bài toán |
| AutoGen | Tương tác giữa các agent; repository hiện ghi maintenance mode | README khuyến nghị người dùng mới xem Microsoft Agent Framework; cần kiểm tra hướng chuyển đổi |
| Python tự viết vòng lặp | Hàm, if/while, DB và cơ chế retry do nhóm xây dựng | Ít phụ thuộc cho quy trình nhỏ; nhóm tự chịu trách nhiệm resume, trace và quản lý state |

Nguồn đối chiếu ngày 17/10/2026: LangGraph overview (docs.langchain.com/oss/python/langgraph/overview); LangChain Runnable (reference.langchain.com/v0.3/python/core/runnables/langchain\_core.runnables.base.Runnable.html); LlamaIndex Workflows (developers.llamaindex.ai/python/framework/getting\_started/concepts/); CrewAI Flows (docs.crewai.com/en/concepts/flows); Microsoft AutoGen README (github.com/microsoft/autogen). Các đường dẫn đầy đủ được liệt kê cuối báo cáo. Lựa chọn Python tự viết là phương án kiến trúc do nhóm phân tích.

**2 trường hợp KHÔNG NÊN dùng LangGraph:**

1\. **Các pipeline tuyến tính đơn giản (như RAG cơ bản):** Nếu ứng dụng chỉ nhận câu hỏi, tìm kiếm vector và sinh câu trả lời một chiều (không có vòng lặp, không gọi tools), việc sử dụng LangGraph sẽ làm phức tạp hóa mã nguồn một cách không cần thiết.

2\. **Ứng dụng một lần gọi và không cần lưu trạng thái có thể dùng hàm Python hoặc pipeline đơn giản. Cần cân nhắc chi phí thiết kế graph; báo cáo chưa đo độ trễ hay tài nguyên để kết luận LangGraph luôn tốn hơn trong mọi trường hợp.**

# **Chương 2: Vị thế công nghệ và phạm vi ứng dụng**

### **2.1. Bối cảnh và trường hợp ứng dụng tiêu biểu:**

LangGraph là một framework điều phối (orchestration framework) ở mức thấp dành cho việc xây dựng các AI Agent có trạng thái, đặc biệt phù hợp với những workflow có nhiều bước, cần duy trì trạng thái, khả năng khôi phục tiến trình và có sự tham gia của con người. Theo tài liệu của LangChain, LangGraph tập trung vào các đặc tính như durable execution, persistence, human-in-the-loop, streaming và khả năng xây dựng workflow kết hợp giữa các bước xác định trước với các bước do Agent điều khiển.

Một số trường hợp ứng dụng tiêu biểu gồm:

1\. Hệ thống xử lý và tìm kiếm CV: LangGraph biểu diễn quy trình nhận PDF → trích xuất → kiểm tra dữ liệu → lập chỉ mục → truy xuất → chấm điểm → HR duyệt → sinh câu hỏi phỏng vấn. CV thiếu thông tin được đánh dấu failed; lỗi quota hoặc timeout ở extraction có thể chuyển sang model local; shortlist cần chờ phản hồi của HR trước khi đi tiếp.

Trong ứng dụng của nhóm, một lô CV có thể được chia thành các nhánh xử lý độc lập. Trạng thái từng CV và kết quả tìm kiếm được hợp nhất có kiểm soát; lỗi một file không nên làm mất kết quả của toàn bộ lô. Đây là lý do sử dụng đồ thị trạng thái thay cho chỉ một lời gọi LLM.

2\. Customer Support Agent với Human-in-the-loop: Một trường hợp khác là hệ thống hỗ trợ khách hàng có các thao tác nhạy cảm cần con người phê duyệt. Ví dụ, Agent tiếp nhận yêu cầu hoàn tiền, tra cứu chính sách và kiểm tra thông tin đơn hàng. Nếu yêu cầu đáp ứng điều kiện, workflow có thể chuyển đến bước tạo lệnh hoàn tiền nhưng tạm dừng để nhân viên quản lý kiểm tra và phê duyệt trước khi thực hiện thao tác cuối cùng.

LangGraph hỗ trợ trực tiếp mô hình này thông qua cơ chế interrupt và persistence/checkpointing. Workflow có thể tạm dừng, lưu trạng thái và tiếp tục từ vị trí trước đó sau khi nhận được quyết định của con người.

### **2.2. Tình hình sử dụng và nhu cầu hiện tại:**

Nhu cầu xây dựng ứng dụng LLM nhiều bước đặt ra yêu cầu theo dõi trạng thái, xử lý lỗi và quản lý các điểm phê duyệt. Đối với hệ thống CV của nhóm, giá trị của tầng điều phối nằm ở khả năng kết nối các bước trích xuất, truy xuất, xếp hạng và HR duyệt theo luồng có thể quan sát và kiểm thử.

Đối với LangGraph, xu hướng phát triển tập trung vào khả năng đưa Agent từ giai đoạn thử nghiệm sang môi trường triển khai thực tế. Tài liệu chính thức hiện mô tả LangGraph là runtime cho các custom agent workflow, cung cấp persistence, fault tolerance, human-in-the-loop và khả năng triển khai các workflow dài hạn, có trạng thái.

Điều này cho thấy LangGraph đặc biệt hướng tới nhóm bài toán mà việc kiểm soát workflow và trạng thái có ý nghĩa quan trọng, thay vì chỉ đơn thuần tạo ra câu trả lời từ LLM.

### **2.3. Xu hướng phát triển và giới hạn**

Một xu hướng đáng chú ý là việc tăng cường các công cụ hỗ trợ phát triển và vận hành Agent. LangGraph Studio được giới thiệu như một môi trường trực quan cho phép nhà phát triển quan sát và tương tác với graph trong quá trình xây dựng Agent, hỗ trợ việc thử nghiệm và gỡ lỗi workflow.

Bên cạnh đó, LangGraph tiếp tục mở rộng khả năng xây dựng workflow. Functional API được giới thiệu năm 2025 cho phép sử dụng các cấu trúc quen thuộc như function, loop và conditional cùng với các khả năng persistence, human-in-the-loop và streaming của LangGraph. Điều này giúp giảm yêu cầu phải biểu diễn mọi logic dưới dạng Graph API truyền thống.

Tuy nhiên, LangGraph vẫn có một số giới hạn. Đây là framework ở mức tương đối thấp, do đó người phát triển phải chủ động thiết kế state, node, edge, cơ chế chuyển trạng thái, checkpoint và các điều kiện rẽ nhánh. Tài liệu chính thức cũng phân biệt LangGraph với các abstraction cấp cao hơn: LangGraph phù hợp khi ứng dụng yêu cầu workflow tùy biến cao, còn các framework cấp cao hơn phù hợp với những Agent có kiến trúc tiêu chuẩn và cần triển khai nhanh.

Ngoài ra, khi workflow trở nên phức tạp, số lượng node và nhánh điều khiển tăng lên có thể làm tăng chi phí thiết kế, kiểm thử và bảo trì. Đây là vấn đề đặc biệt đáng lưu ý đối với các nhóm chưa có nhiều kinh nghiệm về lập trình workflow hoặc hệ thống Agent.

# **PHẦN 2 KIẾN THỨC VÀ THỰC HÀNH TRÊN ỨNG DỤNG**

# **Chương 3 Môi trường và cấu hình thực thi**

## **3.1 Thành phần được sử dụng**

Môi trường dự án chạy bằng .venv và uv.lock; LangGraph đã được cài lại đúng phiên bản 1.2.16 sau khi phát hiện thiếu file trong bản cài cục bộ. Runner hiện dùng sync \--locked và link-mode copy, kiểm tra import trước pytest và cài lại package khi cần. Đây là sửa chữa môi trường, không phải thay thuật toán graph.

| Thành phần | Cấu hình hiện hành | Vai trò |
| ----- | ----- | ----- |
| Cloud chat | gemini-3.5-flash-lite, temperature 0 | Extract, parse\_query, explain và generate\_questions |
| Local chat | qwen2.5:3b qua Ollama | Dự phòng extraction khi cloud bị 429 hoặc timeout |
| Embedding | gemini-embedding-2, 768 chiều | Vector cho profile và truy vấn |
| Collection | cv\_gemini\_embedding\_2\_768\_v1 | Không trộn model hoặc số chiều |
| Local memory | keep\_alive 30m hoặc \-1; num\_ctx 4096 | Giữ cùng model và giới hạn context |
| Điều tiết cloud | Tối đa 12 RPM, cách nhau ít nhất 5 giây giữa các lượt cấp quyền gọi | Dùng chung giữa process cùng app.db |
| Batch embedding | Tối đa 20 CV/request | Tạo vector riêng cho từng CV rồi bulk upsert |

keep\_alive là yêu cầu gửi cho Ollama, không phải bảo đảm hệ điều hành không swap hoặc model không bị một client khác làm ảnh hưởng. Pilot ghi fallback\_ready=true, xác nhận warmup thành công; chưa có lượt CV thật phải chuyển sang Qwen trong pilot này.

## **3.2 Khởi chạy và kiểm tra**

Các lệnh được thực thi tại thư mục repository:

.\\scripts\\run-real-data.ps1 \-Limit 20

.\\.venv\\Scripts\\python.exe \-m uvicorn src.backend.main:app \--host 127.0.0.1 \--port 8000

Runner kiểm tra phụ thuộc, kiểm thử mã nguồn, kết nối và một phép inference MODEL\_OK trước khi xử lý dữ liệu. Phép MODEL\_OK không gửi CV và được ghi thành minh chứng riêng. Giao diện gồm CV, Tìm kiếm và Lịch sử; API tài liệu nằm tại /docs.

## **3.3 Dữ liệu và phân chia nơi lưu**

app.db lưu CV và kết quả tìm kiếm; extraction\_cache lưu JSON đã trích xuất để không phải gọi lại LLM khi bước embedding chưa hoàn thành. checkpoints.db lưu State của graph tìm kiếm theo thread\_id. Chroma lưu vector; thư mục uploads lưu PDF khi tải file qua API. JSONL của corpus là đầu vào riêng và không bị sửa nội dung gốc.

# **Chương 4 State node và điều phối hai pha**

## **4.1 State và reducer trong mã ứng dụng**

IngestState có files, prepared và processed. ProcessCVState giữ cv\_id, raw\_text, clean\_text, extracted, source, fingerprint và error cho một CV. Các nhánh trả prepared hoặc processed; operator.add hợp nhất danh sách thay vì để nhánh kết thúc sau ghi đè kết quả nhánh trước.

SearchState giữ query, criteria, candidates, results, shortlist, explanations, approved\_cv\_ids, approval\_status, questions và errors. results, explanations, questions và errors dùng reducer nối danh sách. rank sắp xếp theo điểm giảm dần và CV ID khi bằng điểm; không dựa vào thứ tự hoàn tất của nhánh song song.

## **4.2 Pha trích xuất và cache**

Mỗi CV đi qua đọc PDF hoặc nhận raw\_text, tiền xử lý, tìm cache, extract và validate. Regex che email, URL, số điện thoại và dọn khoảng trắng trước khi gửi vào LLM. Hợp đồng output dùng các khóa name, role, skills, exp\_yrs, edu và summary; Gemini và Qwen nhận cùng prompt/schema, sau đó cùng được Pydantic kiểm tra và ánh xạ sang cấu trúc lưu trữ.

Attempt 1 gọi Gemini với JSON MIME, temperature 0 và SDK không retry. Khi gặp 429 hoặc timeout, extraction chuyển ngay một lần sang Qwen với format=json; không chờ backoff cloud hoặc quay lại Gemini. Code cũng coi 503 overload là lỗi đủ điều kiện chuyển local. Nếu output sai JSON hoặc thiếu trường bắt buộc, CV đó failed và các CV khác tiếp tục.

JSON hợp lệ được commit ngay vào extraction\_cache. Fingerprint gắn CV ID với phiên bản hợp đồng, input đã tiền xử lý và tên hiển thị. Cache sai schema hoặc input thay đổi sẽ không được dùng lại.

## **4.3 Hàng rào giữa hai pha và batch vector**

Graph cha chỉ chạy node save\_and\_index sau khi tất cả nhánh process\_cv kết thúc. Vì vậy embedding không bắt đầu ngay sau extraction của từng CV. Mỗi nhóm input có hai pha riêng; runner mặc định chia nhóm 20\. Nếu chạy Limit 50, các nhóm 20, 20 và 10 được thực hiện lần lượt.

Pha 2 chỉ dùng profile cô đọng gồm role, skills, kinh nghiệm và summary, dưới 300 từ. Raw CV không được đưa vào embedding. Mỗi profile được bọc bằng một Content độc lập để batchEmbedContents trả một vector riêng, tránh gộp các CV thành một vector. Vector được kiểm tra số lượng, 768 chiều, giá trị hữu hạn và chuẩn hóa trước khi ghi Chroma.

Khóa file của hệ điều hành giữ các lượt ingest dùng chung app.db không xen kẽ hai pha. Pacer SQLite điều tiết chat giữa các process dùng cùng database; ứng dụng hoặc key được sử dụng từ database khác không nằm trong cơ chế phối hợp này. Quota chat và embedding do Google quản lý riêng, không tự chuyển phần tiết kiệm từ model này sang model khác.

## **4.4 Cô lập lỗi và lưu tiến độ**

Node indexing kiểm tra extracted là dict hợp lệ, khác rỗng; truy xuất trường bằng .get với mặc định đúng kiểu. Lỗi provider hoặc schema chỉ đánh dấu hồ sơ liên quan. Embedding gặp 429 sẽ chờ 10, 20, 40 giây, mỗi lần cộng jitter ngẫu nhiên 1–3 giây. Sau ba lần chờ mà vẫn lỗi, cache được giữ để lần sau tiếp tục.

Chroma được upsert theo batch. Nếu bulk upsert lỗi, code thử riêng từng CV bằng vector đã tạo để cô lập lỗi mà không gọi lại embedding. Mỗi CV thành công được commit ready ngay. Lần chạy sau bỏ qua ready có dữ liệu và vector hợp lệ, hoặc tái sử dụng extraction cache nếu chỉ bước vector chưa xong.

# **Chương 5 Checkpoint và sự phê duyệt của con người**

## **5.1 Graph tìm kiếm có trạng thái**

Luồng tìm kiếm hiện hành là parse\_query, retrieve, score\_cv, rank, explain và human\_review. parse\_query dùng Gemini để tạo SearchCriteria; retrieve tạo vector truy vấn và tìm Chroma; score\_cv dùng công thức Python xác định. Chỉ kết quả đạt ngưỡng 0,5 mới được gọi explain.

Nhóm hàm này là công cụ nghiệp vụ được node gọi theo graph. Ứng dụng không sử dụng vòng agent để LLM tự chọn tool bằng bind\_tools hoặc ToolNode. Vì vậy kết quả hiện tại không được mô tả là minh chứng model tự lập kế hoạch hoặc tự quyết định gọi công cụ.

## **5.2 Điểm dừng và tiếp tục đã kiểm chứng**

human\_review gọi interrupt với shortlist và explanations. search\_id được dùng làm thread\_id cho SqliteSaver. API trả awaiting\_review trong khi graph chờ quyết định; chưa có câu hỏi trước khi duyệt. Khi HR gửi approved\_cv\_ids, API nhận claim nguyên tử, kiểm tra ID thuộc shortlist rồi resume bằng Command với cùng thread\_id.

Lượt dev 3 năm có checkpoint dừng lúc 08:28:32 ngày 24/10/2026, Asia/Saigon, với next chứa human\_review, approval\_status=pending và 0 câu hỏi. Sau khi người dùng duyệt 10 CV, trạng thái là approved, next rỗng và 60 câu hỏi được lưu.

Lượt Top 1 kiểm chứng lại có checkpoint dừng lúc 08:35:47 cùng ngày, với 0 câu hỏi. Sau khi duyệt đúng một ứng viên, SQLite và checkpoint cùng ghi 1 CV được duyệt, 6 câu hỏi, không có lỗi và next rỗng. Đây là hai lượt khác nhau; không quy kết ảnh 60 câu hỏi là kết quả của thao tác duyệt một người.

## **5.3 Dữ liệu nghiệp vụ và graph**

app.db và checkpoints.db có vai trò khác nhau. Graph lưu tiến độ và quyết định; API lưu shortlist, cờ approved và câu hỏi để giao diện hoặc lịch sử đọc lại. Việc đọc lại bằng một tiến trình Python độc lập đã xác nhận checkpoint còn trên đĩa. Lượt kiểm chứng này không chủ động tắt và khởi động lại web server để tuyên bố một thử nghiệm restart riêng.

Một lượt completed chỉ cho biết graph đã đi đến cuối. Cần kiểm tra thêm errors và số câu hỏi: trong lịch sử có lượt duyệt một CV nhưng generate\_questions bị ConnectError, cho kết quả 0 câu hỏi. Hai lượt minh chứng mới đều không có lỗi này.

## **5.4 Giới hạn thực thi**

Recursion limit là 25 và max\_concurrency là 4\. Ingest không còn lặp extraction khi output thiếu trường; failover chỉ gồm một lần cloud và một lần local khi lỗi đủ điều kiện. Các request chat còn phải đi qua pacer 12 RPM, nên concurrency 4 không có nghĩa bốn request cloud được gửi đồng thời.

# **PHẦN 3 ỨNG DỤNG JD MATCHING**

# **Chương 11 Kiến trúc và hợp đồng dữ liệu hiện hành**

## **11.1 Chức năng và ranh giới nghiệp vụ**

HR tải CV PDF hoặc importer nhận văn bản corpus; hệ thống lưu hồ sơ đã trích xuất, tìm ứng viên bằng câu tự do, hiển thị điểm và lý do, chờ HR duyệt rồi sinh câu hỏi phỏng vấn. Người dùng quyết định CV được giữ lại. Một điểm cao không phải xác suất đạt yêu cầu và không thay thế việc kiểm tra kinh nghiệm hoặc chuyên môn.

Hệ thống hiện không có xác thực người dùng, OCR, kết nối hệ thống tuyển dụng ngoài hoặc tự gửi thư cho ứng viên. Không đưa các chức năng này vào phần đã hoàn thành.

## **11.2 Sơ đồ xử lý theo nhóm node**

| Giai đoạn | Trình tự thực thi | Nơi lưu đầu ra |
| ----- | ----- | ----- |
| Ingest pha 1 | parse\_pdf hoặc raw\_text → preprocess → cache hoặc extract → validate → cache\_extraction | extraction\_cache trong app.db |
| Ingest pha 2 | Hàng rào tất cả CV → batch embedding → upsert → ready từng CV | Chroma và bảng cv |
| Tìm kiếm | parse\_query → retrieve → score\_cv → rank → explain | Checkpoint và search\_result |
| Duyệt | human\_review interrupt → approved\_cv\_ids → Command resume → generate\_questions | Checkpoint, cờ approved và interview\_question |

Các node đọc PDF, xử lý chuỗi, tính điểm, SQLite và Chroma là code xác định; chỉ extract, parse\_query, explain và generate\_questions dùng chat model. Embedding là lời gọi model riêng, không phải một nhiệm vụ của chat model.

## **11.3 Hợp đồng dữ liệu**

| Đối tượng | Các trường chính | Kiểm tra |
| ----- | ----- | ----- |
| Output extraction | name, role, skills, exp\_yrs, edu, summary | JSON object, Pydantic, full\_name và skills sau mapping |
| CV lưu trữ | id, file\_name, raw\_text, full\_name, role, years\_experience, skills, education, summary, status, error | ready hoặc failed; có thể giữ unknown cho kinh nghiệm |
| SearchCriteria | skills, min\_years, level, keywords | Kiểu dữ liệu và biên số năm hợp lệ |
| Shortlist | cv\_id, rank, score, matched\_skills, missing\_skills, reasons, approved | ID thuộc kết quả tìm kiếm; thứ hạng ổn định |
| ReviewRequest | approved\_cv\_ids | Server lọc theo allowlist shortlist |
| Câu hỏi | cv\_id, category, question | technical, experience, behavioral; kết quả thật mỗi CV có 2 câu mỗi nhóm |

Hợp đồng extraction dùng khóa ngắn để giảm output token nhưng được ánh xạ sang tên trường đầy đủ trong SQLite. Không dựa vào tên model để quyết định cách node phía sau xử lý dữ liệu.

## **11.4 Công thức xếp hạng và ý nghĩa**

Điểm \= 0,5 × skill\_fit \+ 0,3 × similarity \+ 0,2 × experience\_fit.

skill\_fit là tỷ lệ kỹ năng yêu cầu khớp sau khi bỏ khoảng trắng thừa và đổi chữ hoa/thường. Nếu không yêu cầu kỹ năng, skill\_fit mặc định bằng 1\. experience\_fit bằng min(years\_experience/min\_years, 1\) khi có yêu cầu dương; kinh nghiệm thiếu được tính 0\. Nếu không yêu cầu kinh nghiệm, thành phần này bằng 1\.

Chroma dùng cosine distance. Code đổi similarity \= clamp(1 − distance/2, 0, 1), giữ ứng viên vượt MIN\_SIMILARITY=0,55 rồi mới xếp hạng. Threshold này là tham số thực nghiệm, chưa được hiệu chỉnh bằng nhãn tuyển dụng.

min\_years hiện chỉ ảnh hưởng điểm experience\_fit. Nó không phải điều kiện SQL hoặc bộ lọc loại cứng. keywords và level cũng chưa được kiểm tra như điều kiện bắt buộc trong score\_cv. Vì vậy truy vấn dev 3 năm có thể trả CV 2 năm, chưa rõ số năm hoặc thuộc lĩnh vực nghệ thuật; phần hạn chế ở Chương 12 giữ đúng hành vi quan sát này.

## **11.5 Dữ liệu bền vững và API**

Bốn bảng nghiệp vụ gồm cv, search, search\_result và interview\_question. Hai bảng bổ sung là extraction\_cache và api\_pacing. Chroma lưu vector và metadata định dạng compact-v2; checkpoint graph nằm trong SQLite riêng. Xóa CV qua API xóa vector, file tải lên, bản ghi CV và cache của CV đó; lịch sử tìm kiếm vẫn được giữ.

| API | Hành vi hiện hành |
| ----- | ----- |
| POST /cvs | Nhận nhiều PDF, chạy ingest hai pha, trả ready hoặc failed từng file |
| GET /cvs và GET /cvs/{id} | Phân trang danh sách và đọc hồ sơ |
| DELETE /cvs/{id} | Xóa CV và vector qua luồng ứng dụng |
| POST /searches | Tạo lượt tìm, chạy đến interrupt hoặc kết thúc nếu không có kết quả |
| POST /searches/{id}/review | Nhận lựa chọn, resume checkpoint và lưu câu hỏi |
| GET /searches và GET /searches/{id} | Mở lịch sử, shortlist, quyết định và câu hỏi đã lưu |
| GET /health | Kiểm tra web server |

Các giới hạn upload là 20 file/lần, 10 MB/file, 100 trang PDF, 24.000 ký tự văn bản; input chat được rút gọn thêm trước extraction. top\_k cho phép từ 1 đến 20\.

## **11.6 Phủ kỹ thuật và phạm vi kiểm chứng**

| Kỹ thuật | Mã nguồn | Minh chứng đã có |
| ----- | ----- | ----- |
| State và reducer | graphs/ingest.py, graphs/search.py | Danh sách CV, shortlist và câu hỏi hợp nhất trong dữ liệu thật |
| Hàng rào hai pha | graphs/ingest.py | Code extract/cache hoàn tất trước save\_and\_index batch |
| JSON schema chung | schemas.py, services.py | Cấu hình Gemini/Qwen chung hợp đồng; 19 CV extraction từ Gemini trong pilot |
| Vector batch | gemini\_embeddings.py, services.py | 20 vector thật, 768 chiều; code kiểm tra vector riêng từng CV |
| Human in the loop | graphs/search.py, routers/searches.py | Hai checkpoint pending trước duyệt; sau resume có 60 và 6 câu hỏi |
| Cache và checkpoint | db.py, graphs/search.py | Dữ liệu đọc lại được từ process độc lập |
| Chống quota | cloud\_pacing.py, config.py, services.py | Cấu hình và code; pilot khoảng 99 giây phù hợp pacing, chưa có timestamp toàn bộ request để đo RPM thực tế |

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

# PHẦN IV TỰ ĐÁNH GIÁ VÀ BÀN GIAO

## 13.1 Tự đánh giá và phạm vi hoàn thiện

Những cơ chế có thể trình diễn gồm TypedDict, reducer, fan-out, barrier, schema chung, cache/vector và checkpoint/human review. Chất lượng xếp hạng còn phụ thuộc extraction và trọng số chưa hiệu chỉnh bằng bộ nhãn. Hệ thống hỗ trợ HR trong phạm vi thử nghiệm, chưa có auth, OCR hoặc quyết định tuyển dụng tự động. Tests xác nhận logic và các nhánh đã kiểm tra, không đo độ đúng tuyển dụng hoặc LLM.

Ưu tiên tuần 8 là chốt ý nghĩa điểm mềm và diễn giải điều kiện chưa lọc cứng; dẫn output resume với cùng thread_id, giữ pilot 20 CV. Chốt tuần 9 là đóng gói corpus pilot, cấu hình/model, checkpoint mẫu, tests, lab và tự đánh giá/AI/nhật kí. Không tự đặt thêm ngày nộp ngoài mốc tuần 9 đã được phản hồi.

## 13.2 Lab state reducer và checkpoint

labs/state_checkpoint có starter.py, solution.py và README. Đây là chương trình xác định dùng LangGraph/SQLite thật, không phải output model. Starter yêu cầu hàm collect trả cập nhật danh sách một phần tử. Hai nhánh collected=[A] và [B] được reducer hợp nhất thành [A,B]; barrier cho review đọc đủ dữ liệu.

Thực hiện start và resume theo lệnh ở 12.5.3: next=[review] và output rỗng trước duyệt; process mới dùng cùng thread_id=lab-review-1 cho output=[A,B], next rỗng. Resume lần nữa không nhân đôi; thread khác bị từ chối vì không có checkpoint. lab-start.json, lab-resume.json, lab-replay.json và SQLite mẫu nằm trong snapshot.

## 13.3 Sử dụng AI và lỗi có bằng chứng

Trợ lý AI hỗ trợ rà soát, soạn bổ sung, đề xuất test và viết lab. Nhóm cần tự chạy, giải thích và xác nhận đóng góp khi bảo vệ; không mặc định coi nội dung AI là đã kiểm chứng.

Hai lỗi Gemini có nguồn ở mục 12.4.3 là almost 3 thành 3 và thời gian học bị coi là kinh nghiệm. Sai lệch có thể lan sang giải thích/câu hỏi; văn bản trôi chảy không chứng minh dữ liệu đúng.

Mã lab do AI soạn ban đầu dùng SQLite với check_same_thread=True, gây ProgrammingError khi checkpoint chạy ở background thread. Đã sửa thành check_same_thread=False như ứng dụng rồi chạy lại start/resume/replay và regression tests. Không mô tả lỗi kỹ thuật này thành LangGraph tự mất state. Kết quả kiểm thử hiện hành được tổng hợp tại Bảng 12.5; các mốc trước được giữ trong artifact lịch sử.

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

Định danh commit mã nguồn, dataset và hash được tập trung ở 12.5.3. Các chỉnh lý tài liệu sau commit đi kèm source và working-tree.patch; manifest xác định đúng trạng thái từng tệp của bản bàn giao.

# **TÀI LIỆU THAM KHẢO**

1\. LangGraph overview. https\://docs.langchain.com/oss/python/langgraph/overview

2\. LangGraph interrupts. https\://docs.langchain.com/oss/python/langgraph/interrupts

3\. LangGraph persistence. https\://docs.langchain.com/oss/python/langgraph/persistence

4\. Gemini embeddings. https\://ai.google.dev/gemini-api/docs/embeddings

5\. Djinni candidate profiles dataset. https\://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english

6\. Drushchak và Romanyshyn 2024\. Introducing the Djinni Recruitment Dataset. https\://aclanthology.org/2024.unlp-1.2/

7\. Mã nguồn và minh chứng repository CCNLTHD-06, đối chiếu ngày 24/10/2026.

# **PHỤ LỤC CHỈ MỤC ẢNH MINH CHỨNG**

Ảnh gốc và hai ảnh kiểm chứng lại được lưu trong data/benchmarks/proof/report-audit-20261024 của repository. Bảng dưới chỉ định chính xác trạng thái cần đối chiếu, không thay ảnh bằng output sinh sẵn.

| Tệp ảnh | Nội dung đối chiếu |
| :---- | :---- |
| Screenshot\_2026-10-24\_082917.png | dev 3 năm; Top 10; giao diện đang đợi review hoàn tất |
| Screenshot\_2026-10-24\_082952.png | 6 câu hỏi mỗi CV trong năm nhóm đầu |
| Screenshot\_2026-10-24\_083011.png | Năm nhóm còn lại; tổng 10 CV và 60 câu hỏi |
| top1-awaiting-review.jpg | c3adbe3a...; chờ duyệt; checkpoint pending và 0 câu hỏi |
| top1-completed.jpg | c3adbe3a...; 1 CV approved, completed và 6 câu hỏi |
