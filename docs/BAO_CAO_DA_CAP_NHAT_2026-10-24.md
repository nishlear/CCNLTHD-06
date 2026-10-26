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

Chương 12 Thực nghiệm model thật và đối chiếu minh chứng

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

# **Chương 12 Thực nghiệm model thật và đối chiếu minh chứng**

## **12.1 Phạm vi và nguồn bằng chứng**

Thực nghiệm ngày 24/10/2026 dùng dữ liệu hồ sơ thật đã ẩn danh từ Djinni và Gemini trong ứng dụng đang chạy. Minh chứng gồm log pilot, phản hồi MODEL\_OK, SQLite nghiệp vụ, Chroma, lịch sử checkpoint, ba ảnh người dùng cung cấp và lượt Top 1 chạy lại trực tiếp trên giao diện.

Corpus 100.000 hồ sơ, pilot 20 hồ sơ, lượt duyệt 10 người và lượt duyệt 1 người là các đơn vị khác nhau. Không cộng gộp chúng thành số CV đã chạy qua model hoặc dùng kết quả tìm kiếm từ khóa trên corpus để đại diện cho chất lượng vector.

## **12.2 Dữ liệu chuẩn bị và chỉ mục thực tế**

| Đại lượng | Kết quả đã ghi nhận |
| ----- | ----- |
| Nguồn | lang-uk/recruitment-dataset-candidate-profiles-english |
| Dòng Parquet nguồn | 210.250 |
| Hồ sơ hợp lệ duy nhất | 209.638; loại 612 do độ dài |
| Mẫu JSONL | 100.000 hồ sơ, reservoir sampling, seed 20261023 |
| SHA-256 mẫu | e446c3fa164a3003f453f28ad6130fbda79cd2f31f8b86620edabcf338984f8d |
| Chroma đã kiểm tra | 20 vector thuộc 20 hồ sơ được chọn |
| Số chiều và định dạng | 768, compact-v2; norm xấp xỉ 1 |

Dữ liệu là văn bản CV, không phải 100.000 PDF. Source revision đã ghi nhận là f4d04f624126e27a17770d06953dce7e914ce214. Tên Anonymous candidate là định danh hiển thị. Hồ sơ có thể đã cũ; không suy ra số năm hiện tại bằng cách lấy ngày hôm nay trừ một mốc Present trong CV.

Bộ lọc contacts-v2 che thông tin liên hệ trước extraction. Pilot ghi 1 hồ sơ có thay thế 1 URL. Thống kê cũ trên toàn bộ corpus dùng bộ lọc khác được giữ trong file benchmark để đối chiếu, không trộn với số đo pilot này.

Ở snapshot buổi sáng, app.db có 20 hàng ready và 60 hàng failed lịch sử. Đây không phải kết quả 60 lỗi của pilot mới; trong nhóm 20 được chọn, tất cả đều ready và có vector. Báo cáo không xóa dữ liệu nguồn hoặc lịch sử lỗi để làm đẹp tỷ lệ thành công.

## **12.3 Phản hồi Gemini và pilot 20 hồ sơ**

| Minh chứng | Giá trị |
| ----- | ----- |
| MODEL\_OK | succeeded, gemini-3.5-flash-lite, STOP |
| Thời điểm phép thử | 08:13:19 ngày 24/10/2026, Asia/Saigon |
| Response ID | LwbcapPuNO\_n4-EP9b6hyAQ |
| Token phép thử | 7 input, 3 output, 44 thinking; tổng 54 |
| Độ trễ phép thử | 1.853,097 ms |
| Pilot | completed; 19 xử lý, 19 ready, 0 failed; 1 already\_ready |
| Nguồn extraction mới | Gemini cho cả 19 hồ sơ |
| Local fallback | Warmup thành công; không có hồ sơ pilot chuyển sang Qwen |
| Thời gian pilot và truy vấn mẫu | 99,424 giây; riêng báo cáo giữa chừng 95,970 giây |

54 token và 1,853 giây chỉ thuộc request MODEL\_OK, không phải tổng chi phí hoặc latency xử lý 19 CV. Chưa có tổng usage và hóa đơn provider cho toàn bộ pilot. Số 19 là số hồ sơ được xử lý trong lượt này, không chứng minh đó là lần đầu các hồ sơ xuất hiện trong database.

## **12.4 Lượt tìm kiếm và duyệt trong ảnh người dùng**

Search ID: a91025e0f20f489db58dda8d4a739efe. Truy vấn dev 3 năm được phân tích thành skills rỗng, min\_years=3, level=null và keywords chứa dev. Top 10 có điểm chính xác từ 0,747582 đến 0,958391; giao diện làm tròn còn hai chữ số thập phân.

Checkpoint lịch sử ghi awaiting\_review trước quyết định, approval\_status=pending, next=\[human\_review\] và 0 câu hỏi. Cả 10 ô được chọn khi người dùng gửi duyệt. Bản ghi cuối cùng completed, approved cho 10 CV và có 60 câu hỏi: 20 technical, 20 experience, 20 behavioral. Mỗi CV có 6 câu, gồm 2 câu mỗi nhóm. Checkpoint cuối có next rỗng và errors rỗng.

Ba ảnh Screenshot 2026-10-24 082917.png, 082952.png và 083011.png thể hiện shortlist và các nhóm câu hỏi tương ứng. Trạng thái awaiting\_review còn hiện trong ảnh đầu cùng nút Đang tạo câu hỏi là trạng thái giao diện trong khi đợi HTTP review trả về; dữ liệu cuối cùng trong SQLite và checkpoint đã completed.

| Ứng viên trong shortlist | Kinh nghiệm lưu trữ | Điểm | approved |
| ----- | ----- | ----- | ----- |
| 62c4d272 | 3.0 | 0.958391 | Có |
| 700fef76 | 3.0 | 0.956896 | Có |
| 163fb9a3 | 18.0 | 0.950663 | Có |
| 3f56345b | 5.0 | 0.950326 | Có |
| 9f8cbaf0 | 3.0 | 0.948591 | Có |
| 4852f248 | 12.0 | 0.947847 | Có |
| 33d0892e | 10.0 | 0.943763 | Có |
| af8a1764 | 3.0 | 0.943605 | Có |
| 2bc3301a | 2.0 | 0.879044 | Có |
| dabeaff8 | Chưa rõ | 0.747582 | Có |

## **12.5 Lượt chạy lại trực tiếp Top 1**

Truy vấn mới bổ sung kỹ năng rõ ràng: Software engineer có ít nhất 3 năm kinh nghiệm thương mại, biết Akka, Netty và Kafka Streams. Top được đặt bằng 1\. Đây là một lượt tìm mới, không sửa kết quả dev 3 năm đã lưu.

Search ID: c3adbe3a46d348a1be2e0577737d9b4f. Kết quả là Anonymous candidate 3f56345b, Software Engineer, years\_experience=5, score=0.974037; giao diện hiển thị 0,97. CV nguồn ghi rõ 5+ years of commercial experience, nên có căn cứ đáp ứng điều kiện từ 3 năm kinh nghiệm làm việc. Các kỹ năng khớp là Akka, Netty và Kafka Streams.

Trước khi nhấn Gửi duyệt & tạo câu hỏi, checkpoint có next=\[human\_review\], approval\_status=pending và 0 câu hỏi. Sau thao tác trên giao diện, chỉ một CV được approved. Kết quả completed, 6 câu hỏi được lưu, mỗi nhóm technical, experience, behavioral có 2 câu; errors rỗng và next rỗng.

Ví dụ câu hỏi kỹ thuật trong kết quả mới đề cập cách dùng Akka xử lý dữ liệu bất đồng bộ trong backend IoT và tình huống ưu tiên Netty. Câu hỏi kinh nghiệm đề cập Kafka Streams và CI/CD. Một số đầu ra tiếng Việt chưa có đầy đủ dấu; đây là điểm chất lượng sinh văn bản cần cải thiện, không phải lỗi mất trạng thái.

Ảnh top1-awaiting-review.jpg và top1-completed.jpg ghi lại hai thời điểm. Các ảnh cùng dữ liệu SQLite/checkpoint giúp phân biệt trạng thái chờ duyệt và đầu ra đã sinh thành công.

## **12.6 Những giới hạn bộc lộ trong output**

Thứ nhất, min\_years chưa phải bộ lọc cứng. CV 2bc3301a có 2 năm vẫn ở hạng 9 với 0,879044; CV dabeaff8 chưa rõ số năm vẫn ở hạng 10 với 0,747582. Khi skills rỗng, skill\_fit=1 đóng góp sẵn 0,5 điểm. Điểm cao do đó không đồng nghĩa đáp ứng mọi điều kiện tuyển dụng.

Thứ hai, dữ liệu kinh nghiệm chưa phân biệt mức độ chắc chắn. CV 62c4d272 ghi almost 3 years nhưng được lưu thành 3,0; không đủ căn cứ kết luận ít nhất 3 năm. CV 700fef76 nói đã học lập trình hơn 3 năm; con số này chưa chứng minh 3 năm kinh nghiệm làm việc thương mại. Lời giải thích và câu hỏi có thể kế thừa sai lệch của extraction.

Thứ ba, từ khóa dev quá rộng, skills rỗng và level chưa được dùng như điều kiện lọc bắt buộc. Shortlist có nghệ sĩ, quản lý và hỗ trợ khách hàng. Đây là đầu ra truy hồi/xếp hạng hiện tại, không phải bằng chứng hệ thống đã nhận diện đúng nghề developer cho mọi kết quả.

Thứ tư, giao diện tự chọn sẵn các CV có score từ 0,5. Trong lượt dev 3 năm, cả 10 người được chọn dù có người thiếu hoặc không đạt kinh nghiệm. Việc bấm duyệt vẫn là hành động của HR nhưng cần hỗ trợ người dùng tốt hơn bằng cách hiển thị lý do không đạt và không mặc định xem điểm là đủ điều kiện.

Các hướng cần bổ sung là tách điều kiện bắt buộc khỏi điểm mềm, lưu bằng chứng và độ chắc chắn cho số năm, phân biệt học tập với làm việc, kiểm tra nghề/vai trò, và đánh giá bằng bộ CV–JD có nhãn. Những thay đổi này chưa được ghi là đã hoàn thành trong bản cập nhật báo cáo.

## **12.7 Minh chứng và cách tái lập**

| Nguồn | Nội dung đối chiếu |
| ----- | ----- |
| gemini-model-check.json | Model, response ID, token và thời gian của MODEL\_OK |
| gemini-dataset-run.json | Pilot 19 xử lý \+ 1 tái sử dụng; model và cấu hình hai pha |
| report-audit-20261024.json | Hash code, snapshot nghiệp vụ, lịch sử checkpoint, câu hỏi và thống kê vector |
| app.db và checkpoints.db | Các search ID, approved\_cv\_ids, errors và số câu hỏi |
| Chroma | 20 vector 768 chiều, định dạng compact-v2 |
| Ba ảnh người dùng | Lượt dev 3 năm, 10 CV và các nhóm câu hỏi |
| Hai ảnh Top 1 mới | Điểm dừng trước duyệt và kết quả 1 CV/6 câu hỏi |

Các JSON và ảnh audit nằm trong data/benchmarks và data/benchmarks/proof/report-audit-20261024. Hash SHA-256 của các file code liên quan được lưu trong JSON audit; không dùng HEAD cũ để đại diện cho mọi thay đổi chưa commit.

Có thể mở lịch sử theo \#search=a91025e0f20f489db58dda8d4a739efe hoặc \#search=c3adbe3a46d348a1be2e0577737d9b4f trên cùng web server. Mở lại lượt đã completed không cần chạy lại extraction hoặc sinh câu hỏi. Muốn thực nghiệm lượt mới cần tạo search ID mới và ghi riêng thời gian, số người duyệt và output.

## **12.8 Kiểm thử mã nguồn và kết luận thực nghiệm**

Terminal người dùng đã chạy toàn bộ suite với 107 passed và 12 subtests passed trong 7,46 giây. Đây là kết quả kiểm tra logic, dữ liệu, graph và API; không dùng số test làm độ chính xác của Gemini. Những ca kiểm thử thay thế lời gọi provider chỉ phục vụ kiểm tra lỗi/nhánh, không được đưa vào bảng output model thật.

Hai lượt giao diện mới xác nhận được chuỗi parse, retrieval, rank, explain, interrupt, duyệt và sinh câu hỏi với dữ liệu lưu nhất quán. Pilot xác nhận extraction/index trên nhóm 20 hồ sơ, gồm 19 xử lý và 1 tái sử dụng. Chưa có đánh giá Precision@k, Recall@k, nDCG, độ chính xác kinh nghiệm, tải đồng thời thực tế hoặc tổng chi phí provider. Không khẳng định hệ thống đạt chất lượng tuyển dụng chỉ từ một demo hoàn tất.


## **12.9 Kiểm chứng chạy thật bổ sung buổi tối 24/10/2026**

Đợt mới chạy toàn bộ suite: **115 passed, 15 subtests passed trong 13,53 giây**; thêm **4 test frontend passed**. Đã bổ sung dependency SOCKS cho SDK và đồng bộ môi trường theo uv.lock trước bộ kiểm thử cuối.

Pilot mới chọn 20 CV chưa có trong database: **20 xử lý, 20 ready, 0 failed, 0 already_ready**, thời gian **121,124 giây**. Cả 20 extraction dùng Gemini; log ghi một batch embedding HTTP 200 sau khi toàn bộ cache được lưu. Nạp lại cùng tập: **20 already_ready**, 0 xử lý mới, 0,609 giây. Chroma sau lượt này có **40 vector 768 chiều**, toàn bộ compact-v2, hữu hạn và norm xấp xỉ 1; 60 hàng failed lịch sử vẫn được giữ.

Search ID mới **13dbc9cc7fad4f29a96bdf686fd3d9e5** tìm Top 1 với tối thiểu 3 năm và Akka/Netty/Kafka Streams. Ứng viên **3f56345b** có bằng chứng nguồn “5+ years of commercial experience”, điểm **0,977572**. Trước duyệt: pending/human_review, 0 câu hỏi. Sau duyệt một người: completed, **6 câu hỏi**, mỗi nhóm có 2, next và errors đều rỗng. Đây là lượt API mới, không phải lượt trong ảnh buổi sáng.

Qwen thật cũng đã extract riêng một CV từ tập pilot, JSON qua schema, num_ctx=4096 và keep_alive=30m. Đây là kiểm tra local trực tiếp, không phải failover phát sinh tự nhiên; pilot không gặp 429. Chưa có ảnh UI mới do Edge không khởi chạy và tải Chromium bị allowlist chặn. Không dùng ảnh cũ hay phản hồi model giả để đại diện cho lượt mới.

Chi tiết và chỉ mục bằng chứng: [Kiểm chứng chạy thật 24/10/2026](KIEM_CHUNG_CHAY_THAT_2026-10-24.md). Dữ liệu mới nằm trong `data/benchmarks/proof/live-validation-20261024/`, kèm summary JSON và manifest SHA-256. Số liệu ở các mục 12.1–12.8 giữ ngữ cảnh snapshot buổi sáng.

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
