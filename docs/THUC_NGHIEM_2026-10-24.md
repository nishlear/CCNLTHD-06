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
