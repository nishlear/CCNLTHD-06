# Lab State reducer và checkpoint

Phạm vi: bài thực hành xác định, dùng LangGraph và SQLite thật, không gọi LLM. Output lab không được tính là phản hồi model thật.

## Bài tập

Hoàn thành `collect` trong starter.py, rồi thay implementation `collect` trong solution.py bằng lời giải của mình. Mỗi nhánh phải trả một phần cập nhật `collected` là danh sách một phần tử; `Annotated[list[str], operator.add]` hợp nhất các cập nhật. Không sửa state chung tại chỗ.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m labs.state_checkpoint.solution --db data/lab/checkpoints.db --phase start
.\.venv\Scripts\python.exe -m labs.state_checkpoint.solution --db data/lab/checkpoints.db --phase resume
.\.venv\Scripts\python.exe -m labs.state_checkpoint.solution --db data/lab/checkpoints.db --phase resume
```

Lần start: collected=[A,B], next=[review], output rỗng. Lần resume từ process mới: cùng thread_id, approved=true, output=[A,B], next rỗng. Lần resume tiếp theo trả lại trạng thái đã hoàn tất, không nhân đôi danh sách. Thread khác không tìm thấy checkpoint và bị từ chối. Đổi đường dẫn database hoặc thread ID khi bắt đầu bài mới.

Barrier giữa các superstep bảo đảm hai nhánh collect xong trước khi review đọc danh sách. Đoạn trước interrupt có thể chạy lại nên không ghi kết quả nghiệp vụ tại đó. Resume dùng Command với đúng thread_id. Đây là kiểm chứng tiến trình mới đọc/resume checkpoint, không phải thử khởi động lại web server.

Kiểm tra tự động: tests/test_lab_checkpoint.py. Output và database mẫu của lần chạy xác nhận nằm trong gói snapshot bàn giao.
