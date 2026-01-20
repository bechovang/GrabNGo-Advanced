# Smart Retail Dashboard

Dashboard theo dõi thời gian thực cho hệ thống Smart Retail, hiển thị thông tin khách hàng và giỏ hàng trực tiếp trên TV tại cửa hàng.

## Tính năng

### 1. Tổng quan cửa hàng
- **Tổng khách hàng**: Số lượng khách hàng đang trong cửa hàng
- **Khách hàng hoạt động**: Số lượng khách hàng đã xác nhận (confirmed)
- **Sản phẩm đã lấy**: Tổng số sản phẩm đã được khách hàng lấy
- **Thời gian trung bình**: Thời gian trung bình khách hàng ở trong cửa hàng

### 2. Danh sách khách hàng
- **Khách hàng đang chờ xác nhận**: Hiển thị khách hàng chưa quét QR (màu vàng)
- **Khách hàng đã xác nhận**: Hiển thị khách hàng đã quét QR (màu xanh)
- **Thời gian ở cửa hàng**: Thời gian từ khi khách hàng vào đến hiện tại
- **Chọn khách hàng**: Click để xem thông tin chi tiết và giỏ hàng

### 3. Bản đồ cửa hàng
- **QR Zone**: Vùng quét mã QR (bên phải)
- **Shelf Zone**: Vùng kệ hàng (bên trái)
- **Vị trí khách hàng**: Chấm tròn màu đại diện cho từng khách hàng
  - Màu xanh: Trong QR Zone
  - Màu xanh lá: Trong Shelf Zone
  - Màu xanh dương: Không trong zone nào

### 4. Giỏ hàng (Shopping Cart)
- **Thông tin khách hàng**: ID, trạng thái, thời gian ở cửa hàng
- **Danh sách sản phẩm**: Sản phẩm khách hàng đã lấy
  - Trọng lượng (gram)
  - Thời gian lấy
  - Zone/kệ lấy hàng
  - Độ tin cậy
- **Tổng kết**: Tổng số sản phẩm và tổng trọng lượng

## Cài đặt và sử dụng

### 1. Khởi động hệ thống
```bash
python run_dashboard.py
```
Hệ thống sẽ khởi động tracker, MQTT, web server và camera (nếu có).

### 2. Truy cập dashboard
Mở trình duyệt và truy cập:
- Dashboard: http://localhost:8080/dashboard
- Mobile QR Scanner: http://localhost:8080

### 3. Sử dụng dashboard
1. **Theo dõi khách hàng**: Dashboard tự động cập nhật khi có khách hàng vào cửa hàng
2. **Xác nhận khách hàng**: Sử dụng điện thoại để quét QR tại zone xác nhận
3. **Theo dõi giỏ hàng**: Click vào khách hàng để xem sản phẩm đã lấy
4. **Xem bản đồ**: Theo dõi vị trí khách hàng trong cửa hàng

## Kiểm tra và gỡ lỗi

### Test dashboard API
```bash
python test_dashboard.py
```

### Test với dữ liệu giả
Dashboard có thể hoạt động độc lập với dữ liệu giả để kiểm tra giao diện.

### Xem log hệ thống
Log được hiển thị trong console khi chạy `run_dashboard.py`, bao gồm:
- Thông tin tracker
- Thông tin MQTT
- Thống kê định kỳ

## Tùy chỉnh

### Điều chỉnh kích thước zone
Trong `src/main_tracker.py`:
- QR Zone: `self.qr_zone_percent`
- Shelf Zone: `self.shelf_zone_percent`

### Tùy chỉnh giao diện
File `dashboard.html` có thể chỉnh sửa để thay đổi:
- Màu sắc (CSS variables)
- Kích thước font
- Layout columns
- Thông tin hiển thị

### Tùy chỉnh polling rate
Trong `dashboard.html`, thay đổi giá trị trong `setInterval(pollData, 3000)` (hiện tại là 3 giây).

## Tích hợp với hệ thống khác

### Kết nối với hệ thống POS
Thêm API endpoint trong `web_server.py` để gửi dữ liệu giỏ hàng đến hệ thống POS.

### Lưu trữ dữ liệu
Thêm database (SQLite, PostgreSQL) để lưu trữ lịch sử mua hàng và thống kê dài hạn.

### Tích hợp với hệ thống tồn kho
Gửi thông tin sản phẩm được lấy đến hệ thống quản lý tồn kho.

## Gỡ lỗi phổ biến

### Dashboard không hiển thị dữ liệu
1. Kiểm tra xem web server đã khởi động chưa (xem console)
2. Truy cập http://localhost:8080/dashboard/data để kiểm tra API
3. Kiểm tra xem tracker đã khởi động thành công chưa

### Vị trí khách hàng không chính xác
1. Điều chỉnh tọa độ zone trong dashboard.html
2. Cài đặt lại camera để đảm bảo góc phù hợp

### Không nhận được MQTT events
1. Kiểm tra ESP32 có đang hoạt động không
2. Kiểm tra kết nối WiFi của ESP32
3. Xem log MQTT trong console tracker

## Tính năng nâng cao (đề xuất)

1. **Lịch sử mua hàng**: Lưu và hiển thị lịch sử mua hàng của khách hàng
2. **Biểu đồ thống kê**: Thêm biểu đồ thể hiện sản phẩm bán chạy, giờ cao điểm
3. **Cảnh báo**: Thông báo khi khách hàng ở trong zone quá lâu
4. **Tùy chỉnh giao diện**: Cho phép thay đổi màu sắc, layout theo yêu cầu
5. **Đa cửa hàng**: Hỗ trợ theo dõi nhiều cửa hàng trong cùng một dashboard