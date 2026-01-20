# Smart Retail System - Hướng dẫn cài đặt và sử dụng

Hướng dẫn đầy đủ để cài đặt và sử dụng hệ thống Smart Retail bao gồm camera tracking, MQTT và dashboard.

## Yêu cầu hệ thống

### Phần cứng
- **Camera**: USB Webcam hoặc IP Camera
- **ESP32**: Với cảm biến HX711 và load cell
- **Máy tính**: Để chạy hệ thống Python và web server
- **Màn hình TV**: Để hiển thị dashboard (tùy chọn)
- **Mạng**: WiFi cho ESP32 và kết nối mạng cho máy tính

### Phần mềm
- **Python 3.8+**
- **Thư viện Python**: Xem requirements.txt
- **Trình duyệt web**: Chrome, Firefox, Safari
- **Thiết bị di động**: Để quét QR (Android/iOS)

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd GrabNGo-Advanced
```

### 2. Cài đặt Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Nạp code cho ESP32
Xem `DASHBOARD_README.md` để biết cách nạp code cho ESP32.

### 4. Kiểm tra camera
```bash
python test_camera.py
```
Nếu camera hoạt động, bạn sẽ thấy thông báo "Camera test successful!"

### 5. Kiểm tra MQTT
```bash
python test_mqtt_quick.py
```
Nếu ESP32 đang gửi tin nhắn, bạn sẽ thấy các thông báo "CHANGE:..." trong 10 giây.

## Sử dụng hệ thống

### Khởi động hệ thống hoàn chỉnh
**Phương pháp 1: Dùng script khởi động**
```bash
# Trên Windows
start_dashboard.bat

# Trên Linux/Mac
python run_dashboard.py
```

**Phương pháp 2: Khởi động từng phần**
```bash
# Terminal 1: Chạy tracker với camera
python main.py

# Terminal 2: Khởi động dashboard (trình duyệt)
# Mở http://localhost:8080/dashboard
```

### Sử dụng dashboard
1. **Mở dashboard**: Truy cập http://localhost:8080/dashboard trên trình duyệt
2. **Quét QR**: Sử dụng điện thoại để quét QR tại http://localhost:8080
3. **Theo dõi khách hàng**: Dashboard tự động cập nhật thông tin khách hàng và giỏ hàng

### Quy trình hoạt động
1. **Khách hàng vào cửa hàng**: Camera phát hiện và hiển thị trong danh sách "Pending"
2. **Khách hàng quét QR**: Trạng thái chuyển thành "Confirmed"
3. **Khách hàng lấy sản phẩm**: ESP32 gửi MQTT event, hệ thống tự động cập nhật giỏ hàng
4. **Dashboard hiển thị**: Thông tin khách hàng, vị trí và giỏ hàng cập nhật real-time

## Kiểm tra và gỡ lỗi

### Test từng phần
```bash
# Test camera
python test_camera.py

# Test MQTT
python test_mqtt_quick.py

# Test tracker cơ bản
python test_tracker_simple.py

# Test flow với camera
python test_flow_camera.py

# Test MQTT integration
python test_mqtt_integration.py

# Test dashboard
python test_dashboard.py
```

### Test hoàn chỉnh
```bash
# Test toàn bộ flow với tin nhắn giả lập
python test_complete_flow.py
```

## Vấn đề thường gặp và giải pháp

### Camera không hoạt động
1. Kiểm tra xem camera có được kết nối đúng không
2. Kiểm tra xem có ứng dụng nào khác đang sử dụng camera không
3. Thử thay đổi index camera trong code (thay 0 bằng 1 hoặc 2)

### ESP32 không gửi tin nhắn
1. Kiểm tra kết nối WiFi của ESP32
2. Kiểm tra xem ESP32 có được cấp nguồn không
3. Đặt vật > 50g lên cảm biến để kích hoạt tin nhắn
4. Xem log trên ESP32 (nếu có kết nối Serial)

### Dashboard không hiển thị dữ liệu
1. Kiểm tra xem web server đã khởi động chưa
2. Truy cập http://localhost:8080/dashboard/data để kiểm tra API
3. Kiểm tra console log của hệ thống

### Không phát hiện được khách hàng
1. Điều chỉnh camera để đảm bảo ánh sáng tốt
2. Kiểm tra xem có vật cản trở không
3. Tăng độ phân giải camera nếu cần

### Không xác nhận được khách hàng
1. Kiểm tra xem camera phát hiện được keypoints không
2. Đảm bảo khách hàng đứng đủ gần camera
3. Kiểm tra góc camera để đảm bảo toàn thân người

## Tùy chỉnh hệ thống

### Điều chỉnh ngưỡng phát hiện
Trong `src/main_tracker.py`:
- Ngưỡng phát hiện người: `conf` (mặc định 0.5)
- Ngưỡng IoU: `iou` (mặc định 0.7)
- Ngưỡng xác nhận: `self.min_samples_required` và `self.min_confidence_avg`

### Điều kích thước zone
Trong `src/main_tracker.py`:
- QR Zone: `self.qr_zone_percent`
- Shelf Zone: `self.shelf_zone_percent`

### Tùy chỉnh MQTT
Trong ESP32 `main.py`:
- WiFi SSID và password
- MQTT broker và topic
- Ngưỡng trọng lượng: `WEIGHT_CHANGE_THRESHOLD`

## Tích hợp nâng cao

### Kết nối với hệ thống POS
1. Thêm API endpoint trong `web_server.py` để gửi dữ liệu giỏ hàng
2. Tích hợp với API của hệ thống POS hiện có

### Lưu trữ dữ liệu
1. Thêm database (SQLite, PostgreSQL)
2. Sửa đổi code để lưu trữ thông tin khách hàng và giỏ hàng

### Thêm tính năng khác
1. Biểu đồ thống kê doanh thu
2. Phân tích hành vi khách hàng
3. Hệ thống cảnh báo và thông báo
4. Tích hợp với CRM

## Bảo trì

### Hàng ngày
1. Kiểm tra log hệ thống
2. Xóa dữ liệu cũ (nếu cần)
3. Kiểm tra hoạt động của camera và ESP32

### Hàng tuần
1. Cập nhật hệ thống nếu có phiên bản mới
2. Kiểm tra hiệu suất của hệ thống
3. Tối ưu hóa nếu cần

### Hàng tháng
1. Sao lưu dữ liệu quan trọng
2. Kiểm tra và bảo trì phần cứng
3. Đào tạo nhân viên nếu cần