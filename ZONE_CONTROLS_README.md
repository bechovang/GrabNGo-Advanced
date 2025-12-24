# Zone Controls - Hướng dẫn sử dụng

## Tổng quan
Zone controls cho phép tùy chỉnh vị trí và kích thước của QR Zone và Shelf Zone trực tiếp trên dashboard, giúp hệ thống hoạt động chính xác với layout cửa hàng của bạn.

## Cách sử dụng

### 1. Khởi động hệ thống
```bash
# Chạy hệ thống với zone controls
test_zone_controls.bat

# Hoặc
python run_full_system.py
```

### 2. Mở dashboard
- Truy cập: http://localhost:8080/dashboard (hoặc http://[IP]:8080/dashboard)
- Nhấp vào nút "Zones" ở góc trên bên phải

### 3. Chỉnh thông số zone

Trong cửa sổ Zone Settings, bạn sẽ thấy 2 tab:

#### Tab QR Zone:
- **X Position (%)**: Vị trí bắt đầu theo chiều ngang (0-100%)
- **Y Position (%)**: Vị trí bắt đầu theo chiều dọc (0-100%)
- **Width (%)**: Chiều rộng zone (10-100%)
- **Height (%)**: Chiều cao zone (10-100%)

#### Tab Shelf Zone:
- **X Position (%)**: Vị trí bắt đầu theo chiều ngang (0-100%)
- **Y Position (%)**: Vị trí bắt đầu theo chiều dọc (0-100%)
- **Width (%)**: Chiều rộng zone (10-100%)
- **Height (%)**: Chiều cao zone (10-100%)

### 4. Lưu thiết lập
- Nhấp vào nút "Save Settings" để lưu thay đổi
- Hệ thống sẽ tự động áp dụng thiết lập mới
- Các thiết lập được lưu cho lần chạy tiếp theo

## Cách xác định thông số

### 1. Mở dashboard với camera đang chạy
- Đứng trước camera để xem hình ảnh
- Quan sát vị trí thực tế của các zone

### 2. Ước lượng zone trên dashboard
- Đưa chuột vào các zone để xem vị trí
- Xác định xem zone có đúng vị trí mong muốn không

### 3. Chỉnh thông số
- Dùng thanh trượt để điều chỉnh các thông số
- Quan sát thay đổi trên bản đồ khi điều chỉnh

### 4. Kiểm tra với người thật
- Đứng vào QR zone và xem hệ thống có nhận ra không
- Đứng vào Shelf zone và xem hệ thống có nhận ra không

### 5. Tinh chỉnh
- Lặp lại các bước cho đến khi zone chính xác
- Lưu thiết lập cuối cùng

## Các giá trị mặc định và gợi ý

### QR Zone (Quét QR)
- **Vị trí mặc định**: Bên phải màn hình (X=70%, Y=0%)
- **Kích thước mặc định**: Rộng 30%, cao 100%
- **Mục đích**: Vùng quét mã QR xác nhận khách hàng

### Shelf Zone (Kệ hàng)
- **Vị trí mặc định**: Bên trái-trung màn hình (X=5%, Y=30%)
- **Kích thước mặc định**: Rộng 45%, cao 60%
- **Mục đích**: Vùng kệ hàng, nơi khách hàng có thể lấy sản phẩm

## Tùy chỉnh nâng cao

### Định nghĩa vị trí camera
- X=0%: Bên trái cùng
- X=100%: Bên phải cùng
- Y=0%: Cùng trên cùng
- Y=100%: Cùng dưới cùng

### Tính toán zone
- X1, Y1: Vị trí góc trên bên trái
- X2, Y2: Vị trí góc dưới bên phải
- Width = X2 - X1
- Height = Y2 - Y1

### Ví dụ thực tế
Nếu cửa hàng của bạn có:
- Kệ hàng ở bên trái camera
- Vùng quét QR ở bên phải camera

Bạn có thể đặt:
- QR Zone: X1=65%, Y1=10%, Width=30%, Height=80%
- Shelf Zone: X1=5%, Y1=20%, Width=40%, Height=70%

## Lưu ý quan trọng

1. **Tỷ lệ phần trăm**: Các giá trị được tính theo % kích thước camera, không phải pixel
2. **Tự động áp dụng**: Thay đổi được áp dụng ngay cả khi hệ thống đang chạy
3. **Lưu lại**: Thiết lập được lưu và áp dụng khi khởi động lại hệ thống
4. **Không chồng lấn**: QR Zone và Shelf Zone có thể chồng lên nhau nếu cần
5. **Camera cố định**: Nếu camera bị di chuyển, cần điều chỉnh lại zone

## Kiểm tra và gỡ lỗi

### Zone không chính xác
1. Kiểm tra lại các thông số đã nhập
2. Đảm bảo camera không bị nghiêng
3. Xác định hướng của camera (gương hay không)

### Thiết lập không được lưu
1. Kiểm tra kết nối mạng
2. Xem console log để biết có lỗi gì
3. Thử khởi động lại hệ thống

### Zone không áp dụng
1. Kiểm tra xem có nhấn Save Settings không
2. Xem console log để biết có lỗi gì
3. Thử refresh lại dashboard

## Tích hợp với hệ thống khác

### Camera nhiều góc
- Có thể tạo nhiều preset cho từng camera
- Chuyển đổi giữa các preset tùy theo camera đang dùng

### Tự động phát hiện zone
- Có thể tích hợp AI để tự động phát hiện vị trí kệ
- Giảm thiểu cần tùy chỉnh thủ công