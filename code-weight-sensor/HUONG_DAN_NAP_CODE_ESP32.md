# Hướng dẫn chi tiết nạp code vào ESP32 bằng mpremote

## Bước 1: Kiểm tra mpremote và thiết bị

### Kiểm tra mpremote đã được cài đặt chưa

```bash
mpremote --version
```

### Liệt kê các cổng COM có thiết bị MicroPython

```bash
mpremote devs
```

## Bước 2: Kết nối và nạp code

### Nếu ESP32 đang chạy code, cần dừng trước:

```bash
# Dừng code đang chạy
mpremote connect COM5 exec "import sys; sys.exit()"
```

### Nạp code chính (main.py):

```bash
# Nạp file main.py vào ESP32
mpremote connect COM5 cp "weight_sensor_esp32/main.py" ":main.py"
```

### Nạp thư viện hỗ trợ (hx711.py):

```bash
# Nạp file hx711.py vào ESP32
mpremote connect COM5 cp "weight_sensor_esp32/hx711.py" ":hx711.py"
```

## Bước 3: Khởi động lại ESP32 để chạy code mới

```bash
# Reset ESP32 để chạy code mới
mpremote connect COM5 exec "import machine; machine.reset()"
```

## Bước 4: Kiểm tra hoạt động

### Xem danh sách file trên ESP32

```bash
mpremote connect COM5 exec "import os; print(os.listdir())"
```

### Xem log output của ESP32

```bash
mpremote connect COM5 repl
```

### Hoặc chạy code trực tiếp để xem log

```bash
mpremote connect COM5 run "main.py"
```

## Lưu ý quan trọng:

1. **Chọn đúng cổng COM**: Trong ví dụ này là COM5, có thể thay đổi tùy máy tính của bạn

2. **Dừng code trước khi nạp**: Nếu ESP32 đang chạy, cần dừng trước khi nạp code mới

3. **Đảm bảo có thư viện hx711.py**: Cần nạp cả thư viện này để main.py hoạt động

4. **Kiểm tra lại cấu hình**: Có thể cần thay đổi thông tin WiFi, MQTT trong code

## Cách khắc phục lỗi thường gặp

### 1. Cách nhanh nhất: Rút ra cắm lại

Đây là cách hiệu quả 90% các trường hợp.

- Rút cáp USB của mạch ESP32 ra khỏi máy tính
- Đợi khoảng 3-5 giây
- Cắm lại
- Chạy lại lệnh: `mpremote connect COM5 cp "weight_sensor_esp32/main.py" ":main.py"`

### 2. Lỗi UnicodeDecodeError khi dùng repl

Nếu gặp lỗi `UnicodeDecodeError` khi chạy `mpremote repl`, đảm bảo code không có emoji hoặc ký tự đặc biệt. File `main.py` đã được cập nhật để loại bỏ emoji.

### 3. ESP32 không phản hồi

- Kiểm tra cáp USB có kết nối tốt không
- Thử đổi cổng USB khác
- Kiểm tra driver USB-to-Serial đã cài đặt đúng chưa
- Thử reset ESP32 bằng nút RESET trên board

### 4. Không tìm thấy cổng COM

- Kiểm tra Device Manager (Windows) để xem cổng COM nào đang hoạt động
- Đảm bảo ESP32 đã được cấp nguồn
- Thử cài lại driver CP2102 hoặc CH340 (tùy loại ESP32 board)

## Quy trình nạp code nhanh (Quick Start)

```bash
# 1. Kiểm tra thiết bị
mpremote devs

# 2. Nạp code (thay COM5 bằng cổng của bạn)
mpremote connect COM5 cp "weight_sensor_esp32/main.py" ":main.py"
mpremote connect COM5 cp "weight_sensor_esp32/hx711.py" ":hx711.py"

# 3. Reset và xem log
mpremote connect COM5 exec "import machine; machine.reset()"
mpremote connect COM5 repl
```

