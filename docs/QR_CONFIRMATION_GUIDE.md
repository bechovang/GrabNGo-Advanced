# 📱 Hướng Dẫn Sử Dụng Hệ Thống Xác Nhận QR Code

## Tổng Quan

Hệ thống này cho phép xác nhận khách hàng bằng cách quét QR code thay vì nhấn phím trên bàn phím. Khách hàng sẽ có QR code chứa `customer_id`, và nhân viên sẽ quét QR code đó khi khách đứng ở khu vực quét QR (góc trái dưới màn hình).

---

## 🎯 Flow Tổng Thể

```
1. Tạo QR Code cho Khách Hàng
   ↓
2. Khách Hàng Vào Cửa Hàng
   ↓
3. CV System Detect → Tạo PENDING_XXXX
   ↓
4. Khách Đứng Ở Khu Vực Quét QR (Góc Trái Dưới)
   ↓
5. Zone Chuyển Xanh → Mobile Web App Tự Động Bật Scanner
   ↓
6. Nhân Viên Quét QR Code Của Khách
   ↓
7. Hệ Thống Auto-Match & Confirm → PENDING_XXXX → CUST_XXXX
```

---

## 📋 Bước 1: Tạo QR Code Cho Khách Hàng

### **Vấn Đề: Hệ Thống Hiện Tại Chưa Có Phần Tạo QR Code**

Hiện tại, hệ thống CV tracking **chưa có** phần tạo QR code cho khách hàng. Bạn cần tạo QR code **trước** khi khách vào cửa hàng.

### **Các Cách Tạo QR Code:**

#### **Option A: Tạo QR Code Thủ Công (Đơn Giản)**

1. **Tạo Customer ID:**
   - Format: `CUST_001`, `CUST_002`, `CUST_003`, ...
   - Hoặc bất kỳ format nào bạn muốn (ví dụ: `KH_001`, `CUSTOMER_001`)

2. **Tạo QR Code:**
   - Sử dụng website: https://www.qr-code-generator.com/
   - Hoặc app trên điện thoại: "QR Code Generator"
   - **Nội dung QR Code:**
     ```json
     {
       "customer_id": "CUST_001",
       "name": "Nguyễn Văn A",
       "phone": "0123456789"
     }
     ```
   - Hoặc đơn giản chỉ cần text: `CUST_001`

3. **Lưu QR Code:**
   - In ra giấy hoặc lưu trên điện thoại khách hàng
   - Khách hàng giữ QR code này để sử dụng khi mua hàng

#### **Option B: Tạo Hệ Thống Đăng Ký Khách Hàng (Nâng Cao)**

Tạo một web app riêng để:
- Khách hàng đăng ký tài khoản
- Hệ thống tự động tạo `customer_id`
- Tự động generate QR code
- Lưu vào database

**Ví dụ code Python để tạo QR code:**

```python
import qrcode
import json

def create_customer_qr(customer_id, name, phone):
    """Tạo QR code cho khách hàng."""
    # Tạo dữ liệu JSON
    qr_data = {
        "customer_id": customer_id,
        "name": name,
        "phone": phone
    }
    
    # Tạo QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(json.dumps(qr_data))
    qr.make(fit=True)
    
    # Tạo image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Lưu file
    filename = f"qr_codes/{customer_id}.png"
    img.save(filename)
    print(f"✅ Created QR code: {filename}")
    return filename

# Ví dụ sử dụng
create_customer_qr("CUST_001", "Nguyễn Văn A", "0123456789")
```

**Cài đặt thư viện:**
```bash
pip install qrcode[pil]
```

---

## 📱 Bước 2: Mở Web Trên Điện Thoại

### **Yêu Cầu:**
- Điện thoại và máy tính chạy CV system phải **cùng mạng WiFi** (hoặc cùng mạng LAN)
- Hoặc máy tính phải có IP công cộng (nếu điện thoại dùng 4G/5G)

### **Cách 1: Tìm IP Của Máy Tính**

#### **Trên Windows:**
1. Mở Command Prompt (cmd)
2. Gõ lệnh: `ipconfig`
3. Tìm dòng **IPv4 Address**, ví dụ: `192.168.1.100`

#### **Trên Linux/Mac:**
1. Mở Terminal
2. Gõ lệnh: `ifconfig` hoặc `ip addr`
3. Tìm IP address (thường là `192.168.x.x` hoặc `10.0.x.x`)

### **Cách 2: Mở Web Trên Điện Thoại**

1. **Đảm bảo CV system đang chạy:**
   ```bash
   python main.py
   ```
   Bạn sẽ thấy dòng:
   ```
   🌐 Starting Web Server...
      URL: http://0.0.0.0:8080
      Mobile App: http://0.0.0.0:8080/
      (Access from mobile: http://<your-ip>:8080/)
   ```

2. **Mở trình duyệt trên điện thoại:**
   - Chrome, Safari, Firefox đều được
   - Gõ địa chỉ: `http://192.168.1.100:8080` (thay bằng IP của máy tính)
   - Hoặc nếu test trên cùng máy: `http://localhost:8080`

3. **Cho phép truy cập camera:**
   - Trình duyệt sẽ hỏi quyền truy cập camera
   - Nhấn **"Cho phép"** hoặc **"Allow"**

4. **Giao diện sẽ hiển thị:**
   ```
   📱 Customer QR Scanner
   
   ⏸️ Waiting for customer in QR zone...
   
   [Camera View - sẽ bật khi zone active]
   
   Instructions:
   1. Wait for customer to stand in QR zone
   2. When zone turns green, camera will activate
   3. Point camera at customer's QR code
   4. QR code will be scanned automatically
   ```

---

## 🎬 Bước 3: Sử Dụng Hệ Thống

### **Flow Chi Tiết:**

#### **3.1. Khách Hàng Vào Cửa Hàng**

1. CV system tự động detect người
2. Tạo `PENDING_XXXX` (ví dụ: `PENDING_0001`)
3. Hiển thị trên màn hình CV:
   - Box màu cam (đang collect data)
   - Label: `PENDING_0001`
   - Progress bar: `Samples: 2/5`

#### **3.2. Khách Đứng Ở Khu Vực Quét QR**

1. **Khu vực quét QR:**
   - Vị trí: **Góc trái dưới** màn hình
   - Kích thước: 30% chiều rộng, 20% chiều cao (từ dưới lên)
   - Hiển thị: Hình chữ nhật trên màn hình CV

2. **Khi khách đứng vào zone:**
   - Hình chữ nhật chuyển **XANH** (green)
   - Label: `QR ZONE ✅ ACTIVE - PENDING_0001`
   - Mobile web app tự động bật camera
   - Status trên điện thoại: `✅ Ready to scan`

3. **Nếu nhiều người trong zone:**
   - Hình chữ nhật chuyển **ĐỎ** (red)
   - Label: `QR ZONE ⚠️ MULTIPLE (2)`
   - Mobile web app: `⚠️ Multiple people in zone. Please scan one at a time`
   - Camera không bật

#### **3.3. Nhân Viên Quét QR Code**

1. **Khi zone xanh:**
   - Camera trên điện thoại tự động bật
   - Hiển thị khung quét QR (250x250 pixels)

2. **Quét QR code:**
   - Đưa điện thoại về phía QR code của khách
   - Giữ khoảng cách 20-30cm
   - Đảm bảo ánh sáng đủ
   - QR code sẽ được scan tự động

3. **Sau khi quét:**
   - Status: `✅ Scanned: CUST_001`
   - Gửi request lên server: `POST /confirm` với `customer_id: "CUST_001"`
   - Server auto-match với `PENDING_0001` (vì đang trong zone)
   - CV system confirm: `PENDING_0001` → `CUST_001`

4. **Kết quả:**
   - Màn hình CV: Box chuyển **XANH**, label: `CUST_001`
   - Điện thoại: `✅ Confirmed! Customer: CUST_001`
   - Zone reset, sẵn sàng cho khách tiếp theo

---

## 🔧 Cấu Hình & Tùy Chỉnh

### **Thay Đổi Vị Trí QR Zone**

Mở file `src/main_tracker.py`, tìm dòng:

```python
self.qr_zone_percent = {
    'x1_percent': 0.0,    # 0% from left (left edge)
    'y1_percent': 0.8,    # 80% from top (bottom area)
    'x2_percent': 0.3,    # 30% from left (width)
    'y2_percent': 1.0     # 100% from top (bottom edge)
}
```

**Ví dụ: Thay đổi sang góc phải dưới:**
```python
self.qr_zone_percent = {
    'x1_percent': 0.7,    # 70% from left
    'y1_percent': 0.8,    # 80% from top
    'x2_percent': 1.0,    # 100% from left (right edge)
    'y2_percent': 1.0     # 100% from top (bottom edge)
}
```

### **Thay Đổi Port Web Server**

Mở file `src/main_tracker.py`, tìm dòng:

```python
server_thread = threading.Thread(
    target=run_server,
    args=(tracker, '0.0.0.0', 8080, False),  # Port 8080
    daemon=True
)
```

Thay `8080` bằng port khác (ví dụ: `3000`, `5000`).

### **Thay Đổi Overlap Threshold**

Mở file `src/main_tracker.py`, tìm dòng:

```python
self.zone_overlap_threshold = 0.5  # 50% of person must be in zone
```

Thay `0.5` bằng giá trị khác (0.0 - 1.0):
- `0.3` = 30% (dễ hơn, chỉ cần một phần người trong zone)
- `0.7` = 70% (khó hơn, cần hầu hết người trong zone)

---

## ❓ Troubleshooting

### **1. Không Mở Được Web Trên Điện Thoại**

**Nguyên nhân:**
- IP không đúng
- Firewall chặn port 8080
- Không cùng mạng WiFi

**Giải pháp:**
- Kiểm tra IP: `ipconfig` (Windows) hoặc `ifconfig` (Linux/Mac)
- Tắt firewall tạm thời để test
- Đảm bảo điện thoại và máy tính cùng WiFi

### **2. Camera Không Bật**

**Nguyên nhân:**
- Chưa cho phép quyền camera
- Zone chưa active (chưa có người trong zone)

**Giải pháp:**
- Kiểm tra quyền camera trong trình duyệt
- Đảm bảo có 1 người đứng trong QR zone (zone phải xanh)

### **3. QR Code Không Quét Được**

**Nguyên nhân:**
- QR code bị mờ, hỏng
- Ánh sáng không đủ
- Khoảng cách quá xa/gần

**Giải pháp:**
- In QR code rõ ràng, kích thước đủ lớn (ít nhất 5x5cm)
- Đảm bảo ánh sáng đủ
- Giữ khoảng cách 20-30cm

### **4. Zone Không Chuyển Xanh**

**Nguyên nhân:**
- Người không đứng đúng vị trí
- Nhiều người trong zone (>1)
- Overlap threshold quá cao

**Giải pháp:**
- Kiểm tra vị trí zone trên màn hình CV
- Đảm bảo chỉ 1 người trong zone
- Giảm `zone_overlap_threshold` nếu cần

### **5. Confirmation Không Thành Công**

**Nguyên nhân:**
- PENDING track chưa validate (score < 80%)
- Không có PENDING trong zone
- Customer_id không đúng format

**Giải pháp:**
- Đợi PENDING track validate (box chuyển xanh)
- Kiểm tra log trên console để xem lỗi
- Đảm bảo QR code chứa đúng `customer_id`

---

## 📝 Ví Dụ QR Code Format

### **Format 1: JSON (Khuyến Nghị)**
```json
{
  "customer_id": "CUST_001",
  "name": "Nguyễn Văn A",
  "phone": "0123456789"
}
```

### **Format 2: Text Đơn Giản**
```
CUST_001
```

### **Format 3: URL (Nếu Có Hệ Thống Backend)**
```
https://your-system.com/customer/CUST_001
```

**Lưu ý:** Hệ thống hiện tại chỉ cần `customer_id`, các thông tin khác (name, phone) là optional.

---

## 🎯 Tóm Tắt Quick Start

1. **Tạo QR code cho khách:**
   - Dùng website/app tạo QR code
   - Nội dung: `{"customer_id": "CUST_001"}` hoặc `CUST_001`
   - In ra hoặc lưu trên điện thoại khách

2. **Chạy CV system:**
   ```bash
   python main.py
   ```

3. **Mở web trên điện thoại:**
   - Tìm IP máy tính: `ipconfig` (Windows)
   - Mở trình duyệt: `http://<IP>:8080`
   - Cho phép camera

4. **Sử dụng:**
   - Khách vào cửa hàng → CV tạo PENDING
   - Khách đứng góc trái dưới → Zone xanh
   - Quét QR code → Auto confirm!

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:
1. Console log của CV system
2. Browser console trên điện thoại (F12 → Console)
3. Network tab để xem API requests

---

**Chúc bạn sử dụng thành công! 🎉**




