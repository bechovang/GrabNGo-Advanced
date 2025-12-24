# 📱 Hệ Thống Xác Nhận Khách Hàng Qua QR Code

## 🎯 Tổng Quan Nhanh

Hệ thống này cho phép xác nhận khách hàng bằng cách quét QR code thay vì nhấn phím. Khách hàng đứng ở **góc trái dưới** màn hình, nhân viên quét QR code của khách trên điện thoại.

---

## 🚀 Quick Start (3 Bước)

### **Bước 1: Tạo QR Code Cho Khách Hàng**

```bash
# Cài đặt thư viện (nếu chưa có)
pip install qrcode[pil]

# Tạo QR code đơn giản
python create_qr_code.py --customer-id CUST_001

# Hoặc tạo nhiều QR codes cùng lúc
python create_qr_code.py --batch 10 --start-id 1

# Hoặc chế độ tương tác
python create_qr_code.py
```

QR codes sẽ được lưu trong thư mục `qr_codes/`. In ra hoặc gửi cho khách hàng.

### **Bước 2: Chạy CV System**

```bash
python main.py
```

Bạn sẽ thấy:
```
🌐 Starting Web Server...
   URL: http://0.0.0.0:8080
   Mobile App: http://0.0.0.0:8080/
   (Access from mobile: http://<your-ip>:8080/)
```

**Lưu ý:** Ghi nhớ IP address hiển thị (ví dụ: `192.168.1.100`).

### **Bước 3: Mở Web Trên Điện Thoại**

1. **Tìm IP của máy tính:**
   - Windows: Mở cmd, gõ `ipconfig`, tìm **IPv4 Address**
   - Linux/Mac: Mở terminal, gõ `ifconfig`, tìm IP address

2. **Mở trình duyệt trên điện thoại:**
   - Gõ: `http://192.168.1.100:8080` (thay bằng IP của bạn)
   - Cho phép truy cập camera khi được hỏi

3. **Sẵn sàng sử dụng!**

---

## 📖 Hướng Dẫn Chi Tiết

Xem file: [`docs/QR_CONFIRMATION_GUIDE.md`](docs/QR_CONFIRMATION_GUIDE.md)

Bao gồm:
- ✅ Cách tạo QR code (nhiều phương pháp)
- ✅ Cách mở web trên điện thoại (chi tiết)
- ✅ Flow sử dụng từng bước
- ✅ Cấu hình & tùy chỉnh
- ✅ Troubleshooting

---

## 🎬 Flow Sử Dụng

```
1. Khách vào cửa hàng
   → CV detect → Tạo PENDING_0001

2. Khách đứng ở góc trái dưới
   → Zone chuyển XANH
   → Mobile web app bật camera

3. Nhân viên quét QR code
   → Lấy customer_id: "CUST_001"
   → Auto-match với PENDING_0001
   → Confirm: PENDING_0001 → CUST_001
```

---

## 📁 Files Quan Trọng

- `create_qr_code.py` - Script tạo QR code cho khách hàng
- `web_server.py` - Web server cho mobile app
- `mobile_qr_scanner.html` - Giao diện quét QR trên điện thoại
- `src/main_tracker.py` - CV tracking system với QR zone logic
- `docs/QR_CONFIRMATION_GUIDE.md` - Hướng dẫn chi tiết

---

## ❓ FAQ

**Q: Lấy QR code cho khách ở đâu?**  
A: Dùng script `create_qr_code.py` hoặc website/app tạo QR code. Xem chi tiết trong guide.

**Q: Mở web trên điện thoại như nào?**  
A: Tìm IP máy tính (`ipconfig`), mở trình duyệt, gõ `http://<IP>:8080`. Xem chi tiết trong guide.

**Q: Zone ở đâu?**  
A: Góc trái dưới màn hình (0-30% width, 80-100% height). Có thể tùy chỉnh trong code.

**Q: Nhiều người trong zone thì sao?**  
A: Hệ thống sẽ từ chối, yêu cầu chỉ 1 người trong zone.

**Q: QR code format như nào?**  
A: JSON `{"customer_id": "CUST_001"}` hoặc text đơn giản `CUST_001`.

---

## 🔧 Troubleshooting

Xem phần **Troubleshooting** trong [`docs/QR_CONFIRMATION_GUIDE.md`](docs/QR_CONFIRMATION_GUIDE.md)

---

**Chúc bạn sử dụng thành công! 🎉**




