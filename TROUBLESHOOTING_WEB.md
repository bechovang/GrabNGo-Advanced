# 🔧 Khắc Phục Lỗi: Không Vào Được Web Server

## ❓ Vấn Đề

Không thể truy cập web server từ điện thoại hoặc trình duyệt.

---

## 🔍 Bước 1: Kiểm Tra Web Server Có Chạy Không

### **1.1. Chạy CV System**

```bash
python main.py
```

**Kiểm tra output:**
- Bạn phải thấy dòng: `✅ Web server started in background thread`
- Nếu thấy: `⚠️ Warning: Could not start web server` → Xem phần **Lỗi Khi Start Server**

### **1.2. Test Server Bằng Script**

```bash
python test_web_server.py
```

Script này sẽ:
- ✅ Kiểm tra port 8080 có mở không
- ✅ Test HTTP connection
- ✅ Test API endpoint
- ✅ Hiển thị IP address để dùng trên điện thoại

---

## 🔍 Bước 2: Tìm IP Address Đúng

### **Windows:**

1. Mở **Command Prompt** (cmd)
2. Gõ lệnh: `ipconfig`
3. Tìm dòng **IPv4 Address**, ví dụ:
   ```
   IPv4 Address. . . . . . . . . . . . : 192.168.1.100
   ```

### **Linux/Mac:**

1. Mở **Terminal**
2. Gõ lệnh: `ifconfig` hoặc `ip addr`
3. Tìm IP address (thường là `192.168.x.x` hoặc `10.0.x.x`)

### **Hoặc Dùng Script:**

```bash
python test_web_server.py
```

Script sẽ tự động hiển thị IP address.

---

## 🔍 Bước 3: Test Trên Cùng Máy Tính

### **Test Localhost:**

1. Mở trình duyệt trên **cùng máy tính**
2. Gõ: `http://localhost:8080`
3. Hoặc: `http://127.0.0.1:8080`

**Nếu không vào được:**
- ❌ Web server chưa chạy
- ❌ Port 8080 bị chiếm bởi ứng dụng khác
- ❌ Firewall chặn

---

## 🔍 Bước 4: Test Từ Điện Thoại

### **Yêu Cầu:**
- ✅ Điện thoại và máy tính **cùng WiFi**
- ✅ Biết IP address của máy tính
- ✅ Web server đang chạy

### **Cách Test:**

1. **Tìm IP máy tính** (xem Bước 2)
2. **Mở trình duyệt trên điện thoại**
3. **Gõ:** `http://192.168.1.100:8080` (thay bằng IP của bạn)

**Nếu không vào được:**
- ❌ Không cùng WiFi
- ❌ Firewall chặn port 8080
- ❌ IP address sai

---

## 🔧 Giải Pháp

### **Giải Pháp 1: Kiểm Tra Firewall (Windows)**

1. Mở **Windows Defender Firewall**
2. Chọn **Advanced settings**
3. Chọn **Inbound Rules** → **New Rule**
4. Chọn **Port** → **Next**
5. Chọn **TCP** → **Specific local ports**: `8080` → **Next**
6. Chọn **Allow the connection** → **Next**
7. Chọn tất cả profiles → **Next**
8. Đặt tên: "QR Web Server" → **Finish**

**Hoặc tạm thời tắt firewall để test:**
- Control Panel → Windows Defender Firewall → Turn Windows Defender Firewall on or off
- Tắt firewall tạm thời (chỉ để test!)

### **Giải Pháp 2: Chạy Web Server Riêng (Test)**

Nếu web server không start từ `main.py`, thử chạy riêng:

```bash
python web_server.py
```

**Lưu ý:** Web server này sẽ chạy mà không có tracker, chỉ để test connection.

### **Giải Pháp 3: Đổi Port**

Nếu port 8080 bị chiếm, đổi sang port khác:

**Sửa file `src/main_tracker.py`:**
```python
server_thread = threading.Thread(
    target=run_server,
    args=(tracker, '0.0.0.0', 3000, False),  # Đổi 8080 → 3000
    daemon=True
)
```

**Sửa file `mobile_qr_scanner.html`:**
```javascript
let serverUrl = window.location.origin; // Tự động dùng port hiện tại
// Hoặc hardcode:
// let serverUrl = 'http://192.168.1.100:3000';
```

### **Giải Pháp 4: Kiểm Tra Cùng Mạng WiFi**

1. **Trên máy tính:**
   - Mở cmd → `ipconfig`
   - Ghi nhớ IP (ví dụ: `192.168.1.100`)

2. **Trên điện thoại:**
   - Settings → WiFi → Xem IP address
   - Phải cùng subnet (ví dụ: `192.168.1.x`)
   - Nếu khác (ví dụ: `192.168.0.x`) → Không cùng mạng!

3. **Giải pháp:**
   - Kết nối điện thoại vào cùng WiFi với máy tính
   - Hoặc dùng hotspot từ máy tính

### **Giải Pháp 5: Dùng Ngrok (Nếu Khác Mạng)**

Nếu điện thoại và máy tính khác mạng (ví dụ: điện thoại dùng 4G):

1. **Cài đặt ngrok:**
   ```bash
   # Download từ: https://ngrok.com/download
   # Hoặc: pip install pyngrok
   ```

2. **Chạy ngrok:**
   ```bash
   ngrok http 8080
   ```

3. **Lấy public URL:**
   - Ngrok sẽ hiển thị URL: `https://xxxx.ngrok.io`
   - Dùng URL này trên điện thoại

**Lưu ý:** Ngrok free có giới hạn, chỉ dùng để test.

---

## 🐛 Lỗi Thường Gặp

### **Lỗi 1: "Cannot connect to server"**

**Nguyên nhân:**
- Web server chưa chạy
- Firewall chặn
- Port bị chiếm

**Giải pháp:**
- Chạy `python test_web_server.py` để kiểm tra
- Kiểm tra firewall
- Đổi port khác

### **Lỗi 2: "Connection refused"**

**Nguyên nhân:**
- Server chỉ listen trên `127.0.0.1` (localhost)
- Không listen trên `0.0.0.0` (all interfaces)

**Giải pháp:**
- Đảm bảo trong code: `app.run(host='0.0.0.0', port=8080)`
- Không dùng `host='127.0.0.1'` hoặc `host='localhost'`

### **Lỗi 3: "This site can't be reached"**

**Nguyên nhân:**
- IP address sai
- Không cùng mạng
- Server chưa chạy

**Giải pháp:**
- Test trên localhost trước: `http://localhost:8080`
- Kiểm tra IP: `ipconfig` (Windows) hoặc `ifconfig` (Linux/Mac)
- Đảm bảo cùng WiFi

### **Lỗi 4: "Web server started" nhưng không vào được**

**Nguyên nhân:**
- Server start nhưng crash ngay sau đó
- Port bị chiếm
- Import error

**Giải pháp:**
- Kiểm tra console log xem có error không
- Test bằng script: `python test_web_server.py`
- Chạy web server riêng: `python web_server.py`

---

## ✅ Checklist

Trước khi báo lỗi, kiểm tra:

- [ ] Đã chạy `python main.py`?
- [ ] Thấy dòng "✅ Web server started"?
- [ ] Test trên localhost: `http://localhost:8080`?
- [ ] Đã tìm đúng IP address (`ipconfig`)?
- [ ] Điện thoại và máy tính cùng WiFi?
- [ ] Firewall đã cho phép port 8080?
- [ ] Đã test bằng `python test_web_server.py`?

---

## 📞 Test Nhanh

**Chạy script test:**
```bash
python test_web_server.py
```

**Nếu test pass:**
- ✅ Server đang chạy
- ✅ Port mở
- ✅ Dùng IP address hiển thị trên điện thoại

**Nếu test fail:**
- ❌ Xem phần **Giải Pháp** ở trên
- ❌ Kiểm tra console log
- ❌ Kiểm tra firewall

---

**Chúc bạn khắc phục thành công! 🎉**



