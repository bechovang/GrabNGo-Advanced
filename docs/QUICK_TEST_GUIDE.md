# Hướng Dẫn Test Nhanh

## Bước 1: Test Imports (30 giây)

### Chạy:
```bash
python -c "from src.main_tracker import RetailCustomerTracker; print('✅ OK')"
```

### Kết quả mong đợi:
```
✅ OK
```
**Nếu lỗi:** Kiểm tra lại cấu trúc thư mục và imports

---

## Bước 2: Test Tracker Khởi Tạo (1 phút)

### Chạy:
```bash
python -c "from src.main_tracker import RetailCustomerTracker; t = RetailCustomerTracker('models/yolo11n-pose.pt', 'config/botsort_reid.yaml'); print('✅ Tracker OK')"
```

### Kết quả mong đợi:
```
🚀 Initializing Retail Customer Tracker...
✅ Tracker ready | Device: cpu
   Model: models/yolo11n-pose.pt
   ...
✅ Tracker OK
```
**Nếu lỗi:** Kiểm tra model file và config file có tồn tại không

---

## Bước 3: Test Dashboard (2 phút)

### Chạy:
```bash
python run_dashboard.py
```

### Kết quả mong đợi:
```
============================================================
SMART RETAIL DASHBOARD (MQTT Only)
============================================================

1. Initializing tracker...
   ✓ Tracker initialized

2. Initializing MQTT...
   ✓ MQTT initialization complete

3. Starting web server...
   ✓ Web server started
   ✓ Dashboard: http://localhost:8080/dashboard
   ✓ Mobile app: http://localhost:8080

============================================================
SYSTEM RUNNING
============================================================
```

**Sau đó:**
- Mở browser: http://localhost:8080/dashboard
- **Kết quả mong đợi:** Dashboard hiển thị, MQTT status hiển thị (Connected/Disconnected)

**Nếu lỗi:**
- Port 8080 đã được dùng → Đổi port trong `run_dashboard.py`
- Flask chưa cài → `pip install flask flask-cors`

---

## Bước 4: Test Camera Tracking (Nếu có camera) (2 phút)

### Chạy:
```bash
python main.py
```

### Kết quả mong đợi:
```
============================================================
🎯 SMART RETAIL TRACKING SYSTEM
============================================================

🚀 Initializing tracker...
   Tracker initialized!

🔌 Initializing MQTT...
   MQTT initialization complete

📹 Opening camera...
   ✅ Camera ready! Frame size: (720, 1280, 3)

📹 Camera ready!
...
```

**Sau đó:**
- Cửa sổ camera hiển thị
- Tracking hoạt động (vẽ boxes, trajectories)
- Nhấn `q` để thoát

**Nếu lỗi:**
- "Camera not available" → Không có camera hoặc camera bị chiếm
- Model không load → Kiểm tra `models/yolo11n-pose.pt` có tồn tại không

---

## Bước 5: Test MQTT Connection (1 phút)

### Chạy:
```bash
python -c "from src.mqtt.client import MQTTClient; m = MQTTClient('test.mosquitto.org', 'test/topic'); m.connect(); print('✅ MQTT:', m.connected)"
```

### Kết quả mong đợi:
```
🔍 Testing connection to test.mosquitto.org:1883...
   Attempting to connect to test.mosquitto.org:1883...
✅ MQTT client initialized (connecting to test.mosquitto.org...)
   Waiting for connection...
✅ MQTT connected to test.mosquitto.org
   Subscribed to: test/topic
✅ MQTT: True
```

**Nếu lỗi:**
- "Cannot reach" → Không có internet hoặc firewall chặn
- "paho-mqtt not installed" → `pip install paho-mqtt`

---

## Checklist Tổng Kết

Sau khi chạy tất cả:

- [ ] **Bước 1:** Imports OK
- [ ] **Bước 2:** Tracker khởi tạo OK
- [ ] **Bước 3:** Dashboard chạy và hiển thị được
- [ ] **Bước 4:** Camera tracking hoạt động (nếu có camera)
- [ ] **Bước 5:** MQTT kết nối được

---

## Lỗi Thường Gặp

### 1. "No module named 'flask'"
**Fix:**
```bash
pip install flask flask-cors
```

### 2. "Model file not found"
**Fix:**
- Kiểm tra `models/yolo11n-pose.pt` có tồn tại không
- Nếu không có, download model về

### 3. "Camera not available"
**Fix:**
- Kiểm tra camera có kết nối không
- Đóng các app đang dùng camera (Zoom, Teams, etc.)

### 4. "MQTT connection failed"
**Fix:**
- Kiểm tra internet
- Thử broker khác: `broker.hivemq.com` hoặc local broker

### 5. "Port 8080 already in use"
**Fix:**
- Đổi port trong `run_dashboard.py`: `port=8081`
- Hoặc đóng app đang dùng port 8080

---

## Kết Luận

Nếu tất cả 5 bước đều OK → **Hệ thống refactor thành công! ✅**

Thời gian test: ~5-10 phút

