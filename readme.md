# SmartShelf-PACK: Hệ thống Kệ Bán Lẻ Thông Minh với Sensor Fusion

> **Dự án nghiên cứu và phát triển bởi Team Underrated**  
> Dựa trên bài báo khoa học: *"Smart Shelf System for Customer Behavior Tracking in Supermarkets"* (Sensors, 2024).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Keypoint%20R--CNN-EE4C2C)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Phase%201%3A%20Vision%20Backbone-yellow)](https://github.com/)

## 📖 Giới thiệu

**SmartShelf-PACK** là hệ thống theo dõi hành vi khách hàng trong bán lẻ, giải quyết các bài toán khó mà các hệ thống camera đơn thuần (như YOLO) thường gặp phải:
1.  **Occlusion (Che khuất):** Theo dõi chính xác ngay cả khi khách hàng bị che khuất tạm thời.
2.  **Cross-location (Lấy hàng chéo):** Xác định đúng người lấy hàng ngay cả khi họ với tay sang vị trí khác hoặc đứng chen chúc.

Dự án sử dụng kỹ thuật **Sensor Fusion** (Hợp nhất cảm biến), kết hợp dữ liệu từ **Camera 2D** và **Cảm biến trọng lượng (Loadcells)** thông qua thuật toán **PACK-RMPF** (Product Association with Customer Keypoints using RANSAC Modeling and Particle Filtering).

## 🚀 Kiến trúc Hệ thống

Hệ thống bao gồm 3 module chính:

1.  **Hệ thống Thị giác (Vision System) - [Đang thực hiện]:**
    *   **Keypoint R-CNN:** Phát hiện người và trích xuất 17 điểm khớp xương (đặc biệt là cổ tay).
    *   **StrongSORT:** Theo dõi đa đối tượng (Multi-Object Tracking) và gán ID duy nhất (Re-ID).
2.  **Hệ thống Cảm biến (Weight System):**
    *   Mạng lưới Loadcell + HX711 + ESP32 giao tiếp qua MQTT.
    *   Phát hiện sự kiện thay đổi trọng lượng (Pick-up/Put-back) theo thời gian thực.
3.  **Module Hợp nhất (Fusion Core - PACK-RMPF):**
    *   Đồng bộ hóa thời gian (Timestamp Matching).
    *   **Particle Filter:** Ước tính quỹ đạo chuyển động của tay và người.
    *   **RANSAC:** Mô hình hóa xác suất để liên kết hành động lấy hàng với đúng người dùng.

## 🛠️ Cài đặt & Hướng dẫn (Phase 1)

Hiện tại dự án đang ở **Phase 1: Xây dựng Vision Pipeline**.

### Yêu cầu phần cứng
*   PC/Laptop có GPU NVIDIA (Khuyến nghị) để chạy mô hình AI mượt mà.
*   Webcam hoặc Video file để test.

### ✨ Phiên bản SORT Tracker (Đơn giản & Nhanh)

Dự án hiện có **2 phiên bản tracking**:

#### 🚀 **SORT Tracker** (Khuyến nghị - Đơn giản nhất)

**Ưu điểm:**
- ✅ **Không cần clone repo** - SORT được nhúng trực tiếp trong code
- ✅ **Cài đặt cực đơn giản** - chỉ cần 2 lệnh
- ✅ **Nhẹ và nhanh** - SORT chỉ dùng Kalman Filter, không cần Re-ID model
- ✅ **Tracking ổn định** cho các trường hợp đơn giản

**Cài đặt:**
```bash
# 1. Cài đặt dependencies
pip install torch torchvision opencv-python filterpy scipy

# 2. Chạy ngay!
python vision_with_sort.py
```

**Tính năng:**
- ✅ Detection người + keypoints
- ✅ Tracking với ID ổn định 
- ✅ Vẽ cổ tay (wrist) với chấm vàng
- ✅ Mỗi ID có màu riêng
- ✅ Debug info (Frame count, số tracks)
- ✅ In data ra console

#### 🔧 **StrongSORT** (Nâng cao - Cần Re-ID model)

**Cài đặt:**
1.  **Clone dự án:**
    ```bash
    git clone https://github.com/your-username/SmartShelf-PACK.git
    cd SmartShelf-PACK
    ```

2.  **Tạo môi trường ảo (Khuyến nghị):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện phụ thuộc:**
    ```bash
    pip install torch torchvision opencv-python numpy
    ```

4.  **Cài đặt StrongSORT (Submodule):**
    ```bash
    git clone https://github.com/mikel-brostrom/yolov8_tracking.git
    cd yolov8_tracking
    pip install -r requirements.txt
    cd ..
    ```

5.  **Tải trọng số Re-ID:**
    *   Tải file `osnet_x0_25_msmt17.pt` và đặt vào thư mục `yolov8_tracking/strong_sort/deep/checkpoint/`.

## 💻 Cách chạy chương trình

### SORT Tracker (Khuyến nghị):
```bash
python vision_with_sort.py
```

### StrongSORT (Nâng cao):
```bash
python vision_pipeline.py
```