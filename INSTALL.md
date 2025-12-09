# Hướng dẫn cài đặt và chạy code

## 📋 Yêu cầu
- Python 3.11 hoặc 3.12 (KHÔNG dùng Python 3.13)
- GPU NVIDIA với CUDA support (nếu muốn dùng GPU)
- Windows 10/11

## 🚀 Các bước cài đặt

### Bước 1: Kiểm tra GPU và CUDA (nếu dùng GPU)

Mở PowerShell hoặc Command Prompt, chạy:
```bash
nvidia-smi
```

Ghi nhớ **CUDA Version** (ví dụ: 12.7, 12.1, 11.8)

### Bước 2: Kích hoạt Virtual Environment

Bạn đã có thư mục `venv`, kích hoạt nó:

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate
```

Sau khi kích hoạt, bạn sẽ thấy `(venv)` ở đầu dòng lệnh.

### Bước 3: Cập nhật pip

```bash
python -m pip install --upgrade pip
```

### Bước 4: Cài đặt PyTorch với CUDA (QUAN TRỌNG)

**⚠️ LƯU Ý:** PyTorch cần cài riêng với CUDA support, KHÔNG dùng `pip install -r requirements.txt` cho PyTorch.

**Cho CUDA 12.x (12.1, 12.4, 12.7):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Cho CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Nếu không có GPU hoặc muốn dùng CPU:**
```bash
pip install torch torchvision torchaudio
```

### Bước 5: Cài đặt các package còn lại

```bash
pip install ultralytics opencv-python numpy pillow
```

Hoặc nếu đã cài PyTorch, bạn có thể cài từ requirements.txt (bỏ qua torch):
```bash
pip install ultralytics opencv-python numpy pillow
```

### Bước 6: Kiểm tra cài đặt

Tạo file `check_install.py`:
```python
import torch
from ultralytics import YOLO
import cv2

print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"OpenCV version: {cv2.__version__}")
print("=" * 60)
```

Chạy:
```bash
python check_install.py
```

**Kết quả mong đợi:**
- PyTorch version có `+cu124` hoặc `+cu118` (nếu dùng GPU)
- `CUDA available: True` (nếu có GPU)
- OpenCV version hiển thị

### Bước 7: Tải model YOLO (nếu chưa có)

Model `yolov8n-pose.pt` sẽ tự động tải khi chạy code lần đầu, hoặc tải thủ công:
```python
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
```

## 🎯 Chạy code

### Chạy test với ảnh:
```bash
python test_yolo.py
```

### Chạy với webcam:
```bash
python yolo_webcam.py
```

Nhấn `q` để thoát.

## ❌ Xử lý lỗi

### Lỗi: CUDA not available
- Kiểm tra lại bước 4: Cài PyTorch với đúng CUDA version
- Chạy `nvidia-smi` để kiểm tra driver
- Gỡ và cài lại PyTorch:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Lỗi: Webcam không mở được
- Kiểm tra webcam có đang được dùng bởi ứng dụng khác không
- Thử đổi camera index trong code: `source=0` thành `source=1`

### Lỗi: Module not found
- Đảm bảo đã kích hoạt virtual environment
- Cài lại package bị thiếu: `pip install <package_name>`

## 📝 Lưu ý

1. **Luôn kích hoạt venv trước khi chạy code:**
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

2. **Nếu dùng GPU:** Đảm bảo PyTorch version có `+cu124` hoặc `+cu118` (KHÔNG phải `+cpu`)

3. **Model tự động tải:** Lần đầu chạy, model sẽ tự động tải về (~6MB cho yolov8n-pose.pt)

