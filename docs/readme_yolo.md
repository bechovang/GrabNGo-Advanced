# Hướng dẫn cài đặt YOLO Pose trên Windows với GPU

## 📋 Yêu cầu hệ thống

### Phần cứng

- **GPU NVIDIA** với CUDA support (GTX/RTX series)
- RAM: Tối thiểu 8GB (khuyến nghị 16GB+)
- Ổ cứng: Tối thiểu 10GB trống

### Phần mềm

- Windows 10/11
- NVIDIA GPU Driver (phiên bản mới nhất)
- Python 3.11 hoặc 3.12 (⚠️ **KHÔNG dùng Python 3.13**)

---

## 🔍 Bước 1: Kiểm tra GPU và CUDA

### 1.1. Kiểm tra GPU đã cài driver chưa

Mở **Command Prompt** hoặc **PowerShell**, chạy:

```bash
nvidia-smi
```

**Kết quả mong đợi:**

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 566.07                 Driver Version: 566.07         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
...
```

✅ Nếu thấy thông tin GPU → OK, chuyển bước 2

❌ Nếu báo lỗi `'nvidia-smi' is not recognized`:

1. Tải driver NVIDIA tại: https://www.nvidia.com/Download/index.aspx
2. Cài đặt và khởi động lại máy
3. Chạy lại `nvidia-smi`

### 1.2. Ghi nhớ CUDA Version

Từ kết quả `nvidia-smi`, ghi nhớ **CUDA Version** (ví dụ: 12.7, 12.1, 11.8, v.v.)

---

## 🐍 Bước 2: Cài đặt Python 3.11

### 2.1. Kiểm tra Python hiện tại

```bash
python --version
```

⚠️ **Quan trọng:** PyTorch chưa hỗ trợ đầy đủ Python 3.13 với CUDA builds. Bạn PHẢI dùng Python 3.11 hoặc 3.12.

### 2.2. Tải và cài Python 3.11 (nếu cần)

1. Truy cập: https://www.python.org/downloads/release/python-31110/
2. Kéo xuống phần **Files**, tải:

   - **Windows installer (64-bit)** - `python-3.11.10-amd64.exe`

3. Chạy file cài đặt:

   - ✅ **QUAN TRỌNG:** Chọn "**Add python.exe to PATH**"
   - Chọn "**Install Now**"
   - Đợi cài đặt hoàn tất

4. **Khởi động lại Command Prompt** (bắt buộc!)

5. Kiểm tra:

```bash
python --version
# Hoặc
py -3.11 --version
```

---

## 📁 Bước 3: Tạo thư mục dự án và Virtual Environment

### 3.1. Tạo thư mục dự án

```bash
# Tạo thư mục
mkdir C:\YoloPose
cd C:\YoloPose
```

### 3.2. Tạo Virtual Environment

```bash
# Nếu dùng Python 3.11 mặc định
python -m venv venv

# Nếu có nhiều phiên bản Python
py -3.11 -m venv venv
```

### 3.3. Kích hoạt Virtual Environment

```bash
# Windows Command Prompt
venv\Scripts\activate

# Windows PowerShell (nếu gặp lỗi permission)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**Sau khi kích hoạt, bạn sẽ thấy `(venv)` ở đầu dòng lệnh:**

```
(venv) C:\YoloPose>
```

---

## 🔧 Bước 4: Cài đặt PyTorch với CUDA

### 4.1. Cập nhật pip

```bash
python -m pip install --upgrade pip
```

### 4.2. Cài PyTorch với CUDA

**Dựa vào CUDA version từ Bước 1.2:**

#### CUDA 12.x (12.1, 12.4, 12.7):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

⏳ **Lưu ý:** File PyTorch khoảng 2-3GB, cần thời gian tải.

### 4.3. Kiểm tra cài đặt PyTorch

Tạo file `check_gpu.py`:

```python
import torch

print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ WARNING: CUDA not available! Check installation.")
print("=" * 60)
```

Chạy:

```bash
python check_gpu.py
```

**Kết quả mong đợi:**

```
============================================================
PyTorch version: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
cuDNN version: 90100
Number of GPUs: 1
GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU
GPU Memory: 8.00 GB
============================================================
```

✅ **QUAN TRỌNG:**

- Version phải có `+cu124` hoặc `+cu118` (KHÔNG phải `+cpu`)
- `CUDA available` phải là `True`

❌ Nếu thấy `CUDA available: False`:

- Gỡ cài đặt: `pip uninstall torch torchvision torchaudio -y`
- Cài lại với đúng CUDA version
- Kiểm tra driver NVIDIA

---

## 📦 Bước 5: Cài đặt Ultralytics YOLO

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pillow
```

---

## 🧪 Bước 6: Test YOLO Pose

### 6.1. Test cơ bản với ảnh

Tạo file `test_image.py`:

```python
from ultralytics import YOLO
import torch

# Kiểm tra GPU
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
print("\nLoading YOLO Pose model...")
model = YOLO('yolov8n-pose.pt')

# Predict với ảnh từ URL
print("\nRunning prediction on sample image...")
results = model.predict(
    source='https://ultralytics.com/images/bus.jpg',
    device=0,  # GPU
    save=True,
    show=True,
    conf=0.5
)

print("\n✅ Done! Check 'runs/pose/predict' folder for results")
```

Chạy:

```bash
python test_image.py
```

### 6.2. Test với webcam

Tạo file `test_webcam.py`:

```python
from ultralytics import YOLO
import torch
import cv2

def main():
    # Kiểm tra GPU
    if not torch.cuda.is_available():
        print("⚠️ WARNING: GPU not available, using CPU")
        device = 'cpu'
    else:
        device = 0
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("Loading YOLO Pose model...")
    model = YOLO('yolov8n-pose.pt')

    # Mở webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam")
        return

    print("✅ Webcam opened successfully")
    print("Press 'q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Dự đoán
        results = model(frame, device=device, verbose=False, conf=0.5)

        # Vẽ kết quả
        annotated_frame = results[0].plot()

        # Hiển thị thông tin
        frame_count += 1
        cv2.putText(
            annotated_frame,
            f'Frame: {frame_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Hiển thị
        cv2.imshow('YOLO Pose Detection - Press Q to quit', annotated_frame)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Processed {frame_count} frames")

if __name__ == '__main__':
    main()
```

Chạy:

```bash
python test_webcam.py
```

### 6.3. Test với video file

Tạo file `test_video.py`:

```python
from ultralytics import YOLO
import torch

# Load model
model = YOLO('yolov8n-pose.pt')

# Predict với video
print("Processing video...")
results = model.predict(
    source='your_video.mp4',  # Thay bằng đường dẫn video của bạn
    device=0,
    save=True,
    show=False,
    conf=0.5,
    verbose=True
)

print("✅ Done! Check 'runs/pose/predict' folder")
```

---

## 🎯 Bước 7: Code hoàn chỉnh với xử lý Keypoints

Tạo file `pose_analysis.py`:

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Tên các keypoints (COCO format - 17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def analyze_pose(image_path):
    # Load model
    model = YOLO('yolov8n-pose.pt')

    # Predict
    results = model(image_path, device=0)

    # Xử lý kết quả
    for result in results:
        keypoints = result.keypoints

        if keypoints is not None and len(keypoints) > 0:
            # Lấy tọa độ và confidence
            xy = keypoints.xy.cpu().numpy()  # Shape: (num_people, 17, 2)
            conf = keypoints.conf.cpu().numpy()  # Shape: (num_people, 17)

            # Xử lý từng người
            for person_idx in range(xy.shape[0]):
                print(f"\n{'='*60}")
                print(f"Person {person_idx + 1}:")
                print(f"{'='*60}")

                person_kpts = xy[person_idx]
                person_conf = conf[person_idx]

                # In thông tin từng keypoint
                for kpt_idx, kpt_name in enumerate(KEYPOINT_NAMES):
                    x, y = person_kpts[kpt_idx]
                    confidence = person_conf[kpt_idx]

                    if confidence > 0.5:  # Chỉ in keypoints có confidence cao
                        print(f"  {kpt_name:15s}: ({x:6.1f}, {y:6.1f}) - conf: {confidence:.2f}")

        # Hiển thị và lưu kết quả
        result.show()
        result.save('output_pose_analysis.jpg')

if __name__ == '__main__':
    # Test với ảnh
    analyze_pose('https://ultralytics.com/images/bus.jpg')
```

---

## 📊 Các model YOLO Pose có sẵn

| Model             | Size   | Speed     | Accuracy   | Use Case             |
| ----------------- | ------ | --------- | ---------- | -------------------- |
| `yolov8n-pose.pt` | Nano   | Rất nhanh | Trung bình | Webcam real-time     |
| `yolov8s-pose.pt` | Small  | Nhanh     | Tốt        | Cân bằng             |
| `yolov8m-pose.pt` | Medium | TB        | Tốt        | Độ chính xác cao hơn |
| `yolov8l-pose.pt` | Large  | Chậm      | Rất tốt    | Video chất lượng cao |
| `yolov8x-pose.pt` | XLarge | Rất chậm  | Tốt nhất   | Nghiên cứu           |

**Để thay đổi model:**

```python
model = YOLO('yolov8s-pose.pt')  # Thay vì 'n'
```

---

## ⚙️ Tham số quan trọng

```python
results = model.predict(
    source='image.jpg',      # Nguồn: ảnh, video, webcam (0), URL
    device=0,                # 0=GPU, 'cpu'=CPU
    conf=0.5,                # Confidence threshold (0-1)
    iou=0.7,                 # IoU threshold cho NMS
    half=True,               # Dùng FP16 (nhanh hơn trên GPU)
    imgsz=640,               # Kích thước input (320, 640, 1280)
    save=True,               # Lưu kết quả
    show=False,              # Hiển thị kết quả
    verbose=True,            # In log
    stream=False,            # Stream mode cho video
    max_det=10,              # Số người tối đa detect
)
```

---

## 🚀 Tối ưu hiệu suất

### Tăng tốc độ FPS

```python
# 1. Dùng FP16 (half precision)
results = model(frame, device=0, half=True)

# 2. Giảm kích thước input
results = model(frame, device=0, imgsz=320)  # Thay vì 640

# 3. Dùng model nhỏ hơn
model = YOLO('yolov8n-pose.pt')  # Nano - nhanh nhất

# 4. Giảm confidence threshold
results = model(frame, device=0, conf=0.3)

# 5. Stream mode cho video
results = model.predict(source='video.mp4', device=0, stream=True)
```

### Giảm VRAM usage

```python
# 1. Giảm batch size khi train
model.train(data='data.yaml', batch=8)  # Thay vì 16

# 2. Giảm kích thước input
results = model(frame, imgsz=320)

# 3. Dùng model nhỏ
model = YOLO('yolov8n-pose.pt')
```

---

## ❌ Xử lý lỗi thường gặp

### Lỗi: CUDA out of memory

**Nguyên nhân:** GPU không đủ VRAM

**Giải pháp:**

```python
# 1. Dùng model nhỏ hơn
model = YOLO('yolov8n-pose.pt')

# 2. Giảm kích thước input
results = model(frame, imgsz=320)

# 3. Xóa cache
torch.cuda.empty_cache()

# 4. Dùng CPU (chậm hơn)
results = model(frame, device='cpu')
```

### Lỗi: torch.cuda.is_available() = False

**Nguyên nhân:** PyTorch không detect được GPU

**Giải pháp:**

```bash
# 1. Kiểm tra driver
nvidia-smi

# 2. Gỡ và cài lại PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Kiểm tra lại
python -c "import torch; print(torch.cuda.is_available())"
```

### Lỗi: Webcam không mở được

**Giải pháp:**

```python
# Thử các camera index khác
cap = cv2.VideoCapture(0)  # Camera 0
cap = cv2.VideoCapture(1)  # Camera 1

# Kiểm tra camera có hoạt động không
if not cap.isOpened():
    print("Cannot open camera")
else:
    print("Camera OK")
```

### Lỗi: Chạy chậm trên GPU

**Kiểm tra:**

```python
import torch
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

# Đảm bảo dùng device=0
results = model(frame, device=0)  # KHÔNG dùng device='cuda' hay device='cpu'
```

---

## 📚 Training model của bạn (nâng cao)

### Chuẩn bị dataset

Dataset phải theo format COCO Pose:

```
dataset/
├── train/
│   ├── images/
│   └── labels/  # File .txt với keypoints
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

File `data.yaml`:

```yaml
path: ./dataset
train: train/images
val: val/images

# Keypoints
kpt_shape: [17, 3] # 17 keypoints, 3 = [x, y, visibility]

# Classes
names:
  0: person
```

### Training

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n-pose.pt')

# Train
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    patience=50,
    save=True,
    project='runs/pose',
    name='my_pose_model'
)
```

---

## 🔗 Tài liệu tham khảo

- **Ultralytics Docs:** https://docs.ultralytics.com/
- **PyTorch:** https://pytorch.org/
- **YOLO Pose:** https://docs.ultralytics.com/tasks/pose/
- **GitHub Issues:** https://github.com/ultralytics/ultralytics/issues

---

## 📝 Checklist hoàn thành

- [ ] Cài driver NVIDIA và kiểm tra `nvidia-smi`
- [ ] Cài Python 3.11 và thêm vào PATH
- [ ] Tạo virtual environment
- [ ] Cài PyTorch với CUDA (version có `+cu124` hoặc `+cu118`)
- [ ] Kiểm tra `torch.cuda.is_available()` = True
- [ ] Cài Ultralytics và OpenCV
- [ ] Test với ảnh thành công
- [ ] Test với webcam thành công
- [ ] Đọc và hiểu các tham số điều chỉnh

---

## 💡 Tips

1. **Luôn kích hoạt venv trước khi làm việc:**

   ```bash
   venv\Scripts\activate
   ```

2. **Dùng model nhỏ (nano) cho real-time:**

   ```python
   model = YOLO('yolov8n-pose.pt')
   ```

3. **Bật FP16 để tăng tốc:**

   ```python
   results = model(frame, device=0, half=True)
   ```

4. **Lưu model tốt nhất khi training:**

   ```python
   model.train(data='data.yaml', save_period=10)
   ```

5. **Monitor GPU trong khi chạy:**
   ```bash
   # Terminal khác
   watch -n 1 nvidia-smi
   ```

---

**Chúc bạn thành công! 🎉**

Nếu gặp vấn đề, hãy kiểm tra lại từng bước trong checklist.
