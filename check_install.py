import torch
from ultralytics import YOLO
import cv2
import platform

print("=" * 60)
print(f"System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available (Metal): {torch.backends.mps.is_available()}")
print(f"OpenCV version: {cv2.__version__}")
print(f"CPU Cores: {torch.get_num_threads()} threads")
print("=" * 60)

# Test YOLO
print("\nTesting YOLO...")
model = YOLO('yolov8n-pose.pt')
print("✅ YOLO model loaded successfully!")