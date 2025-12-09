from ultralytics import YOLO
import torch
import cv2

# Kiểm tra GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
model = YOLO('yolov8n-pose.pt')

# Test với webcam
print("Starting webcam... Press 'q' to quit")
results = model.predict(source=0, device=0, show=True)