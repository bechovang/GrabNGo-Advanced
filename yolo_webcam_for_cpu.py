import torch
from ultralytics import YOLO
import cv2
import time

# Tối ưu CPU
torch.set_num_threads(8)

print(f"CUDA available: {torch.cuda.is_available()}")
print("Loading YOLO model...")

# Load model
model = YOLO('yolov8n-pose.pt')

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam. Try changing to cv2.VideoCapture(1)")
    exit()

# Cài đặt webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Giảm độ phân giải để tăng FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Webcam started! Press 'q' to quit")
print("=" * 50)

fps_list = []
frame_count = 0

try:
    while True:
        start_time = time.time()
        
        # Đọc frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break
        
        # YOLO inference
        results = model.predict(
            frame,
            device='cpu',
            conf=0.5,
            verbose=False,
            half=False,
            imgsz=640  # Kích thước ảnh inference
        )
        
        # Vẽ kết quả lên frame
        annotated_frame = results[0].plot()
        
        # Tính FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_list.append(fps)
        
        # Hiển thị FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Hiển thị số người phát hiện
        num_people = len(results[0].keypoints)
        cv2.putText(
            annotated_frame,
            f"People: {num_people}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Hiển thị frame
        cv2.imshow('YOLO Pose Detection', annotated_frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        # In FPS trung bình mỗi 30 frames
        if frame_count % 30 == 0:
            avg_fps = sum(fps_list[-30:]) / min(30, len(fps_list))
            print(f"Frame {frame_count}: Avg FPS = {avg_fps:.1f}")

except KeyboardInterrupt:
    print("\n⚠️ Stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    # Thống kê cuối
    if fps_list:
        print("=" * 50)
        print(f"📊 Statistics:")
        print(f"   Total frames: {frame_count}")
        print(f"   Average FPS: {sum(fps_list)/len(fps_list):.1f}")
        print(f"   Max FPS: {max(fps_list):.1f}")
        print(f"   Min FPS: {min(fps_list):.1f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam closed")