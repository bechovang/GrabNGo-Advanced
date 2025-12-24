import cv2
import time

def test_camera():
    print("Testing camera connection...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera!")
        return False
    
    print("Camera opened successfully!")
    
    # Try to read a few frames
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {i+1}")
            cap.release()
            return False
        
        print(f"Frame {i+1} - Size: {frame.shape}")
        time.sleep(0.5)
    
    cap.release()
    print("Camera test successful!")
    return True

if __name__ == "__main__":
    test_camera()