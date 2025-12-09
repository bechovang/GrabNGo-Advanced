import torch
from ultralytics import YOLO
import cv2
import time
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO Pose Detection with GPU/CPU support')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'gpu', 'cpu'],
                       help='Device to use for inference: auto (detects GPU), gpu, or cpu')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index to use (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera frame rate (default: 30)')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt',
                       help='YOLO model to use (default: yolov8n-pose.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for inference (default: 640)')
    parser.add_argument('--display-fps', action='store_true',
                       help='Display FPS counter on screen')
    return parser.parse_args()

def get_inference_device(args):
    """Determine which device to use for inference based on user preference and availability"""
    if args.device == 'gpu':
        if torch.cuda.is_available():
            return 0, torch.cuda.get_device_name(0)
        else:
            print("❌ GPU requested but CUDA not available. Falling back to CPU...")
            return 'cpu', 'CPU'
    
    elif args.device == 'cpu':
        return 'cpu', 'CPU'
    
    else:  # auto
        if torch.cuda.is_available():
            return 0, torch.cuda.get_device_name(0)
        else:
            print("⚠️  GPU not available, using CPU")
            return 'cpu', 'CPU'

def main():
    args = parse_arguments()
    
    # Determine inference device
    device_id, device_name = get_inference_device(args)
    
    # Set number of CPU threads if using CPU
    if device_id == 'cpu':
        torch.set_num_threads(8)
        
    print(f"🚀 Using device: {device_name}")
    print(f"📦 Loading YOLO model: {args.model}")
    
    # Load model
    model = YOLO(args.model)
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.camera}. Try changing the camera index.")
        exit()
    
    # Configure webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    print(f"📹 Webcam started with {args.width}x{args.height} @ {args.fps} FPS")
    print("💡 Press 'q' to quit, 's' to save screenshot")
    print("=" * 50)
    
    fps_list = []
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame")
                break
            
            # YOLO inference
            results = model.predict(
                frame,
                device=device_id,
                conf=args.conf,
                verbose=False,
                half=(device_id != 'cpu'),  # Disable FP16 on CPU
                imgsz=args.imgsz
            )
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            fps_list.append(fps)
            
            # Display FPS if requested
            if args.display_fps:
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Display number of people detected
            num_people = len(results[0].boxes) if results[0].boxes is not None else 0
            cv2.putText(
                annotated_frame,
                f"People: {num_people}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow('YOLO Pose Detection (Press Q to quit)', annotated_frame)
            
            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Save screenshot
                screenshot_filename = f"yolo_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_filename, annotated_frame)
                print(f"📷 Screenshot saved as {screenshot_filename}")
            
            frame_count += 1
            
            # Print average FPS every 30 frames
            if frame_count % 30 == 0:
                avg_fps = sum(fps_list[-30:]) / min(30, len(fps_list))
                print(f"Frame {frame_count}: Avg FPS = {avg_fps:.1f}")

    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Show final statistics
        if fps_list:
            print("=" * 50)
            print("📊 Final Statistics:")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Average FPS: {sum(fps_list)/len(fps_list):.1f}")
            print(f"   Max FPS: {max(fps_list):.1f}")
            print(f"   Min FPS: {min(fps_list):.1f}")
            print(f"   Total runtime: {sum(fps_list)/len(fps_list) * frame_count / 1000:.1f} seconds")

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam closed")

if __name__ == '__main__':
    main()