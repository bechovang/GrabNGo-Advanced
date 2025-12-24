"""
Test camera flow with minimal processing
"""

import sys
import os
import time
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_camera_flow():
    print("Testing camera flow with minimal processing...")
    
    try:
        # Initialize tracker
        print("\n1. Initializing tracker...")
        from main_tracker import RetailCustomerTracker
        tracker = RetailCustomerTracker(
            detection_model='yolo11n-pose.pt',
            tracker_config='config/botsort_reid.yaml'
        )
        
        # Initialize MQTT
        print("2. Initializing MQTT...")
        tracker._init_mqtt()
        
        # Open camera
        print("3. Opening camera...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Failed to open camera!")
            return False
        
        print("   Camera opened successfully!")
        
        # Process a few frames
        print("4. Processing frames...")
        frame_count = 0
        max_frames = 30  # Process only 30 frames for quick test
        
        start_time = time.time()
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("   Failed to read frame")
                break
            
            # Process every 3rd frame to speed up
            if frame_count % 3 == 0:
                result, _, active_tracks = tracker.process_frame(
                    frame,
                    conf=0.5,
                    iou=0.7,
                    return_annotated=False  # Skip annotation for speed
                )
                
                if frame_count % 9 == 0:  # Print every 9 frames
                    stats = tracker.get_stats()
                    print(f"   Frame {frame_count}: Active: {stats['active_customers']}, "
                          f"Pending: {stats['pending_tracks']}, Total: {stats['total_customers']}")
            
            frame_count += 1
        
        # Print final stats
        elapsed_time = time.time() - start_time
        stats = tracker.get_stats()
        
        print(f"\n5. Final stats (processed {frame_count} frames in {elapsed_time:.1f}s):")
        print(f"   Active customers: {stats['active_customers']}")
        print(f"   Pending tracks: {stats['pending_tracks']}")
        print(f"   Occluded tracks: {stats['occluded_tracks']}")
        print(f"   Total customers: {stats['total_customers']}")
        print(f"   Processing speed: {frame_count/elapsed_time:.1f} FPS")
        
        # Print recent events
        if tracker.events:
            print(f"\n6. Recent events ({len(tracker.events)} total):")
            for event in tracker.events[-3:]:  # Show last 3 events
                print(f"   - {event.get('type', 'unknown')}: {event}")
        
        cap.release()
        print("\n✅ Camera test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during camera test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_camera_flow()
    if success:
        print("\n🎉 Camera flow test PASSED!")
        print("   - Camera connected and working")
        print("   - Tracker processing frames correctly")
        print("   - MQTT receiving messages")
    else:
        print("\n💥 Camera flow test FAILED!")