"""
Test the complete flow with camera without visualization
"""

import sys
import os
import time
import cv2
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_tracker import RetailCustomerTracker

def run_tracker_with_camera(duration=10):
    print("Starting tracker with camera...")
    
    # Initialize tracker
    tracker = RetailCustomerTracker(
        detection_model='yolo11n-pose.pt',
        tracker_config='config/botsort_reid.yaml'
    )
    
    # Initialize MQTT
    tracker._init_mqtt()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Failed to open camera!")
        return
    
    print(f"Camera opened! Running for {duration} seconds...")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (without visualization for faster processing)
            result, _, active_tracks = tracker.process_frame(
                frame,
                conf=0.5,
                iou=0.7,
                return_annotated=False  # Skip annotation for speed
            )
            
            frame_count += 1
            
            # Print status every 2 seconds
            if frame_count % 60 == 0:  # Assuming ~30fps
                stats = tracker.get_stats()
                print(f"Frame {frame_count}: Active: {stats['active_customers']}, "
                      f"Pending: {stats['pending_tracks']}, Total: {stats['total_customers']}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cap.release()
        print("\nCamera released")
    
    # Print final stats
    stats = tracker.get_stats()
    print(f"\nFinal stats after {frame_count} frames:")
    print(f"  Active customers: {stats['active_customers']}")
    print(f"  Pending tracks: {stats['pending_tracks']}")
    print(f"  Occluded tracks: {stats['occluded_tracks']}")
    print(f"  Total customers: {stats['total_customers']}")
    
    # Print recent events
    if tracker.events:
        print(f"\nRecent events ({len(tracker.events)} total):")
        for event in tracker.events[-5:]:  # Show last 5 events
            print(f"  - {event.get('type', 'unknown')}: {event}")
    
    return tracker

if __name__ == '__main__':
    tracker = run_tracker_with_camera(duration=10)
    print("\nTest completed!")