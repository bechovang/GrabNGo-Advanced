"""
Smart Retail Tracking System - Production Ready
Using BoT-SORT with native ReID (appearance features)
"""

import torch
import cv2
import json
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime
import numpy as np


class RetailCustomerTracker:
    """
    Production-ready retail customer tracking system.
    Uses BoT-SORT with ReID for robust tracking across occlusions.
    """
    
    def __init__(self, 
                 detection_model='yolo11n.pt',
                 tracker_config='botsort_reid.yaml',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize tracker with ReID configuration.
        
        Args:
            detection_model: YOLO detection model path
            tracker_config: Path to custom tracker config (botsort_reid.yaml)
            device: torch device (cuda or cpu)
        """
        print("🚀 Initializing Retail Customer Tracker...")
        
        self.device = device
        self.model = YOLO(detection_model)
        self.tracker_config = tracker_config
        
        # Customer tracking data
        self.customers = {}  # {track_id: customer_info}
        self.next_customer_id = 1
        self.track_history = defaultdict(lambda: deque(maxlen=50))
        
        # Track buffer for re-identification
        self.lost_tracks = {}  # {track_id: {data, timestamp}}
        self.max_lost_time = 5.0  # 5 seconds max
        
        # Logs
        self.events = []
        
        print(f"✅ Tracker ready | Device: {device}")
        print(f"   Model: {detection_model}")
        print(f"   Config: {tracker_config}")
    
    def process_frame(self, frame, conf=0.6, iou=0.5, return_annotated=True):
        """
        Process single frame with tracking.
        
        Args:
            frame: Input image (BGR, numpy array)
            conf: Detection confidence threshold
            iou: NMS IoU threshold
            return_annotated: Whether to return annotated frame
            
        Returns:
            tuple: (results, annotated_frame or None, track_ids_this_frame)
        """
        # Run YOLO tracking with BoT-SORT + ReID
        # persist=True is CRUCIAL for track continuity
        results = self.model.track(
            frame,
            persist=True,  # ← IMPORTANT: Keep tracker state between frames
            conf=conf,
            iou=iou,
            tracker=self.tracker_config,  # ← Use custom ReID config
            verbose=False,
            device=self.device
        )
        
        result = results[0]
        current_track_ids = set()
        
        # Process detections
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for track_id, box, conf_score in zip(track_ids, boxes, confs):
                current_track_ids.add(int(track_id))
                self._update_track(int(track_id), box, conf_score, frame)
        
        # Handle lost tracks (occlusion detection)
        self._handle_occlusions(current_track_ids)
        
        # Prepare output
        annotated_frame = result.plot() if return_annotated else None
        
        # Draw trajectory
        if return_annotated:
            annotated_frame = self._draw_trajectories(annotated_frame)
        
        return result, annotated_frame, list(current_track_ids)
    
    def _update_track(self, track_id, box, conf, frame):
        """Update or create tracking information for a track."""
        
        if track_id not in self.customers:
            # New customer
            customer_id = f"CUST_{self.next_customer_id:04d}"
            self.next_customer_id += 1
            
            self.customers[track_id] = {
                'customer_id': customer_id,
                'track_id': track_id,
                'entry_time': datetime.now(),
                'entry_box': box,
                'confidence_scores': deque(maxlen=30),
                'suspicious_count': 0,
                'items_detected': set(),
                'last_detection_time': datetime.now()
            }
            
            self.events.append({
                'event': 'ENTRY',
                'customer_id': customer_id,
                'track_id': track_id,
                'timestamp': datetime.now().isoformat(),
                'location': {'x': float(box[0]), 'y': float(box[1])}
            })
            
            print(f"✅ Entry | {customer_id} (Track {track_id})")
        
        # Update existing customer
        customer = self.customers[track_id]
        customer['last_box'] = box
        customer['confidence_scores'].append(conf)
        customer['last_detection_time'] = datetime.now()
        
        # Store trajectory
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        self.track_history[track_id].append((center_x, center_y))
        
        # Detect suspicious behavior (optional: check if reaching for items)
        # This is where you'd add gesture recognition, item detection, etc.
        
        # Clean up lost track entry if customer re-appears
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]
    
    def _handle_occlusions(self, current_tracks):
        """
        Detect and handle occluded (lost) tracks.
        
        This is handled by BoT-SORT track_buffer internally, but we track
        for analysis and potential re-identification.
        """
        current_time = datetime.now()
        
        # Mark lost tracks
        lost = set(self.customers.keys()) - current_tracks
        
        for track_id in lost:
            customer = self.customers[track_id]
            
            if track_id not in self.lost_tracks:
                self.lost_tracks[track_id] = {
                    'lost_time': current_time,
                    'data': customer.copy()
                }
                print(f"⏸️  Occlusion | {customer['customer_id']} (Track {track_id})")
        
        # Clean up tracks lost too long
        for track_id in list(self.lost_tracks.keys()):
            lost_duration = (current_time - self.lost_tracks[track_id]['lost_time']).total_seconds()
            
            if lost_duration > self.max_lost_time:
                customer_id = self.lost_tracks[track_id]['data']['customer_id']
                self._finalize_customer(track_id, customer_id, lost_duration)
                del self.lost_tracks[track_id]
                if track_id in self.customers:
                    del self.customers[track_id]
    
    def _finalize_customer(self, track_id, customer_id, duration):
        """Finalize customer exit."""
        self.events.append({
            'event': 'EXIT',
            'customer_id': customer_id,
            'track_id': track_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': float(duration),
            'suspicious_count': int(self.customers.get(track_id, {}).get('suspicious_count', 0))
        })
        print(f"🚪 Exit | {customer_id} (Duration: {duration:.1f}s)")
    
    def _draw_trajectories(self, frame):
        """Draw movement trajectories on frame."""
        for track_id, points in self.track_history.items():
            if len(points) > 1:
                pts = [(int(p[0]), int(p[1])) for p in points]
                
                # Color based on customer state
                if track_id in self.lost_tracks:
                    color = (0, 165, 255)  # Orange for occluded
                else:
                    color = (0, 255, 0)    # Green for active
                
                # Draw trajectory
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], color, 2)
                
                # Draw current position
                if len(pts) > 0:
                    customer = self.customers.get(track_id, {})
                    customer_id = customer.get('customer_id', 'UNKNOWN')
                    cv2.circle(frame, pts[-1], 5, color, -1)
                    cv2.putText(frame, customer_id, 
                              (pts[-1][0] - 30, pts[-1][1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def get_stats(self):
        """Get current tracking statistics."""
        return {
            'active_customers': len(self.customers),
            'occluded_tracks': len(self.lost_tracks),
            'total_customers': self.next_customer_id - 1,
            'total_events': len(self.events)
        }
    
    def save_events(self, filename='tracking_events.json'):
        """Save all tracking events to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.events, f, indent=2, default=str)
        print(f"💾 Saved {len(self.events)} events to {filename}")


def main():
    """
    Main tracking loop with visualization.
    """
    print("\n" + "="*70)
    print("🎯 SMART RETAIL TRACKING SYSTEM")
    print("   Tracker: BoT-SORT with native ReID")
    print("   Features: Appearance matching, Occlusion handling, Motion prediction")
    print("="*70 + "\n")
    
    # Initialize tracker
    # Make sure botsort_reid.yaml is in ultralytics/cfg/trackers/
    tracker = RetailCustomerTracker(
        detection_model='yolo11n.pt',
        tracker_config='botsort_reid.yaml'
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    frame_count = 0
    fps_list = []
    
    print("📹 Camera ready!")
    print("   Keys: q=quit, s=save logs, i=info\n")
    
    try:
        while True:
            import time
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result, annotated_frame, active_tracks = tracker.process_frame(
                frame,
                conf=0.5,
                iou=0.7
            )
            
            # Display stats
            stats = tracker.get_stats()
            cv2.putText(annotated_frame, 
                       f"Active: {stats['active_customers']} | Occluded: {stats['occluded_tracks']} | Total: {stats['total_customers']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS
            fps = 1 / (time.time() - start_time)
            fps_list.append(fps)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show
            cv2.imshow('Retail Tracking - BoT-SORT + ReID', annotated_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                tracker.save_events()
            elif key == ord('i'):
                print("\n" + "="*70)
                print("📊 TRACKER STATISTICS")
                for k, v in stats.items():
                    print(f"   {k}: {v}")
                print("="*70 + "\n")
            
            frame_count += 1
            if frame_count % 30 == 0:
                avg_fps = sum(fps_list[-30:]) / min(30, len(fps_list))
                print(f"Frame {frame_count}: FPS={avg_fps:.1f}, Active={stats['active_customers']}, Occluded={stats['occluded_tracks']}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Finalize all customers
        current_time = datetime.now()
        for track_id, customer in list(tracker.customers.items()):
            duration = (current_time - customer['entry_time']).total_seconds()
            tracker._finalize_customer(track_id, customer['customer_id'], duration)
        
        # Save logs
        tracker.save_events()
        
        # Statistics
        print("\n" + "="*70)
        print(f"📊 FINAL STATISTICS")
        print(f"   Total frames: {frame_count}")
        print(f"   Average FPS: {sum(fps_list)/len(fps_list):.1f}")
        print(f"   Total customers: {tracker.next_customer_id - 1}")
        print(f"   Total events: {len(tracker.events)}")
        print("="*70)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Done!")


if __name__ == '__main__':
    main()
