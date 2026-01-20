"""
Smart Retail Tracking System - Production Ready
Using BoT-SORT with native ReID (appearance features)
+ Manual Confirmation for New IDs
"""

import torch
import cv2
import json
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
from enum import Enum

# Import modules
from .tracker.reid import LightweightReID
from .mqtt.client import MQTTClient

# Import holding_detector with fallback for different execution contexts
# Make it optional since holding detection is currently disabled
HoldingDetector = None
try:
    try:
        from .holding_detector import HoldingDetector
    except (ImportError, ValueError):
        # Fallback for when running as script or from different context
        import os
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from src.holding_detector import HoldingDetector
except ImportError:
    # MediaPipe might not be available - holding detection will be disabled
    class HoldingDetector:
        def __init__(self):
            pass
        def reset_customer(self, customer_id):
            pass


class TrackState(Enum):
    """Track states for manual confirmation system."""
    PENDING = "PENDING"      # Waiting for manual confirmation
    CONFIRMED = "CONFIRMED"  # Manually confirmed by user


class RetailCustomerTracker:
    """
    Production-ready retail customer tracking system.
    Uses BoT-SORT with ReID for robust tracking across occlusions.
    """
    
    def __init__(self, 
                 detection_model='models/yolo11n-pose.pt',  # Changed to pose model
                 tracker_config='config/botsort_reid.yaml',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize tracker with ReID configuration.
        
        Args:
            detection_model: YOLO detection model path
            tracker_config: Path to custom tracker config (botsort_reid.yaml)
            device: torch device (cuda or cpu)
        """
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
        # ReID enhancements
        self.reid = LightweightReID()
        self.reid_high_thresh = 0.50  # Stage 1: high confidence
        self.reid_low_thresh = 0.30   # Stage 2: low confidence
        self.feature_gallery_size = 10
        
        # Manual Confirmation System
        self.pending_tracks = {}  # {track_id: {data, first_seen_time}}
        self.selected_pending_index = 0  # For 1-9 selection
        self.pending_timeout = 10.0  # Auto-remove pending after 10s
        
        # Validation requirements for confirmation
        self.min_samples_required = 5  # Need 5 feature samples
        self.min_confidence_avg = 0.5   # Average confidence >= 0.5
        self.min_feature_quality = 0.3  # At least 30% valid features
        
        # Holding Detection System (optional - currently disabled)
        try:
            self.holding_detector = HoldingDetector() if HoldingDetector else None
        except Exception:
            self.holding_detector = None
        
        # QR Zone Configuration (right side, full height)
        # Zone will be set dynamically based on frame size, but default to percentage
        self.qr_zone_percent = {
            'x1_percent': 0.7,    # 70% from left (right side)
            'y1_percent': 0.0,    # 0% from top (top edge - full height)
            'x2_percent': 1.0,    # 100% from left (right edge)
            'y2_percent': 1.0     # 100% from top (bottom edge - full height)
        }
        self.qr_zone_pixels = None  # Will be calculated from frame size
        self.zone_active_pending = None  # Which PENDING is currently in zone
        self.zone_overlap_threshold = 0.5  # 50% of person must be in zone
        
        # QR Confirmation System
        self.pending_confirmations = {}  # {customer_id: pending_id} - for auto-matching
        
        # MQTT Configuration for Weight-Based Pickup Detection
        self.mqtt_broker = "test.mosquitto.org"
        self.mqtt_topic_weight = "my-shop/shelf-1/events"
        self.mqtt_client = None  # Will be MQTTClient instance
        
        # Shelf Zone Configuration (where items are placed)
        # Default: Left side, middle-bottom area (adjust based on camera view)
        self.shelf_zone_percent = {
            'x1_percent': 0.0,    # Left edge
            'y1_percent': 0.3,    # 30% from top
            'x2_percent': 0.5,    # 50% from left (middle of frame)
            'y2_percent': 0.9     # 90% from top (near bottom)
        }
        self.shelf_zone_pixels = None  # Will be calculated from frame size
        
        # Zone editing (click and drag)
        self.dragging_zone = None  # 'qr' or 'shelf' or None
        self.drag_start = None
        self.drag_corner = None  # 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'move'
        self.drag_zone_start = None
        
        # Weight Event Handling (will be managed by MQTTClient)
        self.weight_event_timeout = 3.0  # Match events within 3 seconds
        
        # Logs
        self.events = []
        
        # Shared stats manager (for multi-process stats sharing)
        try:
            from .utils.stats_manager import StatsManager
            self.stats_manager = StatsManager()
        except ImportError:
            self.stats_manager = None
        
        # Tracker ready
    
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
        # Run YOLO tracking with BoT-SORT + ReID (for people with pose)
        # persist=True is CRUCIAL for track continuity
        results = self.model.track(
            frame,
            persist=True,  # ← IMPORTANT: Keep tracker state between frames
            conf=conf,
            iou=iou,
            tracker=self.tracker_config,  # ← Use custom ReID config
            verbose=False,
            device=self.device,
            classes=[0]  # Only detect person class for tracking
        )
        
        result = results[0]
        current_track_ids = set()
        
        # Process person detections
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            
            # Keypoints check (no debug print)
            
            for idx, (track_id, box, conf_score) in enumerate(zip(track_ids, boxes, confs)):
                current_track_ids.add(int(track_id))
                person_keypoints = keypoints[idx] if keypoints is not None else None
                self._update_track(int(track_id), box, conf_score, frame, person_keypoints, None)
        
        # Handle lost tracks (occlusion detection)
        self._handle_occlusions(current_track_ids)
        
        # Cleanup old pending tracks
        self._cleanup_pending_tracks()
        
        # Prepare output
        annotated_frame = result.plot(labels=False) if return_annotated else None
        
        # Update QR zone and Shelf zone based on frame size
        if return_annotated and annotated_frame is not None:
            self._update_qr_zone(annotated_frame.shape)
            self._update_shelf_zone(annotated_frame.shape)
            # Check QR zone status
            self._check_qr_zone()
        
        # MQTT messages are handled automatically by loop_start() background thread
        # No need to check here - callbacks will be called automatically
        
        # Draw trajectory and custom overlays
        if return_annotated:
            annotated_frame = self._draw_trajectories(annotated_frame)
            annotated_frame = self._draw_pending_tracks(annotated_frame, result)
            annotated_frame = self._draw_qr_zone(annotated_frame)
            annotated_frame = self._draw_shelf_zone(annotated_frame)
            # Holding status display - TEMPORARILY DISABLED
            # annotated_frame = self._draw_holding_status(annotated_frame)
        
        return result, annotated_frame, list(current_track_ids)
    
    def _update_track(self, track_id, box, conf, frame, keypoints=None, detected_objects=None):
        """Update or create tracking information for a track."""
        
        # Extract appearance features
        features = self.reid.extract_features(frame, box)
        
        # Store frame height for legs visibility check
        frame_height = frame.shape[0] if frame is not None else None

        # Try ReID with lost tracks before creating new
        if track_id not in self.customers and track_id not in self.pending_tracks:
            # Get all current boxes for relative checking
            all_current_boxes = []
            for other_track_id, other_customer in self.customers.items():
                if other_track_id != track_id and other_customer.get('last_box') is not None:
                    all_current_boxes.append(other_customer['last_box'])
            for other_track_id, other_pending in self.pending_tracks.items():
                if other_track_id != track_id and other_pending.get('box') is not None:
                    all_current_boxes.append(other_pending['box'])
            
            # Get keypoints and frame height for legs visibility check
            matched = self._try_reid(track_id, box, features, all_current_boxes, keypoints, frame_height)
            if matched:
                customer = matched
                self.customers[track_id] = customer
            else:
                # Create PENDING track (requires manual confirmation)
                pending_id = f"PENDING_{track_id:04d}"
                self.pending_tracks[track_id] = {
                    'pending_id': pending_id,
                'track_id': track_id,
                    'state': TrackState.PENDING,
                    'first_seen': datetime.now(),
                    'box': box,
                    'features': features,
                'confidence_scores': deque(maxlen=30),
                    'feature_gallery': deque(maxlen=self.feature_gallery_size),
                    'keypoints': keypoints,  # Store keypoints for leg visibility check
                    'frame_height': frame.shape[0] if frame is not None else None,  # Store frame height
                }
                # Add initial samples
                if features is not None:
                    self.pending_tracks[track_id]['feature_gallery'].append(features)
                self.pending_tracks[track_id]['confidence_scores'].append(conf)
                return  # Don't update until confirmed
        
        # Update existing PENDING track
        if track_id in self.pending_tracks:
            pending = self.pending_tracks[track_id]
            pending['box'] = box
            pending['confidence_scores'].append(conf)
            if features is not None:
                pending['feature_gallery'].append(features)
            # Update keypoints and frame height for leg visibility check
            if keypoints is not None:
                pending['keypoints'] = keypoints
            if frame is not None:
                pending['frame_height'] = frame.shape[0]
            
            # Get all current boxes for relative checking
            all_current_boxes = []
            for other_track_id, other_customer in self.customers.items():
                if other_track_id != track_id and other_customer.get('last_box') is not None:
                    all_current_boxes.append(other_customer['last_box'])
            for other_track_id, other_pending in self.pending_tracks.items():
                if other_track_id != track_id and other_pending.get('box') is not None:
                    all_current_boxes.append(other_pending['box'])
            
            # Check if ready for confirmation (with relative checking)
            is_valid, validation_score, _ = self._validate_pending_track(pending, all_current_boxes)
            if is_valid and len(pending['feature_gallery']) == self.min_samples_required:
                # Track ready for confirmation
                pass
            
            return
        
        # Update existing CONFIRMED customer
        if track_id not in self.customers:
            return
        customer = self.customers[track_id]
        customer['last_box'] = box
        customer['confidence_scores'].append(conf)
        customer['last_detection_time'] = datetime.now()
        # Store keypoints and frame height for legs visibility check in re-tracking
        customer['last_keypoints'] = keypoints
        customer['last_frame_height'] = frame_height
        
        # Update feature gallery
        if features is not None:
            if 'feature_gallery' not in customer:
                customer['feature_gallery'] = deque(maxlen=self.feature_gallery_size)
            customer['feature_gallery'].append(features)
        
        # Store trajectory
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        self.track_history[track_id].append((center_x, center_y))
        
        # Holding Detection - TEMPORARILY DISABLED
        # if keypoints is not None:
        #     holding_result = self.holding_detector.detect_holding(...)
        #     customer['holding_status'] = holding_result
        
        # Clean up lost track entry if customer re-appears
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]
    
    def _validate_pending_track(self, pending, all_current_boxes=None):
        """
        Validate if a pending track has enough information for confirmation.
        Also checks for relatives (people nearby) to ensure proper identification.
        
        Returns:
            tuple: (is_valid: bool, validation_score: float, issues: list)
        """
        issues = []
        scores = []
        
        # 1. Check feature samples count
        feature_count = len(pending['feature_gallery'])
        valid_features = sum(1 for f in pending['feature_gallery'] if f is not None)
        
        if feature_count < self.min_samples_required:
            issues.append(f"Need {self.min_samples_required} samples, got {feature_count}")
            scores.append(feature_count / self.min_samples_required)
        else:
            scores.append(1.0)
        
        # 2. Check feature quality (% of valid features)
        if feature_count > 0:
            feature_quality = valid_features / feature_count
            if feature_quality < self.min_feature_quality:
                issues.append(f"Feature quality {feature_quality:.1%} < {self.min_feature_quality:.0%}")
                scores.append(feature_quality / self.min_feature_quality)
            else:
                scores.append(1.0)
        else:
            issues.append("No features extracted")
            scores.append(0.0)
        
        # 3. Check detection confidence
        conf_scores = list(pending['confidence_scores'])
        if len(conf_scores) > 0:
            avg_conf = np.mean(conf_scores)
            if avg_conf < self.min_confidence_avg:
                issues.append(f"Avg confidence {avg_conf:.2f} < {self.min_confidence_avg:.2f}")
                scores.append(avg_conf / self.min_confidence_avg)
            else:
                scores.append(1.0)
        else:
            issues.append("No confidence data")
            scores.append(0.0)
        
        # 4. Check feature consistency (variance)
        if valid_features >= 3:
            valid_feats = [f for f in pending['feature_gallery'] if f is not None]
            feat_array = np.array(valid_feats)
            feat_std = np.std(feat_array, axis=0).mean()
            
            # Lower variance = more consistent (better)
            consistency_score = max(0, 1.0 - feat_std)  # Normalize
            if consistency_score < 0.5:
                issues.append(f"Features inconsistent (var: {feat_std:.3f})")
                scores.append(consistency_score)
            else:
                scores.append(1.0)
        else:
            scores.append(0.5)  # Neutral if not enough samples
        
        # 5. Check if upper body is visible - HARD REQUIREMENT (must see head/torso)
        upper_body_visible = self._check_upper_body_visible(pending)
        if not upper_body_visible:
            issues.append("❌ CRITICAL: Upper body not visible - need to see head/torso")
            scores.append(0.0)  # Critical: must see upper body
            # HARD REQUIREMENT: If upper body not visible, validation fails immediately
            validation_score = 0.0
            is_valid = False
            return is_valid, validation_score, issues
        else:
            scores.append(1.0)
        
        # 6. Check if legs are visible - HARD REQUIREMENT (must see at least 1 ankle keypoint - orange)
        legs_visible = self._check_legs_visible(pending)
        if not legs_visible:
            issues.append("❌ CRITICAL: Legs not visible - need at least 1 ankle keypoint (orange) to identify pants color")
            scores.append(0.0)  # Critical: must see at least 1 ankle keypoint
            # HARD REQUIREMENT: If legs not visible, validation fails immediately
            validation_score = 0.0
            is_valid = False
            return is_valid, validation_score, issues
        else:
            scores.append(1.0)
        
        # 7. Check for relatives nearby - NEW
        # If relatives detected, require more samples for validation
        if all_current_boxes is not None and pending.get('box') is not None:
            box = pending['box']
            box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            
            has_relatives = self._check_relatives_nearby(
                box, box, all_current_boxes, box_center, box_width, box_height
            )
            
            if has_relatives:
                # Require more samples when relatives are nearby
                required_samples = self.min_samples_required + 2
                if feature_count < required_samples:
                    issues.append(f"Relatives detected nearby - need {required_samples} samples (got {feature_count})")
                    scores.append(feature_count / required_samples)
                else:
                    scores.append(1.0)
        
        # Overall validation score
        validation_score = np.mean(scores) if scores else 0.0
        
        # Validation passes if:
        # 1. Upper body and legs are visible (already checked above - hard requirements)
        # 2. Overall score >= 0.8
        # Note: issues from other checks (samples, quality, etc.) are warnings but not blockers
        #       if upper body and legs are visible
        is_valid = validation_score >= 0.8
        
        return is_valid, validation_score, issues
    
    def confirm_pending_track(self, track_id=None):
        """Confirm a pending track to create a customer ID."""
        if track_id is None:
            # Auto-select first pending
            if not self.pending_tracks:
                return
            track_id = list(self.pending_tracks.keys())[self.selected_pending_index % len(self.pending_tracks)]
        
        if track_id not in self.pending_tracks:
            return
        
        pending = self.pending_tracks[track_id]
        
        # Get all current boxes for relative checking
        all_current_boxes = []
        for other_track_id, other_customer in self.customers.items():
            if other_track_id != track_id and other_customer.get('last_box') is not None:
                all_current_boxes.append(other_customer['last_box'])
        for other_track_id, other_pending in self.pending_tracks.items():
            if other_track_id != track_id and other_pending.get('box') is not None:
                all_current_boxes.append(other_pending['box'])
        
        # Validate before confirming (with relative checking)
        is_valid, validation_score, issues = self._validate_pending_track(pending, all_current_boxes)
        
        if not is_valid:
            return
        
        # Create confirmed customer
        customer_id = f"CUST_{self.next_customer_id:04d}"
        self.next_customer_id += 1
        
        self.customers[track_id] = {
            'customer_id': customer_id,
            'track_id': track_id,
            'state': TrackState.CONFIRMED,
            'entry_time': pending['first_seen'],
            'entry_box': pending['box'],
            'last_box': pending['box'],
            'confidence_scores': pending['confidence_scores'],
            'suspicious_count': 0,
            'items_detected': set(),
            'last_detection_time': datetime.now(),
            'feature_gallery': pending['feature_gallery'],
            'holding_status': {},
            'was_holding': False,
            # Shopping cart for weight-based pickup detection
            'shopping_cart': [],
            'pickup_count': 0,
            'last_pickup_time': None,
        }
        
        self.events.append({
            'event': 'ENTRY',
            'customer_id': customer_id,
            'track_id': track_id,
            'timestamp': datetime.now().isoformat(),
            'location': {'x': float(pending['box'][0]), 'y': float(pending['box'][1])}
        })
        
        # Remove from pending
        del self.pending_tracks[track_id]
        
        # Report validation details
        feature_count = len(pending['feature_gallery'])
        avg_conf = np.mean(pending['confidence_scores']) if pending['confidence_scores'] else 0
    
    def select_pending_track(self, index):
        """Select a pending track by index (1-9)."""
        if not self.pending_tracks:
            return
        self.selected_pending_index = index - 1
        track_ids = list(self.pending_tracks.keys())
        if 0 <= self.selected_pending_index < len(track_ids):
            track_id = track_ids[self.selected_pending_index]
            pending = self.pending_tracks[track_id]
    
    def _cleanup_pending_tracks(self):
        """Remove old pending tracks that timeout."""
        current_time = datetime.now()
        to_remove = []
        for track_id, pending in self.pending_tracks.items():
            age = (current_time - pending['first_seen']).total_seconds()
            if age > self.pending_timeout:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.pending_tracks[track_id]
    
    def _update_qr_zone(self, frame_shape):
        """Update QR zone pixel coordinates based on frame size."""
        if frame_shape is None or len(frame_shape) < 2:
            return
        
        height, width = frame_shape[0], frame_shape[1]
        
        # Calculate pixel coordinates from percentages (bottom-left corner)
        self.qr_zone_pixels = {
            'x1': int(width * self.qr_zone_percent['x1_percent']),
            'y1': int(height * self.qr_zone_percent['y1_percent']),
            'x2': int(width * self.qr_zone_percent['x2_percent']),
            'y2': int(height * self.qr_zone_percent['y2_percent'])
        }
    
    def _is_in_qr_zone(self, box):
        """Check if bounding box overlaps with QR zone."""
        if self.qr_zone_pixels is None or box is None:
            return False
        
        x1, y1, x2, y2 = map(int, box)
        zone = self.qr_zone_pixels
        
        # Calculate overlap area
        overlap_x1 = max(x1, zone['x1'])
        overlap_y1 = max(y1, zone['y1'])
        overlap_x2 = min(x2, zone['x2'])
        overlap_y2 = min(y2, zone['y2'])
        
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 0:
                overlap_ratio = overlap_area / box_area
                return overlap_ratio >= self.zone_overlap_threshold
        
        return False
    
    def _check_qr_zone(self):
        """Check which PENDING track is currently in QR zone."""
        pending_in_zone = None
        pending_count = 0
        
        for track_id, pending in self.pending_tracks.items():
            box = pending.get('box')
            if box is None:
                continue
            
            if self._is_in_qr_zone(box):
                pending_count += 1
                if pending_in_zone is None:
                    pending_in_zone = pending.get('pending_id')
        
        # Update zone status
        if pending_count == 1:
            self.zone_active_pending = pending_in_zone
        else:
            self.zone_active_pending = None
        
        return self.zone_active_pending is not None, self.zone_active_pending, pending_count
    
    def _draw_qr_zone(self, frame):
        """Draw QR zone overlay on frame."""
        if self.qr_zone_pixels is None:
            return frame
        
        zone = self.qr_zone_pixels
        x1, y1 = zone['x1'], zone['y1']
        x2, y2 = zone['x2'], zone['y2']
        
        # Check zone status
        zone_active, pending_id, pending_count = self._check_qr_zone()
        
        # Draw rectangle
        is_dragging = self.dragging_zone == 'qr'
        if is_dragging:
            color = (255, 255, 0)  # Yellow when dragging
            thickness = 3
            # Draw corner handles
            corner_size = 10
            cv2.circle(frame, (x1, y1), corner_size, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y1), corner_size, (0, 255, 255), -1)
            cv2.circle(frame, (x1, y2), corner_size, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y2), corner_size, (0, 255, 255), -1)
        elif zone_active and pending_count == 1:
            color = (0, 255, 0)  # Green = active, ready to scan
            thickness = 3
        else:
            color = (0, 0, 255)  # Red = inactive, waiting
            thickness = 2
            # Draw corner handles when not active (for editing)
            corner_size = 8
            cv2.circle(frame, (x1, y1), corner_size, color, 2)
            cv2.circle(frame, (x2, y1), corner_size, color, 2)
            cv2.circle(frame, (x1, y2), corner_size, color, 2)
            cv2.circle(frame, (x2, y2), corner_size, color, 2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if is_dragging:
            label = "QR ZONE [DRAGGING] - Release to finish"
        elif zone_active and pending_count == 1:
            label = f"QR ZONE ✅ ACTIVE - {pending_id}"
        elif pending_count > 1:
            label = f"QR ZONE ⚠️ MULTIPLE ({pending_count})"
        else:
            label = "QR ZONE ⏸️ WAITING (Click & drag to adjust)"
        
        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0] + 10, label_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 5, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def _update_shelf_zone(self, frame_shape):
        """Update shelf zone pixel coordinates from frame size."""
        if len(frame_shape) < 2:
            return
        h, w = frame_shape[:2]
        self.shelf_zone_pixels = {
            'x1': int(w * self.shelf_zone_percent['x1_percent']),
            'y1': int(h * self.shelf_zone_percent['y1_percent']),
            'x2': int(w * self.shelf_zone_percent['x2_percent']),
            'y2': int(h * self.shelf_zone_percent['y2_percent'])
        }
    
    def _is_in_shelf_zone(self, box):
        """Check if person's bounding box overlaps with shelf zone."""
        if self.shelf_zone_pixels is None or box is None:
            return False
        
        x1, y1, x2, y2 = box
        sx1 = self.shelf_zone_pixels['x1']
        sy1 = self.shelf_zone_pixels['y1']
        sx2 = self.shelf_zone_pixels['x2']
        sy2 = self.shelf_zone_pixels['y2']
        
        # Calculate overlap
        overlap_x1 = max(x1, sx1)
        overlap_y1 = max(y1, sy1)
        overlap_x2 = min(x2, sx2)
        overlap_y2 = min(y2, sy2)
        
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return False
        
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        person_area = (x2 - x1) * (y2 - y1)
        overlap_ratio = overlap_area / person_area if person_area > 0 else 0
        
        return overlap_ratio >= 0.3  # At least 30% of person in zone
    
    def _draw_shelf_zone(self, frame):
        """Draw shelf zone on frame for visualization."""
        if self.shelf_zone_pixels is None:
            return frame
        
        zone = self.shelf_zone_pixels
        x1, y1 = zone['x1'], zone['y1']
        x2, y2 = zone['x2'], zone['y2']
        
        # Draw rectangle
        is_dragging = self.dragging_zone == 'shelf'
        if is_dragging:
            color = (255, 255, 0)  # Yellow when dragging
            thickness = 3
            # Draw corner handles
            corner_size = 10
            cv2.circle(frame, (x1, y1), corner_size, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y1), corner_size, (0, 255, 255), -1)
            cv2.circle(frame, (x1, y2), corner_size, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y2), corner_size, (0, 255, 255), -1)
        else:
            color = (0, 255, 255)  # Cyan
            thickness = 2
            # Draw corner handles (for editing)
            corner_size = 8
            cv2.circle(frame, (x1, y1), corner_size, color, 2)
            cv2.circle(frame, (x2, y1), corner_size, color, 2)
            cv2.circle(frame, (x1, y2), corner_size, color, 2)
            cv2.circle(frame, (x2, y2), corner_size, color, 2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if is_dragging:
            label = "SHELF ZONE [DRAGGING] - Release to finish"
        else:
            label = "SHELF ZONE (Click & drag to adjust)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0] + 10, label_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 5, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def _init_mqtt(self):
        """Initialize MQTT client and subscribe to weight events."""
        # Create MQTT client with callback
        self.mqtt_client = MQTTClient(
            broker=self.mqtt_broker,
            topic=self.mqtt_topic_weight,
            on_weight_event=self._handle_weight_event
        )
        
        # Connect
        self.mqtt_client.connect()
    
    @property
    def mqtt_connected(self):
        """Get MQTT connection status (synced with client)."""
        if self.mqtt_client:
            return self.mqtt_client.connected
        return False
    
    def _find_customers_in_shelf_zone(self):
        """Find all confirmed customers currently in shelf zone."""
        candidates = []
        
        for track_id, customer in self.customers.items():
            box = customer.get('last_box')
            if box is None:
                continue
            
            if self._is_in_shelf_zone(box):
                candidates.append({
                    'track_id': track_id,
                    'customer_id': customer.get('customer_id'),
                    'box': box,
                    'keypoints': customer.get('keypoints'),
                    'last_detection_time': customer.get('last_detection_time')
                })
        
        return candidates
    
    def _is_hand_reaching_toward(self, wrist_x, wrist_y, person_box, shelf_center_x, shelf_center_y):
        """Check if hand is extended toward shelf."""
        person_center_x = (person_box[0] + person_box[2]) / 2
        person_center_y = (person_box[1] + person_box[3]) / 2
        
        # Vector from person center to shelf center
        to_shelf_x = shelf_center_x - person_center_x
        to_shelf_y = shelf_center_y - person_center_y
        
        # Vector from person center to wrist
        to_wrist_x = wrist_x - person_center_x
        to_wrist_y = wrist_y - person_center_y
        
        # Calculate angle between vectors
        dot = to_shelf_x * to_wrist_x + to_shelf_y * to_wrist_y
        mag_shelf = np.sqrt(to_shelf_x**2 + to_shelf_y**2)
        mag_wrist = np.sqrt(to_wrist_x**2 + to_wrist_y**2)
        
        if mag_shelf == 0 or mag_wrist == 0:
            return False
        
        cos_angle = dot / (mag_shelf * mag_wrist)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        
        # If angle < 45 degrees, hand is reaching toward shelf
        return angle_deg < 45
    
    def _calculate_hand_reaching_score(self, keypoints, box, shelf_center_x, shelf_center_y):
        """Calculate score based on hand position (reaching toward shelf)."""
        if keypoints is None or len(keypoints.shape) < 2:
            return 0.5  # Neutral if no keypoints
        
        # YOLO pose keypoints: 17 keypoints
        # Index 9: left_wrist, Index 10: right_wrist
        # Format: [x, y, confidence]
        
        score = 0.0
        hand_count = 0
        
        # Check left wrist (index 9)
        if len(keypoints) > 9 and keypoints[9][2] > 0.3:  # confidence > 0.3
            wrist_x, wrist_y = keypoints[9][0], keypoints[9][1]
            # Check if wrist is extended toward shelf
            if self._is_hand_reaching_toward(wrist_x, wrist_y, box, shelf_center_x, shelf_center_y):
                score += 0.5
            hand_count += 1
        
        # Check right wrist (index 10)
        if len(keypoints) > 10 and keypoints[10][2] > 0.3:
            wrist_x, wrist_y = keypoints[10][0], keypoints[10][1]
            if self._is_hand_reaching_toward(wrist_x, wrist_y, box, shelf_center_x, shelf_center_y):
                score += 0.5
            hand_count += 1
        
        if hand_count == 0:
            return 0.5  # Neutral if no hands detected
        
        return score / hand_count if hand_count > 0 else 0.5
    
    def _rank_customers_by_pickup_likelihood(self, candidates, event_timestamp):
        """Rank customers by likelihood of picking up item."""
        if not candidates or self.shelf_zone_pixels is None:
            return []
        
        ranked = []
        shelf_center_x = (self.shelf_zone_pixels['x1'] + self.shelf_zone_pixels['x2']) / 2
        shelf_center_y = (self.shelf_zone_pixels['y1'] + self.shelf_zone_pixels['y2']) / 2
        
        for candidate in candidates:
            box = candidate['box']
            keypoints = candidate.get('keypoints')
            
            # Calculate proximity score (0.0-1.0)
            person_center_x = (box[0] + box[2]) / 2
            person_center_y = (box[1] + box[3]) / 2
            
            distance = np.sqrt(
                (person_center_x - shelf_center_x)**2 + 
                (person_center_y - shelf_center_y)**2
            )
            max_distance = np.sqrt(
                (self.shelf_zone_pixels['x2'] - self.shelf_zone_pixels['x1'])**2 +
                (self.shelf_zone_pixels['y2'] - self.shelf_zone_pixels['y1'])**2
            )
            proximity_score = 1.0 - min(distance / max_distance, 1.0) if max_distance > 0 else 0.5
            
            # Calculate hand position score (0.0-1.0)
            hand_score = self._calculate_hand_reaching_score(keypoints, box, shelf_center_x, shelf_center_y)
            
            # Combined score (weighted)
            combined_score = 0.6 * proximity_score + 0.4 * hand_score
            
            ranked.append({
                **candidate,
                'proximity_score': proximity_score,
                'hand_score': hand_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score (highest first)
        ranked.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ranked
    
    def _handle_weight_event(self, weight_change_g, timestamp):
        """Process weight change event and match with customers."""
        if weight_change_g >= 0:
            # Weight increased (item returned) - optional, skip for now
            return
        
        # Weight decreased (item picked up)
        # Find confirmed customers in shelf zone
        candidates = self._find_customers_in_shelf_zone()
        
        if not candidates:
            self._log_unmatched_event(weight_change_g, timestamp, "no_customer_in_zone")
            return
        
        # Rank candidates by proximity + hand position
        ranked = self._rank_customers_by_pickup_likelihood(candidates, timestamp)
        
        if not ranked:
            self._log_unmatched_event(weight_change_g, timestamp, "no_suitable_candidate")
            return
        
        # Ping closest/most likely customer
        best_customer = ranked[0]
        self._ping_customer_pickup(best_customer, weight_change_g, timestamp)
    
    def _log_unmatched_event(self, weight_change_g, timestamp, reason):
        """Log weight event that couldn't be matched to a customer."""
        event = {
            'type': 'unmatched_weight_event',
            'weight_change_g': weight_change_g,
            'timestamp': timestamp.isoformat(),
            'shelf_id': 'shelf-1',
            'reason': reason
        }
        
        self.events.append(event)
    
    def _ping_customer_pickup(self, customer_data, weight_change_g, timestamp):
        """Ping customer: Update shopping cart with picked up item."""
        track_id = customer_data['track_id']
        customer_id = customer_data['customer_id']
        
        if track_id not in self.customers:
            return
        
        customer = self.customers[track_id]
        
        # Validate weight change (must be negative for pickup)
        if weight_change_g >= 0:
            return
        
        weight_grams = abs(weight_change_g)
        
        # Rate limiting: Check if last ping was too recent (within 2 seconds)
        last_ping_time = customer.get('last_pickup_time')
        if last_ping_time is not None:
            # Handle both datetime object and ISO string
            if isinstance(last_ping_time, str):
                try:
                    from dateutil import parser
                    last_ping_time = parser.parse(last_ping_time)
                except:
                    # Fallback to datetime.fromisoformat
                    try:
                        last_ping_time = datetime.fromisoformat(last_ping_time.replace('Z', '+00:00'))
                    except:
                        last_ping_time = None
            
            if last_ping_time is not None:
                # Ensure timestamp is datetime object
                if isinstance(timestamp, str):
                    try:
                        from dateutil import parser
                        timestamp = parser.parse(timestamp)
                    except:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
            time_since_last = (timestamp - last_ping_time).total_seconds()
            if time_since_last < 2.0:
                return
        
        # Create item entry
        item_entry = {
            'weight_g': weight_grams,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'shelf_id': 'shelf-1',
            'confidence': customer_data.get('combined_score', 0.5)
        }
        
        # Update shopping cart (add to items_detected)
        if 'shopping_cart' not in customer:
            customer['shopping_cart'] = []
        
        # Check for duplicate items (same weight within 3 seconds)
        recent_items = [item for item in customer['shopping_cart'] 
                       if abs(item.get('weight_g', 0) - weight_grams) < 10]  # Within 10g
        if recent_items:
            # Check if any recent item is within 3 seconds
            for item in recent_items:
                item_time_str = item.get('timestamp', '')
                try:
                    if isinstance(item_time_str, str):
                        from dateutil import parser
                        item_time = parser.parse(item_time_str)
                    else:
                        item_time = item_time_str
                    
                    if isinstance(timestamp, str):
                        from dateutil import parser
                        event_time = parser.parse(timestamp)
                    else:
                        event_time = timestamp
                    
                    time_diff = abs((event_time - item_time).total_seconds())
                    if time_diff < 3.0:
                        return
                except Exception as e:
                    # If parsing fails, just add the item
                    pass
        
        customer['shopping_cart'].append(item_entry)
        
        # Update counters
        customer['pickup_count'] = customer.get('pickup_count', 0) + 1
        customer['last_pickup_time'] = timestamp if hasattr(timestamp, 'isoformat') else datetime.now()
        
        if 'items_detected' not in customer:
            customer['items_detected'] = set()
        customer['items_detected'].add(f"item_{len(customer['shopping_cart'])}")
        
        # Log event
        event = {
            'type': 'item_picked_up',
            'customer_id': customer_id,
            'track_id': int(track_id),
            'weight_change_g': weight_change_g,
            'item_weight_g': weight_grams,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'shelf_id': 'shelf-1',
            'confidence': customer_data.get('combined_score', 0.5),
            'proximity_score': customer_data.get('proximity_score', 0.0),
            'hand_score': customer_data.get('hand_score', 0.0)
        }
        
        self.events.append(event)
        
        print(f"   ✅ Pinged {customer_id} (Track {track_id})")
        print(f"      Item: {weight_grams}g")
        print(f"      Confidence: {customer_data.get('combined_score', 0.5):.0%}")
        print(f"      Shopping cart: {len(customer['shopping_cart'])} items")
        print(f"      Total weight: {sum(item.get('weight_g', 0) for item in customer['shopping_cart'])}g")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone editing - click and drag to adjust zones."""
        frame_h, frame_w = param['frame_shape'][:2]
        corner_size = 15  # Size of corner handles
        
        # Check which zone is clicked
        qr_zone = self.qr_zone_pixels
        shelf_zone = self.shelf_zone_pixels
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check QR zone
            if qr_zone:
                qx1, qy1 = qr_zone['x1'], qr_zone['y1']
                qx2, qy2 = qr_zone['x2'], qr_zone['y2']
                
                # Check corners
                if abs(x - qx1) < corner_size and abs(y - qy1) < corner_size:
                    self.dragging_zone = 'qr'
                    self.drag_corner = 'top-left'
                    self.drag_start = (x, y)
                    self.drag_zone_start = qr_zone.copy()
                elif abs(x - qx2) < corner_size and abs(y - qy1) < corner_size:
                    self.dragging_zone = 'qr'
                    self.drag_corner = 'top-right'
                    self.drag_start = (x, y)
                    self.drag_zone_start = qr_zone.copy()
                elif abs(x - qx1) < corner_size and abs(y - qy2) < corner_size:
                    self.dragging_zone = 'qr'
                    self.drag_corner = 'bottom-left'
                    self.drag_start = (x, y)
                    self.drag_zone_start = qr_zone.copy()
                elif abs(x - qx2) < corner_size and abs(y - qy2) < corner_size:
                    self.dragging_zone = 'qr'
                    self.drag_corner = 'bottom-right'
                    self.drag_start = (x, y)
                    self.drag_zone_start = qr_zone.copy()
                # Check if clicking inside QR zone (for moving)
                elif qx1 <= x <= qx2 and qy1 <= y <= qy2:
                    self.dragging_zone = 'qr'
                    self.drag_corner = 'move'
                    self.drag_start = (x, y)
                    self.drag_zone_start = qr_zone.copy()
            
            # Check Shelf zone
            if shelf_zone and self.dragging_zone is None:
                sx1, sy1 = shelf_zone['x1'], shelf_zone['y1']
                sx2, sy2 = shelf_zone['x2'], shelf_zone['y2']
                
                # Check corners
                if abs(x - sx1) < corner_size and abs(y - sy1) < corner_size:
                    self.dragging_zone = 'shelf'
                    self.drag_corner = 'top-left'
                    self.drag_start = (x, y)
                    self.drag_zone_start = shelf_zone.copy()
                elif abs(x - sx2) < corner_size and abs(y - sy1) < corner_size:
                    self.dragging_zone = 'shelf'
                    self.drag_corner = 'top-right'
                    self.drag_start = (x, y)
                    self.drag_zone_start = shelf_zone.copy()
                elif abs(x - sx1) < corner_size and abs(y - sy2) < corner_size:
                    self.dragging_zone = 'shelf'
                    self.drag_corner = 'bottom-left'
                    self.drag_start = (x, y)
                    self.drag_zone_start = shelf_zone.copy()
                elif abs(x - sx2) < corner_size and abs(y - sy2) < corner_size:
                    self.dragging_zone = 'shelf'
                    self.drag_corner = 'bottom-right'
                    self.drag_start = (x, y)
                    self.drag_zone_start = shelf_zone.copy()
                # Check if clicking inside shelf zone (for moving)
                elif sx1 <= x <= sx2 and sy1 <= y <= sy2:
                    self.dragging_zone = 'shelf'
                    self.drag_corner = 'move'
                    self.drag_start = (x, y)
                    self.drag_zone_start = shelf_zone.copy()
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_zone:
            if self.drag_corner == 'move':
                # Move entire zone
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                new_x1 = max(0, min(frame_w, self.drag_zone_start['x1'] + dx))
                new_y1 = max(0, min(frame_h, self.drag_zone_start['y1'] + dy))
                new_x2 = max(0, min(frame_w, self.drag_zone_start['x2'] + dx))
                new_y2 = max(0, min(frame_h, self.drag_zone_start['y2'] + dy))
                
                # Ensure minimum size
                if new_x2 - new_x1 < 50:
                    new_x2 = new_x1 + 50
                if new_y2 - new_y1 < 50:
                    new_y2 = new_y1 + 50
                
                if self.dragging_zone == 'qr':
                    self.qr_zone_pixels = {'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2}
                    self.qr_zone_percent = {
                        'x1_percent': new_x1 / frame_w,
                        'y1_percent': new_y1 / frame_h,
                        'x2_percent': new_x2 / frame_w,
                        'y2_percent': new_y2 / frame_h
                    }
                else:  # shelf
                    self.shelf_zone_pixels = {'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2}
                    self.shelf_zone_percent = {
                        'x1_percent': new_x1 / frame_w,
                        'y1_percent': new_y1 / frame_h,
                        'x2_percent': new_x2 / frame_w,
                        'y2_percent': new_y2 / frame_h
                    }
            else:
                # Resize corner
                if self.dragging_zone == 'qr':
                    current = self.qr_zone_pixels.copy()
                else:
                    current = self.shelf_zone_pixels.copy()
                
                if self.drag_corner == 'top-left':
                    new_x1 = max(0, min(x, current['x2'] - 50))
                    new_y1 = max(0, min(y, current['y2'] - 50))
                    if self.dragging_zone == 'qr':
                        self.qr_zone_pixels['x1'] = new_x1
                        self.qr_zone_pixels['y1'] = new_y1
                        self.qr_zone_percent['x1_percent'] = new_x1 / frame_w
                        self.qr_zone_percent['y1_percent'] = new_y1 / frame_h
                    else:
                        self.shelf_zone_pixels['x1'] = new_x1
                        self.shelf_zone_pixels['y1'] = new_y1
                        self.shelf_zone_percent['x1_percent'] = new_x1 / frame_w
                        self.shelf_zone_percent['y1_percent'] = new_y1 / frame_h
                elif self.drag_corner == 'top-right':
                    new_x2 = max(current['x1'] + 50, min(frame_w, x))
                    new_y1 = max(0, min(y, current['y2'] - 50))
                    if self.dragging_zone == 'qr':
                        self.qr_zone_pixels['x2'] = new_x2
                        self.qr_zone_pixels['y1'] = new_y1
                        self.qr_zone_percent['x2_percent'] = new_x2 / frame_w
                        self.qr_zone_percent['y1_percent'] = new_y1 / frame_h
                    else:
                        self.shelf_zone_pixels['x2'] = new_x2
                        self.shelf_zone_pixels['y1'] = new_y1
                        self.shelf_zone_percent['x2_percent'] = new_x2 / frame_w
                        self.shelf_zone_percent['y1_percent'] = new_y1 / frame_h
                elif self.drag_corner == 'bottom-left':
                    new_x1 = max(0, min(x, current['x2'] - 50))
                    new_y2 = max(current['y1'] + 50, min(frame_h, y))
                    if self.dragging_zone == 'qr':
                        self.qr_zone_pixels['x1'] = new_x1
                        self.qr_zone_pixels['y2'] = new_y2
                        self.qr_zone_percent['x1_percent'] = new_x1 / frame_w
                        self.qr_zone_percent['y2_percent'] = new_y2 / frame_h
                    else:
                        self.shelf_zone_pixels['x1'] = new_x1
                        self.shelf_zone_pixels['y2'] = new_y2
                        self.shelf_zone_percent['x1_percent'] = new_x1 / frame_w
                        self.shelf_zone_percent['y2_percent'] = new_y2 / frame_h
                elif self.drag_corner == 'bottom-right':
                    new_x2 = max(current['x1'] + 50, min(frame_w, x))
                    new_y2 = max(current['y1'] + 50, min(frame_h, y))
                    if self.dragging_zone == 'qr':
                        self.qr_zone_pixels['x2'] = new_x2
                        self.qr_zone_pixels['y2'] = new_y2
                        self.qr_zone_percent['x2_percent'] = new_x2 / frame_w
                        self.qr_zone_percent['y2_percent'] = new_y2 / frame_h
                    else:
                        self.shelf_zone_pixels['x2'] = new_x2
                        self.shelf_zone_pixels['y2'] = new_y2
                        self.shelf_zone_percent['x2_percent'] = new_x2 / frame_w
                        self.shelf_zone_percent['y2_percent'] = new_y2 / frame_h
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_zone = None
            self.drag_corner = None
            self.drag_start = None
            self.drag_zone_start = None
    
    def save_zone_config(self):
        """Save zone configuration to file."""
        import json
        import os
        config = {
            'qr_zone': self.qr_zone_percent,
            'shelf_zone': self.shelf_zone_percent
        }
        config_path = 'config/zone_config.json'
        os.makedirs('config', exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        # Zone configuration saved
    
    def load_zone_config(self):
        """Load zone configuration from file."""
        import json
        import os
        config_path = 'config/zone_config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'qr_zone' in config:
                        self.qr_zone_percent = config['qr_zone']
                    if 'shelf_zone' in config:
                        self.shelf_zone_percent = config['shelf_zone']
                pass  # Zone config loaded
            except Exception as e:
                pass  # Zone config load failed
    
    def confirm_pending_with_customer_id(self, customer_id, pending_id=None):
        """
        Confirm a PENDING track with customer_id from QR scan.
        
        Args:
            customer_id: Customer ID from QR code (e.g., "CUST_001")
            pending_id: Optional PENDING ID. If None, uses zone_active_pending.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        # Re-check zone status to get latest state (avoid race condition)
        zone_active, zone_pending_id, pending_count = self._check_qr_zone()
        
        # If pending_id not provided, use zone_active_pending (from latest check)
        if pending_id is None:
            pending_id = zone_pending_id if zone_active and pending_count == 1 else self.zone_active_pending
        
        if pending_id is None:
            return False, "No PENDING track in QR zone"
        
        # Find track_id from pending_id
        track_id = None
        for tid, pending in self.pending_tracks.items():
            if pending.get('pending_id') == pending_id:
                track_id = tid
                break
        
        if track_id is None:
            return False, f"PENDING track {pending_id} not found"
        
        # Confirm the track
        pending = self.pending_tracks[track_id]
        
        # Validate before confirming
        all_current_boxes = []
        for other_track_id, other_customer in self.customers.items():
            if other_track_id != track_id and other_customer.get('last_box') is not None:
                all_current_boxes.append(other_customer['last_box'])
        for other_track_id, other_pending in self.pending_tracks.items():
            if other_track_id != track_id and other_pending.get('box') is not None:
                all_current_boxes.append(other_pending['box'])
        
        is_valid, validation_score, issues = self._validate_pending_track(pending, all_current_boxes)
        
        if not is_valid:
            return False, f"Validation failed (score: {validation_score:.0%})"
        
        # Move from pending to confirmed
        customer_data = {
            'customer_id': customer_id,
            'first_seen': pending['first_seen'],
            'last_detection_time': datetime.now(),
            'feature_gallery': pending['feature_gallery'],
            'confidence_scores': pending['confidence_scores'],
            'last_box': pending.get('box'),
            'keypoints': pending.get('keypoints'),
            'frame_height': pending.get('frame_height'),
            # Shopping cart for weight-based pickup detection
            'shopping_cart': [],
            'pickup_count': 0,
            'last_pickup_time': None,
            'items_detected': set()
        }
        
        self.customers[track_id] = customer_data
        del self.pending_tracks[track_id]
        
        # Reset zone
        self.zone_active_pending = None
        
        # Log event
        self.events.append({
            'type': 'confirmed',
            'customer_id': customer_id,
            'track_id': int(track_id),
            'pending_id': pending_id,
            'timestamp': datetime.now().isoformat(),
            'validation_score': validation_score
        })
        
        return True, f"Confirmed {customer_id}"
    
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
                # Compute average feature for gallery
                avg_feat = None
                if customer.get('feature_gallery'):
                    avg_feat = np.mean(customer['feature_gallery'], axis=0)
                # Get keypoints and frame height from last detection for legs visibility check
                last_keypoints = None
                last_frame_height = None
                # Try to get from customer data if stored
                if 'last_keypoints' in customer:
                    last_keypoints = customer['last_keypoints']
                if 'last_frame_height' in customer:
                    last_frame_height = customer['last_frame_height']
                
                self.lost_tracks[track_id] = {
                    'lost_time': current_time,
                    'data': customer.copy(),
                    'last_box': customer.get('last_box', customer.get('entry_box')),
                    'features': avg_feat,
                    'keypoints': last_keypoints,  # Store for legs visibility check
                    'frame_height': last_frame_height  # Store for legs visibility check
                }
        
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
        # Clean up holding detector state
        if self.holding_detector:
            self.holding_detector.reset_customer(customer_id)
        
        self.events.append({
            'event': 'EXIT',
            'customer_id': customer_id,
            'track_id': track_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': float(duration),
            'suspicious_count': int(self.customers.get(track_id, {}).get('suspicious_count', 0))
        })
        # Customer exit logged

    def _iou(self, box1, box2):
        if box1 is None or box2 is None:
            return 0.0
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter + 1e-8
        return inter / union

    def _check_upper_body_visible(self, pending):
        """
        Check if upper body (head/torso) is visible.
        Only needs 1 keypoint: nose OR 1 shoulder (enough to get clothing color).
        
        Returns:
            bool: True if at least 1 upper body keypoint is visible, False otherwise
        """
        box = pending.get('box')
        keypoints = pending.get('keypoints')
        
        if box is None:
            return False
        
        if keypoints is None or len(keypoints) < 17:
            return False
        
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        
        # COCO pose keypoints for upper body:
        # 0: nose
        # 5: left_shoulder, 6: right_shoulder
        
        # Check nose (head) - enough to get head/upper clothing color
        nose = keypoints[0] if len(keypoints) > 0 else None
        if nose is not None and len(nose) >= 3 and nose[2] > 0.3:
            nose_y = nose[1]
            # Nose should be in upper 40% of box
            if y1 <= nose_y <= y1 + box_height * 0.4:
                return True
        
        # Check shoulders (torso) - enough to get shirt color
        left_shoulder = keypoints[5] if len(keypoints) > 5 else None
        right_shoulder = keypoints[6] if len(keypoints) > 6 else None
        
        if left_shoulder is not None and len(left_shoulder) >= 3 and left_shoulder[2] > 0.3:
            shoulder_y = left_shoulder[1]
            # Shoulder should be in upper 60% of box
            if y1 <= shoulder_y <= y1 + box_height * 0.6:
                return True
        
        if right_shoulder is not None and len(right_shoulder) >= 3 and right_shoulder[2] > 0.3:
            shoulder_y = right_shoulder[1]
            if y1 <= shoulder_y <= y1 + box_height * 0.6:
                return True
        
        # No upper body keypoints found
        return False
    
    def _check_legs_visible(self, pending):
        """
        Check if legs are visible in the bounding box.
        REQUIRES: At least 1 ankle keypoint (orange in COCO pose) must be visible.
        This is important to identify pants color for ReID.
        
        Returns:
            bool: True if at least 1 ankle keypoint is visible, False otherwise
        """
        box = pending.get('box')
        keypoints = pending.get('keypoints')
        frame_height = pending.get('frame_height')
        
        if box is None:
            return False
        
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        box_bottom = y2
        box_width = x2 - x1
        
        # REQUIRED: Check using keypoints (ankle keypoints) - MUST HAVE AT LEAST 1
        # COCO pose keypoints: 15=left_ankle, 16=right_ankle (orange color in visualization)
        if keypoints is not None and len(keypoints) >= 17:
            left_ankle = keypoints[15] if len(keypoints) > 15 else None
            right_ankle = keypoints[16] if len(keypoints) > 16 else None
            
            # Check if ankle keypoints are visible and within box
            ankles_visible = 0
            if left_ankle is not None and len(left_ankle) >= 3:
                ankle_y = left_ankle[1]
                ankle_x = left_ankle[0]
                ankle_conf = left_ankle[2]
                
                # Lower confidence threshold for ankle (0.2 instead of 0.3)
                if ankle_conf > 0.2:
                    # Ankle should be in lower 30% of box and within box bounds
                    # Also allow if ankle is just below box (within 20% of box height)
                    lower_bound = y1 + box_height * 0.7
                    upper_bound = y2 + box_height * 0.2  # Allow slightly below box
                    
                    if lower_bound <= ankle_y <= upper_bound and x1 - box_width * 0.2 <= ankle_x <= x2 + box_width * 0.2:
                        ankles_visible += 1
            
            if right_ankle is not None and len(right_ankle) >= 3:
                ankle_y = right_ankle[1]
                ankle_x = right_ankle[0]
                ankle_conf = right_ankle[2]
                
                # Lower confidence threshold for ankle (0.2 instead of 0.3)
                if ankle_conf > 0.2:
                    lower_bound = y1 + box_height * 0.7
                    upper_bound = y2 + box_height * 0.2  # Allow slightly below box
                    
                    if lower_bound <= ankle_y <= upper_bound and x1 - box_width * 0.2 <= ankle_x <= x2 + box_width * 0.2:
                        ankles_visible += 1
            
            # REQUIRED: At least 1 ankle keypoint must be visible
            if ankles_visible >= 1:
                return True
            else:
                return False
        
        # If no keypoints, cannot verify legs
        return False
    
    def _check_relatives_nearby(self, current_box, lost_box, all_current_boxes, 
                               box_center, box_width, box_height):
        """
        Check if there are relatives (other people) nearby in the same area.
        Checks for people above, below, and to the left.
        
        Returns:
            bool: True if relatives detected nearby
        """
        if all_current_boxes is None or len(all_current_boxes) == 0:
            return False
        
        # Define proximity zones
        # Above: within 1.5x box height above
        # Below: within 1.5x box height below
        # Left: within 1.5x box width to the left
        proximity_threshold_h = box_height * 1.5
        proximity_threshold_w = box_width * 1.5
        
        # Get lost box center
        lost_center = ((lost_box[0] + lost_box[2]) / 2, (lost_box[1] + lost_box[3]) / 2)
        
        relatives_count = 0
        for other_box in all_current_boxes:
            if other_box is None:
                continue
            
            other_center = ((other_box[0] + other_box[2]) / 2, 
                          (other_box[1] + other_box[3]) / 2)
            
            # Calculate distances
            dx = other_center[0] - box_center[0]
            dy = other_center[1] - box_center[1]
            
            # Check if in proximity zones (above, below, or left)
            is_above = dy < 0 and abs(dy) < proximity_threshold_h and abs(dx) < box_width
            is_below = dy > 0 and abs(dy) < proximity_threshold_h and abs(dx) < box_width
            is_left = dx < 0 and abs(dx) < proximity_threshold_w and abs(dy) < box_height
            
            if is_above or is_below or is_left:
                relatives_count += 1
        
        # If 2+ relatives nearby, consider it a family/group
        return relatives_count >= 1
    
    def _try_reid(self, track_id, box, features, all_current_boxes=None, keypoints=None, frame_height=None):
        """
        Try to re-identify a lost track using appearance + IoU.
        Also checks for relatives (people nearby) and legs visibility to avoid confusion.
        """
        if features is None:
            return None
        best = None
        now = datetime.now()
        
        # Get current box center for spatial checks
        box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        # Check if current detection has upper body and legs visible - HARD REQUIREMENT
        current_pending = {
            'box': box,
            'keypoints': keypoints,
            'frame_height': frame_height
        }
        current_upper_visible = self._check_upper_body_visible(current_pending)
        current_legs_visible = self._check_legs_visible(current_pending)
        if not current_upper_visible:
            return None  # Cannot re-identify without seeing upper body
        if not current_legs_visible:
            return None  # Cannot re-identify without seeing ankle keypoint
        
        for lost_id, data in list(self.lost_tracks.items()):
            # Time gating
            lost_time = data['lost_time']
            if (now - lost_time).total_seconds() > self.max_lost_time:
                continue
            
            # IoU gating
            lost_box = data.get('last_box')
            if lost_box is None:
                continue
            iou = self._iou(box, lost_box)
            if iou < 0.1:  # allow low IoU but still gate a bit
                continue
            
            # Check if lost track had upper body and legs visible - HARD REQUIREMENT
            lost_pending = {
                'box': lost_box,
                'keypoints': data.get('keypoints'),
                'frame_height': data.get('frame_height')
            }
            lost_upper_visible = self._check_upper_body_visible(lost_pending)
            lost_legs_visible = self._check_legs_visible(lost_pending)
            if not lost_upper_visible:
                continue  # Skip if lost track didn't have upper body
            if not lost_legs_visible:
                continue  # Skip if lost track didn't have ankle keypoint
            
            # Check for relatives (people nearby) - NEW
            # Check if there are other people in the same area (above, below, left)
            has_relatives = self._check_relatives_nearby(
                box, lost_box, all_current_boxes, 
                box_center, box_width, box_height
            )
            
            if has_relatives:
                # If relatives nearby, require higher similarity to avoid confusion
                min_sim_thresh = self.reid_high_thresh + 0.1  # Stricter
                # Relatives detected - require higher similarity
            else:
                min_sim_thresh = self.reid_high_thresh
            
            # Similarity
            sim = LightweightReID.similarity(features, data.get('features'))
            
            # Two-stage matching (with relative check)
            if sim >= min_sim_thresh or (sim >= self.reid_low_thresh and iou >= 0.2 and not has_relatives):
                if best is None or sim > best['sim']:
                    best = {
                        'lost_id': lost_id,
                        'sim': sim,
                        'iou': iou,
                        'data': data['data'],
                        'features': data.get('features'),
                        'has_relatives': has_relatives
                    }
        
        if best is None:
            return None
        # Reuse customer data with new track_id
        customer = best['data']
        customer['track_id'] = track_id
        customer['last_box'] = box
        customer['last_detection_time'] = now
        # Update keypoints and frame height for legs visibility check
        customer['last_keypoints'] = keypoints
        customer['last_frame_height'] = frame_height
        # merge galleries
        gallery = deque(maxlen=self.feature_gallery_size)
        if customer.get('feature_gallery'):
            for f in customer['feature_gallery']:
                gallery.append(f)
        if best.get('features') is not None:
            gallery.append(best['features'])
        if features is not None:
            gallery.append(features)
        customer['feature_gallery'] = gallery
        # clean old entries
        if best['lost_id'] in self.customers:
            del self.customers[best['lost_id']]
        del self.lost_tracks[best['lost_id']]
        return customer
    
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
    
    def _draw_pending_tracks(self, frame, result):
        """Draw pending tracks with orange color and validation status."""
        if not self.pending_tracks:
            return frame
        
        # Get boxes and track_ids from result
        if result.boxes is None or result.boxes.id is None:
            return frame
        
        track_ids = result.boxes.id.int().cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for track_id, box in zip(track_ids, boxes):
            if track_id in self.pending_tracks:
                pending = self.pending_tracks[track_id]
                x1, y1, x2, y2 = map(int, box)
                
                # Get all current boxes for relative checking
                all_current_boxes = []
                for other_track_id, other_customer in self.customers.items():
                    if other_track_id != track_id and other_customer.get('last_box') is not None:
                        all_current_boxes.append(other_customer['last_box'])
                for other_track_id, other_pending in self.pending_tracks.items():
                    if other_track_id != track_id and other_pending.get('box') is not None:
                        all_current_boxes.append(other_pending['box'])
                
                # Validate track (with relative checking)
                is_valid, validation_score, issues = self._validate_pending_track(pending, all_current_boxes)
                
                # Color based on validation status
                if is_valid:
                    color = (0, 255, 0)  # Green = ready to confirm
                    box_thickness = 3
                else:
                    color = (0, 165, 255)  # Orange = collecting info
                    box_thickness = 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                
                # Draw pending ID
                label = f"{pending['pending_id']}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw validation status
                feature_count = len(pending['feature_gallery'])
                progress_text = f"Samples: {feature_count}/{self.min_samples_required}"
                
                if is_valid:
                    status_text = f"READY (Press 'c') | {validation_score:.0%}"
                    status_color = (0, 255, 0)
                else:
                    status_text = f"Collecting... | {validation_score:.0%}"
                    status_color = (0, 165, 255)
                
                # Draw progress bar
                bar_width = int((x2 - x1) * 0.8)
                bar_x = x1 + 10
                bar_y = y2 + 10
                bar_height = 15
                
                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                
                # Progress fill
                fill_width = int(bar_width * validation_score)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), status_color, -1)
                
                # Border
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
                
                # Draw status text
                cv2.putText(frame, progress_text, (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.putText(frame, status_text, (bar_x, bar_y + bar_height + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        return frame
    
    def _draw_holding_status(self, frame):
        """Draw holding status for confirmed customers with clear visual indicators and detailed info."""
        for track_id, customer in self.customers.items():
            holding_status = customer.get('holding_status', {})
            
            # Skip if no holding status
            if not holding_status:
                continue
            
            # Get customer position
            last_box = customer.get('last_box')
            if last_box is None:
                continue
            
            x1, y1, x2, y2 = map(int, last_box)
            status = holding_status.get('status', 'transitioning')
            is_holding = holding_status.get('is_holding', False)
            conf = holding_status.get('confidence', 0.0)
            method = holding_status.get('method', 'unknown')
            hand_used = holding_status.get('hand_used', 'unknown')
            
            # Determine display based on status
            if status == 'confirmed_holding':
                # CONFIRMED HOLDING - Green, large text
                bg_color = (0, 200, 0)  # Green background
                text_color = (255, 255, 255)  # White text
                border_color = (0, 255, 0)  # Bright green border
                status_text = "✅ CONFIRMED HOLDING"
                conf_text = f"Score: {conf:.2f}"
                method_text = f"Method: {method}"
                hand_text = f"Hand: {hand_used}"
                font_scale = 0.7
                thickness = 2
                icon = "🤚"
                
            elif status == 'confirmed_not_holding':
                # CONFIRMED NOT HOLDING - Gray, smaller text
                bg_color = (100, 100, 100)  # Gray background
                text_color = (255, 255, 255)  # White text
                border_color = (150, 150, 150)  # Light gray border
                status_text = "❌ NOT HOLDING"
                conf_text = f"Score: {conf:.2f}"
                method_text = f"Method: {method}"
                hand_text = ""
                font_scale = 0.5
                thickness = 1
                icon = "👐"
                
            else:  # transitioning
                # TRANSITIONING - Yellow/Orange, medium text
                bg_color = (0, 165, 255)  # Orange background
                text_color = (255, 255, 255)  # White text
                border_color = (0, 200, 255)  # Bright orange border
                status_text = "⏳ CHECKING..."
                conf_text = f"Score: {conf:.2f}"
                method_text = f"Method: {method}"
                hand_text = ""
                font_scale = 0.6
                thickness = 2
                icon = "🔍"
            
            # Calculate text positions
            full_text = f"{icon} {status_text}"
            text_size, _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, thickness)
            
            # Calculate method and hand text sizes safely
            if method_text:
                method_size, _ = cv2.getTextSize(method_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, 1)
            else:
                method_size = (0, 0)
            
            if hand_text:
                hand_size, _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, 1)
            else:
                hand_size = (0, 0)
            
            # Calculate total width and height
            total_width = max(text_size[0], conf_size[0], method_size[0], hand_size[0]) + 20
            total_height = text_size[1] + conf_size[1] + (method_size[1] if method_text else 0) + (hand_size[1] if hand_text else 0) + 20
            
            # Position: Top-right of bounding box
            text_x = x2 - total_width + 10
            text_y = y1 - 5
            
            # Draw background rectangle with rounded corners effect
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - total_height), 
                         (x2, text_y + 5), 
                         bg_color, -1)
            
            # Draw border
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - total_height), 
                         (x2, text_y + 5), 
                         border_color, 2)
            
            # Draw status text (main)
            y_offset = text_y - 5
            cv2.putText(frame, full_text, 
                       (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
            # Draw confidence text
            y_offset -= (conf_size[1] + 5)
            cv2.putText(frame, conf_text, 
                       (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, text_color, thickness - 1)
            
            # Draw method text (if available)
            if method_text:
                y_offset -= (method_size[1] + 3)
                cv2.putText(frame, method_text, 
                           (text_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, text_color, 1)
            
            # Draw hand used text (if available and holding)
            if hand_text and status == 'confirmed_holding':
                y_offset -= (hand_size[1] + 3)
                cv2.putText(frame, hand_text, 
                           (text_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, text_color, 1)
            
            # Draw indicator circle on bounding box corner (top-left)
            indicator_pos = (x1 + 15, y1 + 15)
            cv2.circle(frame, indicator_pos, 10, bg_color, -1)
            cv2.circle(frame, indicator_pos, 10, border_color, 2)
            
            # Draw small icon in circle (if holding)
            if status == 'confirmed_holding':
                # Draw a small hand icon (circle with dot)
                cv2.circle(frame, indicator_pos, 4, text_color, -1)
        
        return frame
    
    def get_stats(self):
        """Get current tracking statistics."""
        stats = {
            'active_customers': len(self.customers),
            'occluded_tracks': len(self.lost_tracks),
            'pending_tracks': len(self.pending_tracks),
            'total_customers': self.next_customer_id - 1,
            'total_events': len(self.events)
        }
        
        # Save to shared file for other processes
        if self.stats_manager:
            # Calculate items_taken and avg_time for dashboard
            items_taken = sum(len(c.get('shopping_cart', [])) for c in self.customers.values())
            if self.customers:
                total_duration = sum(
                    (datetime.now() - (c.get('entry_time') or datetime.now())).total_seconds()
                    for c in self.customers.values()
                )
                avg_duration = total_duration / len(self.customers)
                avg_minutes = int(avg_duration / 60)
                avg_seconds = int(avg_duration % 60)
                avg_time = f"{avg_minutes}m {avg_seconds}s" if avg_minutes > 0 else f"{avg_seconds}s"
            else:
                avg_time = '0m'
            
            dashboard_stats = {
                'total_customers': stats['total_customers'],
                'active_customers': stats['active_customers'],
                'items_taken': items_taken,
                'avg_time': avg_time,
                'total_events': stats['total_events']
            }
            self.stats_manager.save_stats(dashboard_stats)
            
            # Save customers data
            customers_data = {}
            for track_id, customer in self.customers.items():
                entry_time = customer.get('entry_time') or customer.get('first_seen')
                last_detection_time = customer.get('last_detection_time') or customer.get('last_seen')
                
                customers_data[f"customer_{track_id}"] = {
                    'track_id': track_id,
                    'customer_id': customer.get('customer_id', 'UNKNOWN'),
                    'confirmed': customer.get('state') and (hasattr(customer.get('state'), 'name') and customer.get('state').name == 'CONFIRMED' or str(customer.get('state')) == 'CONFIRMED'),
                    'first_seen': entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time) if entry_time else None,
                    'last_seen': last_detection_time.isoformat() if hasattr(last_detection_time, 'isoformat') else str(last_detection_time) if last_detection_time else None,
                    'shopping_cart': customer.get('shopping_cart', []),
                    'pickup_count': customer.get('pickup_count', 0)
                }
            self.stats_manager.save_customers_data(customers_data)
            
            # Save MQTT events
            mqtt_events = [e for e in self.events if e.get('type') in ['item_picked_up', 'unmatched_weight_event']]
            self.stats_manager.save_mqtt_events(mqtt_events)
        
        return stats
    
    def save_events(self, filename='data/logs/tracking_events.json'):
        """Save all tracking events to JSON."""
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.events, f, indent=2, default=str)
        # Events saved


def main():
    """
    Main tracking loop with visualization.
    """
    # Initialize tracker
    tracker = RetailCustomerTracker(
        detection_model='models/yolo11n-pose.pt',  # Use pose model for keypoints
        tracker_config='config/botsort_reid.yaml'
    )
    
    # Load zone configuration if exists
    tracker.load_zone_config()
    
    # Initialize MQTT for weight-based pickup detection
    tracker._init_mqtt()
    
    # Start web server for QR scanner (in background thread)
    try:
        from src.web.server import run_server
        import threading
        import time
        server_thread = threading.Thread(
            target=run_server,
            args=(tracker, '0.0.0.0', 8080, False),
            daemon=True
        )
        server_thread.start()
        time.sleep(1)  # Give server time to start
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    # Open webcam with timeout
    import time
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        # Set timeout for camera initialization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Try to read a frame to verify camera works
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            cap = None
    except Exception as e:
        if cap:
            cap.release()
        cap = None
    
    if cap is None or not cap.isOpened():
        # Keep running for dashboard/MQTT only
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return
    
    frame_count = 0
    fps_list = []
    
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
            y_pos = 30
            cv2.putText(annotated_frame, 
                       f"Active: {stats['active_customers']} | Pending: {stats['pending_tracks']} | Occluded: {stats['occluded_tracks']} | Total: {stats['total_customers']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # FPS
            y_pos += 30
            fps = 1 / (time.time() - start_time)
            fps_list.append(fps)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Pending Tracks Panel
            if tracker.pending_tracks:
                y_pos += 40
                cv2.putText(annotated_frame, "PENDING TRACKS:", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                for idx, (track_id, pending) in enumerate(list(tracker.pending_tracks.items())[:9]):
                    y_pos += 25
                    prefix = ">" if idx == tracker.selected_pending_index else " "
                    age = (datetime.now() - pending['first_seen']).total_seconds()
                    
                    # Get all current boxes for relative checking
                    all_current_boxes = []
                    for other_track_id, other_customer in tracker.customers.items():
                        if other_track_id != track_id and other_customer.get('last_box') is not None:
                            all_current_boxes.append(other_customer['last_box'])
                    for other_track_id, other_pending in tracker.pending_tracks.items():
                        if other_track_id != track_id and other_pending.get('box') is not None:
                            all_current_boxes.append(other_pending['box'])
                    
                    # Get validation status (with relative checking)
                    is_valid, validation_score, _ = tracker._validate_pending_track(pending, all_current_boxes)
                    status_icon = "✓" if is_valid else "⏳"
                    
                    text = f"{prefix} {idx+1}. {pending['pending_id']} ({age:.1f}s) {status_icon}{validation_score:.0%}"
                    
                    # Color coding
                    if idx == tracker.selected_pending_index:
                        color = (255, 255, 0)  # Yellow = selected
                    elif is_valid:
                        color = (0, 255, 0)    # Green = ready
                    else:
                        color = (0, 165, 255)  # Orange = collecting
                    
                    cv2.putText(annotated_frame, text, 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Show
            cv2.imshow('Retail Tracking - BoT-SORT + ReID', annotated_frame)
            
            # Set mouse callback for zone editing (update on each frame to get current frame shape)
            cv2.setMouseCallback('Retail Tracking - BoT-SORT + ReID', 
                                tracker._mouse_callback, 
                                {'frame_shape': annotated_frame.shape})
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Confirm selected pending track
                tracker.confirm_pending_track()
            elif key in [ord(str(i)) for i in range(1, 10)]:
                # Select pending track 1-9
                num = int(chr(key))
                tracker.select_pending_track(num)
            elif key == ord('s'):
                tracker.save_events()
            elif key == ord('w'):
                tracker.save_zone_config()
            elif key == ord('i'):
                # Info key (no print)
                pass
            
            frame_count += 1
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # Finalize all customers
        current_time = datetime.now()
        for track_id, customer in list(tracker.customers.items()):
            duration = (current_time - customer['entry_time']).total_seconds()
            tracker._finalize_customer(track_id, customer['customer_id'], duration)
        
        # Save logs
        tracker.save_events()
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
