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
    print("⚠️  Warning: HoldingDetector not available (mediapipe may be missing). Holding detection disabled.")
    class HoldingDetector:
        def __init__(self):
            pass
        def reset_customer(self, customer_id):
            pass


class TrackState(Enum):
    """Track states for manual confirmation system."""
    PENDING = "PENDING"      # Waiting for manual confirmation
    CONFIRMED = "CONFIRMED"  # Manually confirmed by user


class LightweightReID:
    """Lightweight ReID: LAB color + HOG + texture + edge density."""

    def __init__(self):
        self.feature_dim = 512

    def extract_features(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, (128, 256))
            h = crop.shape[0]
            head = crop[: int(h * 0.3), :]
            torso = crop[int(h * 0.3) : int(h * 0.7), :]
            legs = crop[int(h * 0.7) :, :]
            feats = []
            for region in (head, torso, legs):
                feats.append(self._lab(region))
            for region in (head, torso, legs):
                feats.append(self._hog(region))
            for region in (head, torso, legs):
                feats.append(self._texture(region))
            for region in (head, torso, legs):
                feats.append(self._edge_density(region))
            features = np.concatenate(feats)
            if len(features) > self.feature_dim:
                features = features[: self.feature_dim]
            else:
                features = np.pad(features, (0, self.feature_dim - len(features)), "constant")
            features = features / (np.linalg.norm(features) + 1e-8)
            return features
        except Exception:
            return None

    def _lab(self, img):
        if img.size == 0:
            return np.zeros(64)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab], [0], None, [32], [0, 100])
        hist_a = cv2.calcHist([lab], [1], None, [16], [0, 255])
        hist_b = cv2.calcHist([lab], [2], None, [16], [0, 255])
        hist_l = cv2.normalize(hist_l, hist_l).flatten()
        hist_a = cv2.normalize(hist_a, hist_a).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        return np.concatenate([hist_l, hist_a, hist_b])

    def _hog(self, img):
        if img.size == 0:
            return np.zeros(64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            gray = cv2.resize(gray, (16, 16))
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        direction = ((direction + 180) % 360).astype(np.uint8)
        h, w = magnitude.shape
        cell = 8
        n_x, n_y = w // cell, h // cell
        hist = np.zeros(64)
        for i in range(0, min(n_y * cell, h), cell):
            for j in range(0, min(n_x * cell, w), cell):
                cell_mag = magnitude[i : i + cell, j : j + cell]
                cell_dir = direction[i : i + cell, j : j + cell]
                for mag, d in zip(cell_mag.flatten(), cell_dir.flatten()):
                    bin_idx = int(d / 360 * 64) % 64
                    hist[bin_idx] += mag
        hist = hist / (np.linalg.norm(hist) + 1e-8)
        return hist

    def _texture(self, img):
        if img.size == 0:
            return np.zeros(32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            gray = cv2.resize(gray, (16, 16))
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        hist = cv2.calcHist([local_var.astype(np.uint8)], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _edge_density(self, img):
        if img.size == 0:
            return np.zeros(16)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = max(1, h // grid_h), max(1, w // grid_w)
        densities = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = edges[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                density = np.sum(cell > 0) / (cell_h * cell_w)
                densities.append(density)
        densities = np.array(densities[:16])
        if len(densities) < 16:
            densities = np.pad(densities, (0, 16 - len(densities)), "constant")
        return densities

    @staticmethod
    def similarity(f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        try:
            dp = np.dot(f1, f2)
            return float(dp / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))
        except Exception:
            return 0.0


class RetailCustomerTracker:
    """
    Production-ready retail customer tracking system.
    Uses BoT-SORT with ReID for robust tracking across occlusions.
    """
    
    def __init__(self, 
                 detection_model='yolo11n-pose.pt',  # Changed to pose model
                 tracker_config='config/botsort_reid.yaml',
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
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Shelf Zone Configuration (where items are placed)
        # Default: Left side, middle-bottom area (adjust based on camera view)
        self.shelf_zone_percent = {
            'x1_percent': 0.0,    # Left edge
            'y1_percent': 0.3,    # 30% from top
            'x2_percent': 0.5,    # 50% from left (middle of frame)
            'y2_percent': 0.9     # 90% from top (near bottom)
        }
        self.shelf_zone_pixels = None  # Will be calculated from frame size
        
        # Weight Event Handling
        self.recent_weight_events = deque(maxlen=10)  # Last 10 events
        self.weight_event_timeout = 3.0  # Match events within 3 seconds
        
        # Logs
        self.events = []
        
        print(f"✅ Tracker ready | Device: {device}")
        print(f"   Model: {detection_model}")
        print(f"   Config: {tracker_config}")
        print(f"   Holding detection: Hand Region Analysis (Edge + YOLO + Texture)")
        print(f"   QR Zone: Right side, full height (70-100% width, 0-100% height)")
        print(f"   Shelf Zone: Left-middle area (0-50% width, 30-90% height)")
    
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
            
            # DEBUG: Check if keypoints are available
            if keypoints is None:
                print(f"⚠️  DEBUG | YOLO result has NO keypoints! Model might not be pose model.")
            else:
                print(f"✓ DEBUG | Keypoints available: shape={keypoints.shape}")
            
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
                print(f"🔄 ReID | {customer['customer_id']} (New Track {track_id})")
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
                print(f"⏳ Pending | {pending_id} (Track {track_id}) - Collecting info...")
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
                # Just reached minimum - notify user
                print(f"✓ Ready | {pending['pending_id']} - Can confirm now (Press 'c')")
            
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
            print(f"   ❌ Upper body not visible for {pending.get('pending_id', 'track')} - CANNOT CONFIRM without seeing upper body")
            # HARD REQUIREMENT: If upper body not visible, validation fails immediately
            validation_score = 0.0
            is_valid = False
            return is_valid, validation_score, issues
        else:
            scores.append(1.0)
            print(f"   ✅ Upper body visible for {pending.get('pending_id', 'track')}")
        
        # 6. Check if legs are visible - HARD REQUIREMENT (must see at least 1 ankle keypoint - orange)
        legs_visible = self._check_legs_visible(pending)
        if not legs_visible:
            issues.append("❌ CRITICAL: Legs not visible - need at least 1 ankle keypoint (orange) to identify pants color")
            scores.append(0.0)  # Critical: must see at least 1 ankle keypoint
            print(f"   ❌ Legs not visible for {pending.get('pending_id', 'track')} - CANNOT CONFIRM without seeing ankle keypoint")
            # HARD REQUIREMENT: If legs not visible, validation fails immediately
            validation_score = 0.0
            is_valid = False
            return is_valid, validation_score, issues
        else:
            scores.append(1.0)
            print(f"   ✅ Legs visible for {pending.get('pending_id', 'track')} - ankle keypoint detected")
        
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
                    print(f"   ⚠️  Relatives detected near {pending.get('pending_id', 'track')}, requiring extra validation")
        
        # Overall validation score
        validation_score = np.mean(scores) if scores else 0.0
        
        # Validation passes if:
        # 1. Upper body and legs are visible (already checked above - hard requirements)
        # 2. Overall score >= 0.8
        # Note: issues from other checks (samples, quality, etc.) are warnings but not blockers
        #       if upper body and legs are visible
        is_valid = validation_score >= 0.8
        
        # Debug output
        if is_valid:
            print(f"   ✅ Validation PASSED: score={validation_score:.1%} (upper body + ankle visible)")
        else:
            print(f"   ❌ Validation FAILED: score={validation_score:.1%} < 0.8")
            if issues:
                print(f"      Issues: {', '.join(issues[:3])}")  # Show first 3 issues
        
        return is_valid, validation_score, issues
    
    def confirm_pending_track(self, track_id=None):
        """Confirm a pending track to create a customer ID."""
        if track_id is None:
            # Auto-select first pending
            if not self.pending_tracks:
                print("⚠️  No pending tracks to confirm")
                return
            track_id = list(self.pending_tracks.keys())[self.selected_pending_index % len(self.pending_tracks)]
        
        if track_id not in self.pending_tracks:
            print(f"⚠️  Track {track_id} not in pending")
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
            print(f"❌ Cannot confirm {pending['pending_id']} - Insufficient information:")
            for issue in issues:
                print(f"   • {issue}")
            print(f"   Validation score: {validation_score:.1%} (need ≥80%)")
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
        print(f"✅ Confirmed | {customer_id} (Track {track_id})")
        print(f"   Validation: {validation_score:.0%} | Samples: {feature_count} | Conf: {avg_conf:.2f}")
    
    def select_pending_track(self, index):
        """Select a pending track by index (1-9)."""
        if not self.pending_tracks:
            return
        self.selected_pending_index = index - 1
        track_ids = list(self.pending_tracks.keys())
        if 0 <= self.selected_pending_index < len(track_ids):
            track_id = track_ids[self.selected_pending_index]
            pending = self.pending_tracks[track_id]
            print(f"👉 Selected: {pending['pending_id']}")
    
    def _cleanup_pending_tracks(self):
        """Remove old pending tracks that timeout."""
        current_time = datetime.now()
        to_remove = []
        for track_id, pending in self.pending_tracks.items():
            age = (current_time - pending['first_seen']).total_seconds()
            if age > self.pending_timeout:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            pending = self.pending_tracks[track_id]
            print(f"⏱️  Timeout | {pending['pending_id']} removed (no confirmation)")
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
        if zone_active and pending_count == 1:
            color = (0, 255, 0)  # Green = active, ready to scan
            thickness = 3
        else:
            color = (0, 0, 255)  # Red = inactive, waiting
            thickness = 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if zone_active and pending_count == 1:
            label = f"QR ZONE ✅ ACTIVE - {pending_id}"
        elif pending_count > 1:
            label = f"QR ZONE ⚠️ MULTIPLE ({pending_count})"
        else:
            label = "QR ZONE ⏸️ WAITING"
        
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
        
        # Draw rectangle (yellow/cyan for shelf zone)
        color = (0, 255, 255)  # Cyan
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = "SHELF ZONE"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0] + 10, label_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 5, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def _init_mqtt(self):
        """Initialize MQTT client and subscribe to weight events."""
        try:
            import paho.mqtt.client as mqtt
            import uuid
            import socket
            import time
            
            # Check if paho-mqtt is installed
            try:
                import paho.mqtt.client as mqtt
            except ImportError:
                print("⚠️  paho-mqtt not installed. Install with: pip install paho-mqtt")
                print("   Weight-based pickup detection will be disabled")
                self.mqtt_connected = False
                return
            
            # Test network connectivity first
            print(f"🔍 Testing connection to {self.mqtt_broker}:1883...")
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(5)  # 5 second timeout
                result = test_socket.connect_ex((self.mqtt_broker, 1883))
                test_socket.close()
                
                if result != 0:
                    print(f"⚠️  Cannot reach {self.mqtt_broker}:1883 (connection refused or timeout)")
                    print(f"   This might be due to:")
                    print(f"   - Firewall blocking port 1883")
                    print(f"   - Network connectivity issues")
                    print(f"   - Broker is down")
                    print(f"   💡 Tip: Try using a local MQTT broker or check your network")
                    self.mqtt_connected = False
                    return
            except Exception as e:
                print(f"⚠️  Network test failed: {e}")
                print(f"   Cannot verify connectivity to {self.mqtt_broker}")
                # Continue anyway, let MQTT client handle the connection
            
            client_id = f"cv-system-{uuid.uuid4().hex[:8]}"
            self.mqtt_client = mqtt.Client(client_id=client_id)
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Set connection timeout and connect
            print(f"   Attempting to connect to {self.mqtt_broker}:1883...")
            try:
                self.mqtt_client.connect_async(self.mqtt_broker, 1883, 60)
                self.mqtt_client.loop_start()  # Start background thread
                
                print(f"✅ MQTT client initialized (connecting to {self.mqtt_broker}...)")
                print(f"   Waiting for connection... (this may take a few seconds)")
                
                # Wait a bit for connection to establish
                time.sleep(2)
                
                # Check connection status
                if not self.mqtt_connected:
                    print(f"⚠️  MQTT connection pending... (checking again in 3 seconds)")
                    # Give it more time
                    time.sleep(3)
                    if not self.mqtt_connected:
                        print(f"❌ MQTT connection timeout - broker may be unreachable")
                        print(f"   Current status: mqtt_connected = {self.mqtt_connected}")
                        print(f"   💡 Run 'python test_mqtt_connection.py' to diagnose the issue")
            except Exception as conn_error:
                print(f"❌ Failed to start MQTT connection: {conn_error}")
                self.mqtt_connected = False
                    
        except ImportError:
            print("⚠️  paho-mqtt not installed. Install with: pip install paho-mqtt")
            print("   Weight-based pickup detection will be disabled")
            self.mqtt_connected = False
        except Exception as e:
            print(f"⚠️  MQTT initialization failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("   Weight-based pickup detection will be disabled")
            self.mqtt_connected = False
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback when MQTT client connects."""
        if rc == 0:
            self.mqtt_connected = True
            # Subscribe to weight events topic
            result = client.subscribe(self.mqtt_topic_weight)
            if result[0] == 0:
                print(f"✅ MQTT connected to {self.mqtt_broker}")
                print(f"   Subscribed to: {self.mqtt_topic_weight}")
            else:
                print(f"⚠️  MQTT connected but subscription failed with code {result[0]}")
        else:
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_messages.get(rc, f"Unknown error code {rc}")
            print(f"❌ MQTT connection failed with code {rc}: {error_msg}")
            print(f"   Broker: {self.mqtt_broker}:1883")
            print(f"   Topic: {self.mqtt_topic_weight}")
            self.mqtt_connected = False
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback when MQTT client disconnects."""
        self.mqtt_connected = False
        if rc != 0:
            print(f"⚠️  MQTT disconnected unexpectedly (code {rc})")
        else:
            print("ℹ️  MQTT disconnected")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT weight change events."""
        try:
            topic_str = msg.topic.decode('utf-8') if isinstance(msg.topic, bytes) else msg.topic
            message_str = msg.payload.decode('utf-8') if isinstance(msg.payload, bytes) else str(msg.payload)
            
            if topic_str == self.mqtt_topic_weight:
                # Parse: "CHANGE:-480"
                if message_str.startswith("CHANGE:"):
                    weight_change = int(message_str.split(":")[1])
                    timestamp = datetime.now()
                    
                    # Store event
                    self.recent_weight_events.append({
                        'weight_change_g': weight_change,
                        'timestamp': timestamp
                    })
                    
                    print(f"\n⚖️  Weight Event: {weight_change:+d}g at {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
                    
                    # Process weight event
                    self._handle_weight_event(weight_change, timestamp)
                else:
                    print(f"⚠️  Unknown MQTT message format: {message_str}")
        except Exception as e:
            print(f"⚠️  Error processing MQTT message: {e}")
            import traceback
            traceback.print_exc()
    
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
        print(f"   🔍 Looking for customer who picked up item ({abs(weight_change_g)}g)...")
        
        # Find confirmed customers in shelf zone
        candidates = self._find_customers_in_shelf_zone()
        
        if not candidates:
            print(f"   ⚠️  No confirmed customers in shelf zone")
            print(f"   📝 Logging as unmatched event")
            self._log_unmatched_event(weight_change_g, timestamp, "no_customer_in_zone")
            return
        
        # Rank candidates by proximity + hand position
        ranked = self._rank_customers_by_pickup_likelihood(candidates, timestamp)
        
        if not ranked:
            print(f"   ⚠️  No suitable candidate found")
            self._log_unmatched_event(weight_change_g, timestamp, "no_suitable_candidate")
            return
        
        # Ping closest/most likely customer
        best_customer = ranked[0]
        print(f"   ✅ Found {len(ranked)} candidate(s), best: {best_customer['customer_id']} (score: {best_customer['combined_score']:.0%})")
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
        print(f"   📝 Unmatched event logged: {abs(weight_change_g)}g, reason: {reason}")
    
    def _ping_customer_pickup(self, customer_data, weight_change_g, timestamp):
        """Ping customer: Update shopping cart with picked up item."""
        track_id = customer_data['track_id']
        customer_id = customer_data['customer_id']
        
        if track_id not in self.customers:
            print(f"   ⚠️  Customer {customer_id} not found in customers dict")
            return
        
        customer = self.customers[track_id]
        
        # Validate weight change (must be negative for pickup)
        if weight_change_g >= 0:
            print(f"   ⚠️  Invalid weight change: {weight_change_g}g (expected negative for pickup)")
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
                        print(f"   ⚠️  Could not parse last_pickup_time: {last_ping_time}")
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
                    print(f"   ⏸️  Rate limited: Last ping was {time_since_last:.1f}s ago (min 2s)")
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
                        print(f"   ⚠️  Duplicate item detected: {weight_grams}g within {time_diff:.1f}s - skipping")
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
    
    def confirm_pending_with_customer_id(self, customer_id, pending_id=None):
        """
        Confirm a PENDING track with customer_id from QR scan.
        
        Args:
            customer_id: Customer ID from QR code (e.g., "CUST_001")
            pending_id: Optional PENDING ID. If None, uses zone_active_pending.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        print(f"\n🔔 QR Confirmation Request:")
        print(f"   Customer ID: {customer_id}")
        print(f"   Pending ID (provided): {pending_id}")
        print(f"   Zone active pending: {self.zone_active_pending}")
        print(f"   Current pending tracks: {list(self.pending_tracks.keys())}")
        
        # Re-check zone status to get latest state (avoid race condition)
        zone_active, zone_pending_id, pending_count = self._check_qr_zone()
        print(f"   Zone status (re-checked): active={zone_active}, pending_id={zone_pending_id}, count={pending_count}")
        
        # If pending_id not provided, use zone_active_pending (from latest check)
        if pending_id is None:
            pending_id = zone_pending_id if zone_active and pending_count == 1 else self.zone_active_pending
            print(f"   Using pending_id: {pending_id}")
        
        if pending_id is None:
            # Debug: List all pending tracks and their positions
            print(f"   ❌ No PENDING track in QR zone")
            print(f"   Available pending tracks:")
            for tid, pending in self.pending_tracks.items():
                box = pending.get('box')
                in_zone = self._is_in_qr_zone(box) if box else False
                print(f"      - Track {tid}: {pending.get('pending_id')}, in_zone={in_zone}, box={box}")
            return False, "No PENDING track in QR zone"
        
        # Find track_id from pending_id
        track_id = None
        for tid, pending in self.pending_tracks.items():
            if pending.get('pending_id') == pending_id:
                track_id = tid
                print(f"   ✅ Found track_id: {track_id} for pending_id: {pending_id}")
                break
        
        if track_id is None:
            print(f"   ❌ PENDING track {pending_id} not found in pending_tracks")
            print(f"   Available pending_ids: {[p.get('pending_id') for p in self.pending_tracks.values()]}")
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
        
        print(f"   Validating pending track...")
        is_valid, validation_score, issues = self._validate_pending_track(pending, all_current_boxes)
        print(f"   Validation result: valid={is_valid}, score={validation_score:.0%}")
        if issues:
            print(f"   Issues: {issues}")
        
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
        
        print(f"✅ Confirmed | {customer_id} (Track {track_id}) from QR scan")
        print(f"   Validation: {validation_score:.0%} | Samples: {len(customer_data['feature_gallery'])}")
        
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
        print(f"🚪 Exit | {customer_id} (Duration: {duration:.1f}s)")

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
            print(f"      └─> ❌ No keypoints for upper body check")
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
                print(f"      └─> ✅ Nose visible at y={nose_y:.0f} - enough for clothing color")
                return True
        
        # Check shoulders (torso) - enough to get shirt color
        left_shoulder = keypoints[5] if len(keypoints) > 5 else None
        right_shoulder = keypoints[6] if len(keypoints) > 6 else None
        
        if left_shoulder is not None and len(left_shoulder) >= 3 and left_shoulder[2] > 0.3:
            shoulder_y = left_shoulder[1]
            # Shoulder should be in upper 60% of box
            if y1 <= shoulder_y <= y1 + box_height * 0.6:
                print(f"      └─> ✅ Left shoulder visible at y={shoulder_y:.0f} - enough for clothing color")
                return True
        
        if right_shoulder is not None and len(right_shoulder) >= 3 and right_shoulder[2] > 0.3:
            shoulder_y = right_shoulder[1]
            if y1 <= shoulder_y <= y1 + box_height * 0.6:
                print(f"      └─> ✅ Right shoulder visible at y={shoulder_y:.0f} - enough for clothing color")
                return True
        
        # No upper body keypoints found
        print(f"      └─> ❌ Upper body not visible (no nose or shoulder keypoints)")
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
            print(f"      └─> ❌ No box data for legs check")
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
            
            # DEBUG: Show ankle keypoint data
            print(f"      └─> DEBUG Ankle check:")
            if left_ankle is not None:
                print(f"          Left ankle: x={left_ankle[0]:.0f}, y={left_ankle[1]:.0f}, conf={left_ankle[2]:.3f}")
            else:
                print(f"          Left ankle: None")
            if right_ankle is not None:
                print(f"          Right ankle: x={right_ankle[0]:.0f}, y={right_ankle[1]:.0f}, conf={right_ankle[2]:.3f}")
            else:
                print(f"          Right ankle: None")
            print(f"          Box: x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}, height={box_height:.0f}")
            print(f"          Lower 30% range: y={y1 + box_height * 0.7:.0f} to {y2:.0f}")
            
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
                        print(f"      └─> ✅ Left ankle (orange) visible at ({ankle_x:.0f}, {ankle_y:.0f}), conf={ankle_conf:.3f}")
                    else:
                        print(f"      └─> ⚠️  Left ankle out of bounds: y={ankle_y:.0f} (need {lower_bound:.0f}-{upper_bound:.0f}), x={ankle_x:.0f}")
                else:
                    print(f"      └─> ⚠️  Left ankle confidence too low: {ankle_conf:.3f} < 0.2")
            
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
                        print(f"      └─> ✅ Right ankle (orange) visible at ({ankle_x:.0f}, {ankle_y:.0f}), conf={ankle_conf:.3f}")
                    else:
                        print(f"      └─> ⚠️  Right ankle out of bounds: y={ankle_y:.0f} (need {lower_bound:.0f}-{upper_bound:.0f}), x={ankle_x:.0f}")
                else:
                    print(f"      └─> ⚠️  Right ankle confidence too low: {ankle_conf:.3f} < 0.2")
            
            # REQUIRED: At least 1 ankle keypoint must be visible
            if ankles_visible >= 1:
                print(f"      └─> ✅ Legs visible ({ankles_visible} ankle keypoint(s) detected)")
                return True
            else:
                print(f"      └─> ❌ No ankle keypoints (orange) detected - need at least 1")
                return False
        
        # If no keypoints, cannot verify legs
        print(f"      └─> ❌ No keypoints available - cannot verify ankle keypoints")
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
            print(f"   ❌ ReID: Current detection has no upper body visible - cannot re-identify")
            return None  # Cannot re-identify without seeing upper body
        if not current_legs_visible:
            print(f"   ❌ ReID: Current detection has no ankle keypoint (orange) - cannot re-identify")
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
                print(f"   ❌ ReID: Lost track {lost_id} had no upper body visible - skipping")
                continue  # Skip if lost track didn't have upper body
            if not lost_legs_visible:
                print(f"   ❌ ReID: Lost track {lost_id} had no ankle keypoint (orange) - skipping")
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
                print(f"   ⚠️  Relatives detected near lost track {lost_id}, requiring higher similarity")
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
        return {
            'active_customers': len(self.customers),
            'occluded_tracks': len(self.lost_tracks),
            'pending_tracks': len(self.pending_tracks),
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
    print("   QR Confirmation: Bottom-left zone scanning")
    print("="*70 + "\n")
    
    # Initialize tracker
    print("🚀 Initializing tracker...")
    print("   This may take a moment to load YOLO model...")
    tracker = RetailCustomerTracker(
        detection_model='yolo11n-pose.pt',  # Use pose model for keypoints
        tracker_config='config/botsort_reid.yaml'
    )
    print("   Tracker initialized!")
    
    # Initialize MQTT for weight-based pickup detection
    print("🔌 Initializing MQTT...")
    tracker._init_mqtt()
    print("   MQTT initialization complete")
    
    # Open webcam with timeout
    print("📹 Opening camera...")
    import time
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        # Set timeout for camera initialization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Try to read a frame to verify camera works
        print("   Testing camera...")
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("❌ Camera opened but cannot read frames!")
            cap.release()
            cap = None
        else:
            print(f"✅ Camera ready! Frame size: {test_frame.shape}")
    except Exception as e:
        print(f"❌ Camera error: {e}")
        if cap:
            cap.release()
        cap = None
    
    if cap is None or not cap.isOpened():
        print("⚠️  Camera not available!")
        print("   Dashboard will still work for MQTT events")
        print("   Press Ctrl+C to exit, or wait for MQTT events...")
        print("\n   Dashboard: http://localhost:8080/dashboard")
        
        # Keep running for dashboard/MQTT only
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        return
    
    frame_count = 0
    fps_list = []
    
    print("📹 Camera ready!")
    print("\n📋 Validation Requirements:")
    print(f"   • Min samples: {tracker.min_samples_required} frames")
    print(f"   • Min confidence: {tracker.min_confidence_avg:.0%}")
    print(f"   • Min feature quality: {tracker.min_feature_quality:.0%}")
    print("\n⌨️  Keys:")
    print("      q = quit")
    print("      c = confirm selected pending track (if validated)")
    print("      1-9 = select pending track by number")
    print("      s = save logs")
    print("      i = info")
    print("\n💡 Note: This is camera tracking only.")
    print("   For dashboard, run: python run_dashboard.py\n")
    
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
            elif key == ord('i'):
                print("\n" + "="*70)
                print("📊 TRACKER STATISTICS")
                for k, v in stats.items():
                    print(f"   {k}: {v}")
                print("="*70 + "\n")
            
            frame_count += 1
            if frame_count % 30 == 0:
                avg_fps = sum(fps_list[-30:]) / min(30, len(fps_list))
                print(f"Frame {frame_count}: FPS={avg_fps:.1f}, Active={stats['active_customers']}, Pending={stats['pending_tracks']}, Occluded={stats['occluded_tracks']}")
    
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
