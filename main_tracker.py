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
from holding_detector import HoldingDetector


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
        
        # Holding Detection System
        self.holding_detector = HoldingDetector()
        
        # Logs
        self.events = []
        
        print(f"✅ Tracker ready | Device: {device}")
        print(f"   Model: {detection_model}")
        print(f"   Config: {tracker_config}")
        print(f"   Holding detection: Hand Region Analysis (Edge + YOLO + Texture)")
    
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
        
        # Draw trajectory and custom overlays
        if return_annotated:
            annotated_frame = self._draw_trajectories(annotated_frame)
            annotated_frame = self._draw_pending_tracks(annotated_frame, result)
            annotated_frame = self._draw_holding_status(annotated_frame)
        
        return result, annotated_frame, list(current_track_ids)
    
    def _update_track(self, track_id, box, conf, frame, keypoints=None, detected_objects=None):
        """Update or create tracking information for a track."""
        
        # Extract appearance features
        features = self.reid.extract_features(frame, box)

        # Try ReID with lost tracks before creating new
        if track_id not in self.customers and track_id not in self.pending_tracks:
            matched = self._try_reid(track_id, box, features)
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
            
            # Check if ready for confirmation
            is_valid, validation_score, _ = self._validate_pending_track(pending)
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
        
        # Update feature gallery
        if features is not None:
            if 'feature_gallery' not in customer:
                customer['feature_gallery'] = deque(maxlen=self.feature_gallery_size)
            customer['feature_gallery'].append(features)
        
        # Store trajectory
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        self.track_history[track_id].append((center_x, center_y))
        
        # Holding Detection (for confirmed customers only) - Pure Hand-State
        if keypoints is not None:
            # DEBUG: Show holding detection is running
            print(f"🔍 DEBUG | Running holding detection for {customer['customer_id']}")
            print(f"   Keypoints shape: {keypoints.shape}")
            
            holding_result = self.holding_detector.detect_holding(
                customer_id=customer['customer_id'],
                person_bbox=box,
                keypoints=keypoints,
                detected_objects=None,  # Not used in pure hand-state
                frame=frame
            )
            
            # DEBUG: Show holding result
            print(f"   Result: is_holding={holding_result.get('is_holding')}, "
                  f"confidence={holding_result.get('confidence', 0):.2f}, "
                  f"method={holding_result.get('method')}")
            
            # Update customer holding status
            customer['holding_status'] = holding_result
            
            # Log holding events
            if holding_result.get('is_holding'):
                if not customer.get('was_holding', False):
                    # Started holding
                    self.events.append({
                        'event': 'STARTED_HOLDING',
                        'customer_id': customer['customer_id'],
                        'track_id': track_id,
                        'timestamp': datetime.now().isoformat(),
                        'method': holding_result.get('method', 'hand-state'),
                        'confidence': holding_result.get('confidence', 0.0)
                    })
                    print(f"🤚 Holding | {customer['customer_id']} started holding")
                customer['was_holding'] = True
            else:
                if customer.get('was_holding', False):
                    # Stopped holding
                    self.events.append({
                        'event': 'STOPPED_HOLDING',
                        'customer_id': customer['customer_id'],
                        'track_id': track_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"👐 Released | {customer['customer_id']} released item")
                customer['was_holding'] = False
        else:
            # DEBUG: Show why holding detection not running
            print(f"⚠️  DEBUG | No keypoints for {customer['customer_id']}")
        
        # Clean up lost track entry if customer re-appears
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]
    
    def _validate_pending_track(self, pending):
        """
        Validate if a pending track has enough information for confirmation.
        
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
        
        # Overall validation score
        validation_score = np.mean(scores) if scores else 0.0
        is_valid = validation_score >= 0.8 and len(issues) == 0
        
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
        
        # Validate before confirming
        is_valid, validation_score, issues = self._validate_pending_track(pending)
        
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
                self.lost_tracks[track_id] = {
                    'lost_time': current_time,
                    'data': customer.copy(),
                    'last_box': customer.get('last_box', customer.get('entry_box')),
                    'features': avg_feat
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

    def _try_reid(self, track_id, box, features):
        """Try to re-identify a lost track using appearance + IoU."""
        if features is None:
            return None
        best = None
        now = datetime.now()
        for lost_id, data in list(self.lost_tracks.items()):
            # Time gating
            lost_time = data['lost_time']
            if (now - lost_time).total_seconds() > self.max_lost_time:
                continue
            # IoU gating
            iou = self._iou(box, data.get('last_box'))
            if iou < 0.1:  # allow low IoU but still gate a bit
                continue
            # Similarity
            sim = LightweightReID.similarity(features, data.get('features'))
            # Two-stage matching
            if sim >= self.reid_high_thresh or (sim >= self.reid_low_thresh and iou >= 0.2):
                if best is None or sim > best['sim']:
                    best = {
                        'lost_id': lost_id,
                        'sim': sim,
                        'iou': iou,
                        'data': data['data'],
                        'features': data.get('features')
                    }
        if best is None:
            return None
        # Reuse customer data with new track_id
        customer = best['data']
        customer['track_id'] = track_id
        customer['last_box'] = box
        customer['last_detection_time'] = now
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
                
                # Validate track
                is_valid, validation_score, issues = self._validate_pending_track(pending)
                
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
        """Draw holding status for confirmed customers with clear visual indicators."""
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
            
            # Determine display based on status
            if status == 'confirmed_holding':
                # CONFIRMED HOLDING - Green, large text
                bg_color = (0, 200, 0)  # Green background
                text_color = (255, 255, 255)  # White text
                status_text = "✅ CONFIRMED HOLDING"
                conf_text = f"Score: {conf:.2f}"
                font_scale = 0.7
                thickness = 2
                icon = "🤚"
                
            elif status == 'confirmed_not_holding':
                # CONFIRMED NOT HOLDING - Gray, smaller text
                bg_color = (100, 100, 100)  # Gray background
                text_color = (255, 255, 255)  # White text
                status_text = "❌ NOT HOLDING"
                conf_text = f"Score: {conf:.2f}"
                font_scale = 0.5
                thickness = 1
                icon = "👐"
                
            else:  # transitioning
                # TRANSITIONING - Yellow/Orange, medium text
                bg_color = (0, 165, 255)  # Orange background
                text_color = (255, 255, 255)  # White text
                status_text = "⏳ CHECKING..."
                conf_text = f"Score: {conf:.2f}"
                font_scale = 0.6
                thickness = 2
                icon = "🔍"
            
            # Calculate text positions
            full_text = f"{icon} {status_text}"
            text_size, _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness)
            
            # Position: Top-right of bounding box
            text_x = x2 - max(text_size[0], conf_size[0]) - 15
            text_y = y1 - 10
            
            # Draw background rectangle
            bg_height = text_size[1] + conf_size[1] + 15
            bg_width = max(text_size[0], conf_size[0]) + 20
            cv2.rectangle(frame, 
                         (text_x - 5, text_y - bg_height), 
                         (x2, text_y + 5), 
                         bg_color, -1)
            
            # Draw border
            cv2.rectangle(frame, 
                         (text_x - 5, text_y - bg_height), 
                         (x2, text_y + 5), 
                         (255, 255, 255), 2)
            
            # Draw status text
            cv2.putText(frame, full_text, 
                       (text_x, text_y - conf_size[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
            # Draw confidence text
            cv2.putText(frame, conf_text, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, text_color, thickness)
            
            # Draw indicator circle on bounding box corner
            indicator_pos = (x2 - 10, y1 + 10)
            cv2.circle(frame, indicator_pos, 8, bg_color, -1)
            cv2.circle(frame, indicator_pos, 8, (255, 255, 255), 2)
        
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
    print("="*70 + "\n")
    
    # Initialize tracker
    # Make sure botsort_reid.yaml is in ultralytics/cfg/trackers/
    tracker = RetailCustomerTracker(
        detection_model='yolo11n-pose.pt',  # Use pose model for keypoints
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
    print("\n📋 Validation Requirements:")
    print(f"   • Min samples: {tracker.min_samples_required} frames")
    print(f"   • Min confidence: {tracker.min_confidence_avg:.0%}")
    print(f"   • Min feature quality: {tracker.min_feature_quality:.0%}")
    print("\n⌨️  Keys:")
    print("      q = quit")
    print("      c = confirm selected pending track (if validated)")
    print("      1-9 = select pending track by number")
    print("      s = save logs")
    print("      i = info\n")
    
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
                    
                    # Get validation status
                    is_valid, validation_score, _ = tracker._validate_pending_track(pending)
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
