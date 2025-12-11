"""
Holding Detection Module - Hand Region Content Analysis
Detects if a person is holding ANY object by analyzing content in hand region.
Uses 3 methods: Edge Detection + Mini YOLO + Texture Analysis.
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict
from ultralytics import YOLO


class HoldingDetector:
    """
    Hand Region Content Analysis: Detects objects in hand region using:
    1. Edge Detection (detects edges/contours)
    2. Mini YOLO (object detection in hand region)
    3. Texture Analysis (variance, patterns)
    """
    
    def __init__(self, yolo_model_path='yolo11n.pt'):
        # Temporal smoothing
        self.frames_to_confirm_holding = 3  # ~0.1s at 30fps
        self.frames_to_confirm_release = 5  # ~0.17s
        self.decay_rate = 0.5
        
        # Combined detection threshold
        self.min_combined_score = 0.50  # 50% combined score to confirm holding
        
        # Feature weights (sum = 1.0)
        self.weight_edge = 0.40      # 40% - Edge detection
        self.weight_yolo = 0.35      # 35% - Mini YOLO detection
        self.weight_texture = 0.25   # 25% - Texture analysis
        
        # Edge detection parameters
        self.edge_density_threshold = 0.15  # 15% edge density = likely holding (default)
        self.edge_density_threshold_small = 0.12  # 12% for small hand regions (< 1000px²)
        self.edge_density_threshold_large = 0.18  # 18% for large hand regions (> 2500px²)
        self.small_region_area = 1000  # Threshold for small region (px²)
        self.large_region_area = 2500  # Threshold for large region (px²)
        self.canny_low = 50
        self.canny_high = 150
        
        # Mini YOLO parameters
        self.yolo_confidence_min = 0.25  # Lower confidence for hand region
        self.yolo_model = None  # Lazy load
        self.yolo_model_path = yolo_model_path
        
        # Texture analysis parameters
        self.texture_variance_threshold = 800  # Variance threshold (pixel intensity)
        self.texture_lbp_threshold = 0.12  # LBP variance threshold
        
        # Hand region extraction
        self.hand_region_width_ratio = 0.24  # 24% of person height (width)
        self.hand_region_height_ratio = 0.28  # 28% of person height (height)
        
        # Keypoint confidence
        self.min_wrist_confidence = 0.3
        
        # COCO keypoint indices
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        
        # Holding state tracking
        self.holding_states = {}  # {customer_id: holding_state}
    
    def detect_holding(self, 
                      customer_id: str,
                      person_bbox: np.ndarray,
                      keypoints: np.ndarray,
                      detected_objects=None,  # Not used, kept for compatibility
                      frame: np.ndarray = None) -> Dict:
        """
        Hand Region Content Analysis: Detects objects in hand region.
        
        Args:
            customer_id: Customer identifier
            person_bbox: [x1, y1, x2, y2]
            keypoints: YOLO pose keypoints [17, 3] (x, y, confidence)
            detected_objects: Not used (kept for compatibility)
            frame: Current frame (REQUIRED for hand region analysis)
            
        Returns:
            Dict with:
                - is_holding: bool
                - confidence: float (0.0-1.0)
                - method: str ('hand-region')
                - object_class: str ('unknown' - we don't detect object type)
                - hand_used: str ('left', 'right', 'both', 'unknown')
        """
        # Initialize holding state for new customer
        if customer_id not in self.holding_states:
            self.holding_states[customer_id] = {
                'frames_holding': 0,
                'frames_not_holding': 0,
                'is_holding': False
            }
        
        state = self.holding_states[customer_id]
        
        # Check if frame is available
        if frame is None:
            return {'detected': False, 'confidence': 0.0, 'method': 'hand-region', 
                   'object_class': 'unknown', 'hand_used': 'unknown'}
        
        # Hand Region Content Analysis
        result = self._analyze_hand_region(person_bbox, keypoints, frame)
        
        # Apply temporal smoothing
        result = self._apply_temporal_smoothing(customer_id, result)
        
        return result
    
    def _analyze_hand_region(self,
                             person_bbox: np.ndarray,
                             keypoints: np.ndarray,
                             frame: np.ndarray) -> Dict:
        """
        Hand Region Content Analysis: Combines 3 methods.
        
        1. Edge Detection (40%): Detects edges/contours in hand region
        2. Mini YOLO (35%): Object detection in hand region
        3. Texture Analysis (25%): Variance and patterns
        """
        
        # Extract hand regions
        hand_regions = self._extract_hand_regions(person_bbox, keypoints, frame)
        
        if not hand_regions:
            return {'detected': False, 'confidence': 0.0, 'method': 'hand-region',
                   'object_class': 'unknown', 'hand_used': 'unknown'}
        
        # Analyze each hand region and take best score
        best_score = 0.0
        best_hand = 'unknown'
        all_scores = []
        
        for hand_name, hand_region in hand_regions.items():
            if hand_region is None or hand_region.size == 0:
                continue
            
            # Method 1: Edge Detection (40% weight)
            edge_score = self._detect_edges(hand_region)
            
            # Method 2: Mini YOLO (35% weight)
            yolo_score = self._detect_with_mini_yolo(hand_region)
            
            # Method 3: Texture Analysis (25% weight)
            texture_score = self._analyze_texture(hand_region)
            
            # Combined score
            combined_score = (self.weight_edge * edge_score +
                            self.weight_yolo * yolo_score +
                            self.weight_texture * texture_score)
            
            all_scores.append({
                'hand': hand_name,
                'edge': edge_score,
                'yolo': yolo_score,
                'texture': texture_score,
                'combined': combined_score
            })
            
            if combined_score > best_score:
                best_score = combined_score
                best_hand = hand_name
        
        # Determine hand used
        hand_used = 'unknown'
        if len(hand_regions) == 2:
            hand_used = 'both'
        elif 'left' in hand_regions and hand_regions['left'] is not None:
            hand_used = 'left'
        elif 'right' in hand_regions and hand_regions['right'] is not None:
            hand_used = 'right'
        
        # DEBUG
        print(f"      └─> Hand region analysis:")
        for score_info in all_scores:
            print(f"          {score_info['hand']}: edge={score_info['edge']:.2f}, "
                  f"yolo={score_info['yolo']:.2f}, texture={score_info['texture']:.2f}, "
                  f"combined={score_info['combined']:.2f}")
        print(f"      └─> Best score: {best_score:.2f} (threshold: {self.min_combined_score})")
        
        if best_score >= self.min_combined_score:
            print(f"      └─> ✅ Hand region match!")
            return {
                'detected': True,
                'confidence': best_score,
                'method': 'hand-region',
                'object_class': 'unknown',
                'hand_used': hand_used
            }
        
        print(f"      └─> ❌ Hand region below threshold")
        return {'detected': False, 'confidence': best_score, 'method': 'hand-region'}
    
    def _extract_hand_regions(self,
                              person_bbox: np.ndarray,
                              keypoints: np.ndarray,
                              frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract hand regions around wrist keypoints."""
        hand_regions = {}
        
        person_height = person_bbox[3] - person_bbox[1]
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate hand region size
        region_width = int(person_height * self.hand_region_width_ratio)
        region_height = int(person_height * self.hand_region_height_ratio)
        
        # Extract left hand region
        left_wrist = keypoints[self.LEFT_WRIST]
        if left_wrist[2] >= self.min_wrist_confidence:
            wx, wy = int(left_wrist[0]), int(left_wrist[1])
            
            x1 = max(0, wx - region_width // 2)
            y1 = max(0, wy - region_height // 4)  # More above wrist
            x2 = min(frame_width, wx + region_width // 2)
            y2 = min(frame_height, wy + region_height * 3 // 4)  # More below wrist
            
            if x2 > x1 and y2 > y1:
                hand_regions['left'] = frame[y1:y2, x1:x2]
            else:
                hand_regions['left'] = None
        else:
            hand_regions['left'] = None
        
        # Extract right hand region
        right_wrist = keypoints[self.RIGHT_WRIST]
        if right_wrist[2] >= self.min_wrist_confidence:
            wx, wy = int(right_wrist[0]), int(right_wrist[1])
            
            x1 = max(0, wx - region_width // 2)
            y1 = max(0, wy - region_height // 4)
            x2 = min(frame_width, wx + region_width // 2)
            y2 = min(frame_height, wy + region_height * 3 // 4)
            
            if x2 > x1 and y2 > y1:
                hand_regions['right'] = frame[y1:y2, x1:x2]
            else:
                hand_regions['right'] = None
        else:
            hand_regions['right'] = None
        
        return hand_regions
    
    def _detect_edges(self, hand_region: np.ndarray) -> float:
        """
        Method 1: Edge Detection (Size-Adaptive)
        Detects edges/contours in hand region.
        Objects have more edges than just a hand.
        
        Adaptive threshold based on hand region size:
        - Small regions (< 1000px²): Lower threshold (0.12) - more lenient
        - Medium regions (1000-2500px²): Default threshold (0.15)
        - Large regions (> 2500px²): Higher threshold (0.18) - more strict
        """
        if hand_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(hand_region.shape) == 3:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_region
        
        # Calculate region area for adaptive threshold
        region_area = gray.shape[0] * gray.shape[1]
        
        # Determine adaptive threshold based on region size
        if region_area < self.small_region_area:
            # Small hand region (person far from camera)
            adaptive_threshold = self.edge_density_threshold_small  # 0.12 (more lenient)
            size_category = "small"
        elif region_area > self.large_region_area:
            # Large hand region (person close to camera)
            adaptive_threshold = self.edge_density_threshold_large  # 0.18 (more strict)
            size_category = "large"
        else:
            # Medium hand region
            adaptive_threshold = self.edge_density_threshold  # 0.15 (default)
            size_category = "medium"
        
        # Resize if too small (for edge detection quality)
        if gray.shape[0] < 20 or gray.shape[1] < 20:
            gray = cv2.resize(gray, (40, 40))
            # Recalculate area after resize
            region_area = gray.shape[0] * gray.shape[1]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Normalize to 0-1 score using adaptive threshold
        # Hand alone: 5-10% density
        # Hand + object: 15-30% density
        edge_score = min(1.0, edge_density / adaptive_threshold)
        
        # DEBUG: Show adaptive threshold info
        print(f"          Edge: density={edge_density:.3f}, threshold={adaptive_threshold:.3f} ({size_category}, area={region_area}px²), score={edge_score:.2f}")
        
        return edge_score
    
    def _detect_with_mini_yolo(self, hand_region: np.ndarray) -> float:
        """
        Method 3: Mini YOLO Detection
        Run YOLO detection on hand region only.
        If ANY object detected (except person) → likely holding.
        """
        if hand_region.size == 0:
            return 0.0
        
        # Lazy load YOLO model
        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO(self.yolo_model_path)
            except Exception as e:
                print(f"      ⚠️  Could not load YOLO model: {e}")
                return 0.0
        
        # Resize hand region if too small (YOLO needs min size)
        min_size = 64
        if hand_region.shape[0] < min_size or hand_region.shape[1] < min_size:
            scale = max(min_size / hand_region.shape[0], min_size / hand_region.shape[1])
            new_h = int(hand_region.shape[0] * scale)
            new_w = int(hand_region.shape[1] * scale)
            hand_region = cv2.resize(hand_region, (new_w, new_h))
        
        try:
            # Run YOLO on hand region
            results = self.yolo_model(
                hand_region,
                conf=self.yolo_confidence_min,
                verbose=False
            )
            
            result = results[0]
            
            # Check if any objects detected (ignore 'person' class)
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                class_names = result.names
                
                # Find best non-person object
                best_conf = 0.0
                for cls, conf in zip(classes, confs):
                    obj_name = class_names[cls]
                    if obj_name != 'person':
                        best_conf = max(best_conf, float(conf))
                
                if best_conf > 0:
                    yolo_score = min(1.0, best_conf / 0.5)  # Normalize to 0-1
                    return yolo_score
            
            return 0.0
            
        except Exception as e:
            # YOLO might fail on very small regions
            return 0.0
    
    def _analyze_texture(self, hand_region: np.ndarray) -> float:
        """
        Method 4: Texture Analysis
        Objects have different texture than skin (variance, patterns).
        """
        if hand_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(hand_region.shape) == 3:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_region
        
        # Resize if too small
        if gray.shape[0] < 20 or gray.shape[1] < 20:
            gray = cv2.resize(gray, (40, 40))
        
        # Method 1: Variance of pixel intensities
        variance = np.var(gray.astype(np.float32))
        
        # Method 2: Local Binary Pattern (LBP) variance
        # Simple LBP approximation
        lbp_variance = self._calculate_lbp_variance(gray)
        
        # Combine texture scores
        variance_score = min(1.0, variance / self.texture_variance_threshold)
        lbp_score = min(1.0, lbp_variance / self.texture_lbp_threshold)
        
        texture_score = (variance_score + lbp_score) / 2.0
        
        return texture_score
    
    def _calculate_lbp_variance(self, gray: np.ndarray) -> float:
        """Calculate Local Binary Pattern variance (vectorized for speed)."""
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        
        # Resize if too large (for speed)
        if gray.shape[0] > 100 or gray.shape[1] > 100:
            gray = cv2.resize(gray, (50, 50))
        
        h, w = gray.shape
        
        # Vectorized LBP calculation
        center = gray[1:h-1, 1:w-1]
        
        # Compare with 8 neighbors
        neighbors = [
            gray[0:h-2, 0:w-2],      # top-left
            gray[0:h-2, 1:w-1],      # top
            gray[0:h-2, 2:w],        # top-right
            gray[1:h-1, 2:w],        # right
            gray[2:h, 2:w],          # bottom-right
            gray[2:h, 1:w-1],        # bottom
            gray[2:h, 0:w-2],        # bottom-left
            gray[1:h-1, 0:w-2]       # left
        ]
        
        # Build LBP code
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        for i, neighbor in enumerate(neighbors):
            lbp |= ((neighbor > center).astype(np.uint8) << i)
        
        # Calculate variance of LBP codes
        lbp_variance = np.var(lbp.astype(np.float32))
        
        return lbp_variance
    
    def _apply_temporal_smoothing(self, customer_id: str, detection_result: Dict) -> Dict:
        """Apply temporal smoothing to reduce flickering."""
        state = self.holding_states[customer_id]
        
        if detection_result['detected']:
            state['frames_holding'] += 1
            state['frames_not_holding'] = 0
        else:
            state['frames_not_holding'] += 1
            state['frames_holding'] = max(0, state['frames_holding'] - self.decay_rate)
        
        # DEBUG
        print(f"      └─> Temporal: frames_holding={state['frames_holding']:.1f}/{self.frames_to_confirm_holding}, "
              f"frames_not={state['frames_not_holding']}/{self.frames_to_confirm_release}")
        
        # Determine final holding status
        if state['frames_holding'] >= self.frames_to_confirm_holding:
            state['is_holding'] = True
            detection_result['is_holding'] = True
            detection_result['status'] = 'confirmed_holding'  # For display
            print(f"      └─> ✅ CONFIRMED HOLDING!")
        elif state['frames_not_holding'] >= self.frames_to_confirm_release:
            state['is_holding'] = False
            detection_result['is_holding'] = False
            detection_result['status'] = 'confirmed_not_holding'  # For display
            print(f"      └─> ❌ CONFIRMED NOT HOLDING")
        else:
            # Keep previous state during transition
            detection_result['is_holding'] = state['is_holding']
            detection_result['status'] = 'transitioning'  # For display
            print(f"      └─> ⏳ Transitioning... (keeping previous: {state['is_holding']})")
        
        return detection_result
    
    def draw_debug_visualization(self,
                                frame: np.ndarray,
                                person_bbox: np.ndarray,
                                keypoints: np.ndarray,
                                holding_result: Dict) -> np.ndarray:
        """Draw debug visualization on frame."""
        
        # Draw person bbox
        x1, y1, x2, y2 = map(int, person_bbox)
        color = (0, 255, 0) if holding_result.get('is_holding') else (128, 128, 128)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw hand keypoints and hand regions
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        
        person_height = person_bbox[3] - person_bbox[1]
        region_width = int(person_height * self.hand_region_width_ratio)
        region_height = int(person_height * self.hand_region_height_ratio)
        
        # Draw left hand region
        if left_wrist[2] >= self.min_wrist_confidence:
            wx, wy = int(left_wrist[0]), int(left_wrist[1])
            hx1 = max(0, wx - region_width // 2)
            hy1 = max(0, wy - region_height // 4)
            hx2 = min(frame.shape[1], wx + region_width // 2)
            hy2 = min(frame.shape[0], wy + region_height * 3 // 4)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)  # Blue for left
            cv2.circle(frame, (wx, wy), 5, (255, 0, 0), -1)
        
        # Draw right hand region
        if right_wrist[2] >= self.min_wrist_confidence:
            wx, wy = int(right_wrist[0]), int(right_wrist[1])
            hx1 = max(0, wx - region_width // 2)
            hy1 = max(0, wy - region_height // 4)
            hx2 = min(frame.shape[1], wx + region_width // 2)
            hy2 = min(frame.shape[0], wy + region_height * 3 // 4)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)  # Red for right
            cv2.circle(frame, (wx, wy), 5, (0, 0, 255), -1)
        
        # Draw holding status
        status_text = f"Holding: {holding_result.get('is_holding', False)}"
        conf_text = f"Score: {holding_result.get('confidence', 0.0):.2f}"
        
        cv2.putText(frame, status_text, (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, conf_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def get_holding_status(self, customer_id: str) -> bool:
        """Get current holding status for a customer."""
        if customer_id in self.holding_states:
            return self.holding_states[customer_id]['is_holding']
        return False
    
    def reset_customer(self, customer_id: str):
        """Reset holding state for a customer (when they exit)."""
        if customer_id in self.holding_states:
            del self.holding_states[customer_id]

