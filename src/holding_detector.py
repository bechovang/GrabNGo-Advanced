"""
Holding Detection Module - Simplified Approach
Detects if a person is holding ANY object using:
1. MediaPipe Hands (finger state detection)
2. Dominant Color Detection
3. Color Variance Analysis
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict
import mediapipe as mp


class HoldingDetector:
    """
    Simplified Holding Detection using 3 methods:
    1. MediaPipe Hands (finger state: nắm lại vs mở ra)
    2. Dominant Color Detection (màu chính trong hand region)
    3. Color Variance Analysis (độ đa dạng màu sắc)
    """
    
    def __init__(self):
        # Temporal smoothing
        self.frames_to_confirm_holding = 3  # ~0.1s at 30fps
        self.frames_to_confirm_release = 5  # ~0.17s
        self.decay_rate = 0.5
        
        # Combined detection threshold
        self.min_combined_score = 0.50  # 50% combined score to confirm holding (reduced because finger state unreliable)
        
        # Feature weights (sum = 1.0)
        # Adjusted: Finger state unreliable, so reduce its weight
        self.weight_finger = 0.30      # 30% - Finger state (nắm lại) - reduced because often fails
        self.weight_dominant = 0.40    # 40% - Dominant color (increased)
        self.weight_variance = 0.30   # 30% - Color variance (increased)
        
        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lowered from 0.5 to detect more hands
            min_tracking_confidence=0.3     # Lowered from 0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Finger state detection parameters
        self.fist_threshold = 0.25  # Distance ratio threshold for "nắm lại" (increased from 0.15)
        # If fingertips are within 25% of palm distance → nắm lại
        # Increased because MediaPipe may not detect perfectly in small hand regions
        
        # Dominant color parameters
        self.dominant_color_k = 3  # K-means clusters (skin, object, background)
        self.skin_color_tolerance = 30  # HSV tolerance for skin color
        
        # Color variance parameters
        self.color_variance_threshold = 1200  # Variance threshold (pixel intensity)
        # Da: variance thấp (~500-800)
        # Có object: variance cao (>1200)
        
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
        Simplified Holding Detection using 3 methods.
        
        Args:
            customer_id: Customer identifier
            person_bbox: [x1, y1, x2, y2]
            keypoints: YOLO pose keypoints [17, 3] (x, y, confidence)
            detected_objects: Not used (kept for compatibility)
            frame: Current frame (REQUIRED)
            
        Returns:
            Dict with:
                - is_holding: bool
                - confidence: float (0.0-1.0)
                - method: str ('finger-color')
                - object_class: str ('unknown')
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
            return {'detected': False, 'confidence': 0.0, 'method': 'finger-color', 
                   'object_class': 'unknown', 'hand_used': 'unknown'}
        
        # Analyze with 3 methods
        result = self._analyze_holding(person_bbox, keypoints, frame)
        
        # Apply temporal smoothing
        result = self._apply_temporal_smoothing(customer_id, result)
        
        return result
    
    def _analyze_holding(self,
                         person_bbox: np.ndarray,
                         keypoints: np.ndarray,
                         frame: np.ndarray) -> Dict:
        """
        Analyze holding using 3 methods:
        1. MediaPipe Hands (finger state)
        2. Dominant Color Detection
        3. Color Variance Analysis
        """
        
        # Extract hand regions
        hand_regions = self._extract_hand_regions(person_bbox, keypoints, frame)
        
        if not hand_regions:
            return {'detected': False, 'confidence': 0.0, 'method': 'finger-color',
                   'object_class': 'unknown', 'hand_used': 'unknown'}
        
        # Analyze each hand region and take best score
        best_score = 0.0
        best_hand = 'unknown'
        all_scores = []
        
        for hand_name, hand_region in hand_regions.items():
            if hand_region is None or hand_region.size == 0:
                continue
            
            # Method 1: MediaPipe Hands (finger state) - 50% weight
            finger_score = self._detect_finger_state(hand_region, frame)
            
            # Method 2: Dominant Color Detection - 30% weight
            dominant_score = self._detect_dominant_color(hand_region)
            
            # Method 3: Color Variance Analysis - 20% weight
            variance_score = self._analyze_color_variance(hand_region)
            
            # Combined score
            combined_score = (self.weight_finger * finger_score +
                           self.weight_dominant * dominant_score +
                           self.weight_variance * variance_score)
            
            all_scores.append({
                'hand': hand_name,
                'finger': finger_score,
                'dominant': dominant_score,
                'variance': variance_score,
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
        print(f"      └─> Holding analysis:")
        for score_info in all_scores:
            print(f"          {score_info['hand']}: finger={score_info['finger']:.2f}, "
                  f"dominant={score_info['dominant']:.2f}, variance={score_info['variance']:.2f}, "
                  f"combined={score_info['combined']:.2f}")
        print(f"      └─> Best score: {best_score:.2f} (threshold: {self.min_combined_score})")
        
        if best_score >= self.min_combined_score:
            print(f"      └─> ✅ Holding detected!")
            return {
                'detected': True,
                'confidence': best_score,
                'method': 'finger-color',
                'object_class': 'unknown',
                'hand_used': hand_used
            }
        
        print(f"      └─> ❌ Below threshold")
        return {'detected': False, 'confidence': best_score, 'method': 'finger-color'}
    
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
    
    def _detect_finger_state(self, hand_region: np.ndarray, full_frame: np.ndarray) -> float:
        """
        Method 1: MediaPipe Hands - Finger State Detection
        Detects if hand is "nắm lại" (fist) or "mở ra" (open).
        
        Returns:
            float: 1.0 if nắm lại (likely holding), 0.0 if mở ra
        """
        if hand_region.size == 0:
            return 0.0
        
        # Resize hand region if too small (MediaPipe works better with larger images)
        min_size = 128  # Minimum size for MediaPipe
        h, w = hand_region.shape[:2]
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            hand_region = cv2.resize(hand_region, (new_w, new_h))
        
        # Convert BGR to RGB for MediaPipe
        hand_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = self.hands.process(hand_rgb)
        
        if not results.multi_hand_landmarks:
            # No hand detected in region
            return 0.0
        
        # Get first hand (usually the most confident)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get key landmarks
        # MediaPipe Hands has 21 landmarks:
        # 0: Wrist
        # 4: Thumb tip
        # 8: Index finger tip
        # 12: Middle finger tip
        # 16: Ring finger tip
        # 20: Pinky tip
        
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Calculate distances from fingertips to wrist
        def distance(lm1, lm2):
            return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)
        
        thumb_dist = distance(thumb_tip, wrist)
        index_dist = distance(index_tip, wrist)
        middle_dist = distance(middle_tip, wrist)
        ring_dist = distance(ring_tip, wrist)
        pinky_dist = distance(pinky_tip, wrist)
        
        # Average distance
        avg_dist = (thumb_dist + index_dist + middle_dist + ring_dist + pinky_dist) / 5.0
        
        # Reference distance (wrist to middle finger MCP - knuckle)
        # Use middle finger MCP (landmark 9) as reference
        middle_mcp = hand_landmarks.landmark[9]
        reference_dist = distance(middle_mcp, wrist)
        
        # Normalize by reference distance
        if reference_dist > 0:
            normalized_dist = avg_dist / reference_dist
        else:
            normalized_dist = 1.0
        
        # If normalized distance is small → nắm lại (fist)
        # If normalized distance is large → mở ra (open)
        if normalized_dist < self.fist_threshold:
            # Nắm lại → likely holding
            finger_score = 1.0
            print(f"          Finger: NẮM LẠI (dist={normalized_dist:.3f} < {self.fist_threshold})")
        else:
            # Mở ra → not holding
            finger_score = 0.0
            print(f"          Finger: MỞ RA (dist={normalized_dist:.3f} >= {self.fist_threshold})")
        
        return finger_score
    
    def _detect_dominant_color(self, hand_region: np.ndarray) -> float:
        """
        Method 2: Dominant Color Detection
        Finds dominant color in hand region.
        If dominant color is NOT skin color → likely holding object.
        
        Returns:
            float: 1.0 if dominant color is NOT skin, 0.0 if skin
        """
        if hand_region.size == 0:
            return 0.0
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Reshape for K-means
        hsv_flat = hsv.reshape(-1, 3)
        
        # Simple K-means (using OpenCV)
        # Use fewer clusters for speed
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = min(self.dominant_color_k, len(hsv_flat))
        
        if len(hsv_flat) < k:
            return 0.0
        
        # Run K-means
        _, labels, centers = cv2.kmeans(
            hsv_flat.astype(np.float32),
            k,
            None,
            criteria,
            3,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Find dominant cluster (most pixels)
        unique, counts = np.unique(labels, return_counts=True)
        dominant_idx = unique[np.argmax(counts)]
        dominant_color = centers[dominant_idx]
        
        # Check if dominant color is skin color
        # Skin color in HSV: H ~0-20, S ~20-255, V ~70-255
        h, s, v = dominant_color
        
        is_skin = (0 <= h <= 20) and (20 <= s <= 255) and (70 <= v <= 255)
        
        if is_skin:
            # Dominant color is skin → not holding
            dominant_score = 0.0
            print(f"          Dominant: SKIN COLOR (H={h:.0f}, S={s:.0f}, V={v:.0f})")
        else:
            # Dominant color is NOT skin → likely holding
            dominant_score = 1.0
            print(f"          Dominant: NON-SKIN COLOR (H={h:.0f}, S={s:.0f}, V={v:.0f})")
        
        return dominant_score
    
    def _analyze_color_variance(self, hand_region: np.ndarray) -> float:
        """
        Method 3: Color Variance Analysis
        Calculates variance of pixel intensities.
        Skin: low variance (uniform color)
        Object: high variance (different colors)
        
        Returns:
            float: 1.0 if high variance (likely object), 0.0 if low variance (skin)
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
        
        # Calculate variance of pixel intensities
        variance = np.var(gray.astype(np.float32))
        
        # Normalize to 0-1 score
        # Skin: variance ~500-800
        # Object: variance >1200
        if variance >= self.color_variance_threshold:
            variance_score = 1.0
            print(f"          Variance: HIGH ({variance:.0f} >= {self.color_variance_threshold}) → likely object")
        else:
            # Linear mapping for intermediate values
            variance_score = min(1.0, variance / self.color_variance_threshold)
            print(f"          Variance: LOW ({variance:.0f} < {self.color_variance_threshold}) → likely skin")
        
        return variance_score
    
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
            cv2.circle(frame, (wx, wy), 5, (0, 0, 0), -1)
        
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
