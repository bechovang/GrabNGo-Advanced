"""
Holding Detection Module
Detects if a person is holding an object (bottle or snack bag)
Uses Hybrid approach: Object-based detection + Hand state fallback
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict, List


class HoldingDetector:
    """
    Detects if a person is holding an object using hand keypoints and object detection.
    Optimized for medium-sized objects (bottles, snack bags).
    """
    
    def __init__(self):
        # Object detection settings
        self.bottle_classes = ['bottle', 'cup', 'wine glass']
        self.object_confidence_min = 0.25  # Lenient for bags
        self.medium_object_size_range = (8000, 30000)  # px²
        
        # Hand box estimation
        self.hand_size_ratio = 0.08  # 8% of person height
        self.hand_extension_vertical = 1.5  # Extend down for bottles
        
        # IoU matching
        self.iou_threshold = 0.12  # Fixed for medium objects
        
        # Temporal smoothing
        self.frames_to_confirm_holding = 3  # ~0.1s at 30fps
        self.frames_to_confirm_release = 5  # ~0.17s
        self.decay_rate = 0.5
        
        # Hand state fallback
        self.min_hand_state_score = 0.6
        self.holding_zone_y_range = (0.3, 0.8)  # Relative to person height
        self.two_hand_max_distance = 150  # px
        
        # Keypoint confidence
        self.min_wrist_confidence = 0.3
        
        # COCO keypoint indices
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        
        # Holding state tracking
        self.holding_states = {}  # {customer_id: holding_state}
    
    def detect_holding(self, 
                      customer_id: str,
                      person_bbox: np.ndarray,
                      keypoints: np.ndarray,
                      detected_objects: List[Dict],
                      frame: np.ndarray) -> Dict:
        """
        Main detection method. Returns holding status for a person.
        
        Args:
            customer_id: Customer identifier
            person_bbox: [x1, y1, x2, y2]
            keypoints: YOLO pose keypoints [17, 3] (x, y, confidence)
            detected_objects: List of detected objects with bbox and class
            frame: Current frame (for debugging/visualization)
            
        Returns:
            Dict with:
                - is_holding: bool
                - confidence: float
                - method: str ('object-based', 'hand-state', 'none')
                - object_class: str or 'unknown'
                - hand_used: str ('left', 'right', 'both', 'unknown')
        """
        # Initialize holding state for new customer
        if customer_id not in self.holding_states:
            self.holding_states[customer_id] = {
                'frames_holding': 0,
                'frames_not_holding': 0,
                'is_holding': False,
                'wrist_history': deque(maxlen=10)
            }
        
        state = self.holding_states[customer_id]
        
        # Try Method 1: Object-based detection
        result = self._detect_with_objects(person_bbox, keypoints, detected_objects)
        
        # If no object found, try Method 2: Hand state fallback
        if not result['detected']:
            result = self._detect_hand_state(person_bbox, keypoints, state)
        
        # Apply temporal smoothing
        result = self._apply_temporal_smoothing(customer_id, result)
        
        return result
    
    def _detect_with_objects(self, 
                            person_bbox: np.ndarray,
                            keypoints: np.ndarray,
                            detected_objects: List[Dict]) -> Dict:
        """Object-based detection using Hand Box + IoU."""
        
        # Extract hand keypoints
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        
        # DEBUG
        print(f"      └─> Wrist confidence: L={left_wrist[2]:.2f}, R={right_wrist[2]:.2f}")
        
        # Check keypoint confidence
        if left_wrist[2] < self.min_wrist_confidence and right_wrist[2] < self.min_wrist_confidence:
            print(f"      └─> ❌ Wrist confidence too low (need ≥{self.min_wrist_confidence})")
            return {'detected': False, 'confidence': 0.0, 'method': 'none'}
        
        # Create hand boxes
        person_height = person_bbox[3] - person_bbox[1]
        left_hand_box = self._create_hand_box(left_wrist, person_height) if left_wrist[2] >= self.min_wrist_confidence else None
        right_hand_box = self._create_hand_box(right_wrist, person_height) if right_wrist[2] >= self.min_wrist_confidence else None
        
        # Filter relevant objects
        relevant_objects = self._filter_relevant_objects(detected_objects, person_bbox)
        
        # DEBUG
        print(f"      └─> Filtered {len(relevant_objects)}/{len(detected_objects)} relevant objects")
        if len(relevant_objects) > 0:
            print(f"          Objects: {[obj['class'] for obj in relevant_objects]}")
        
        # Check IoU for each object
        best_match = None
        best_iou = 0.0
        best_hand = 'unknown'
        
        for obj in relevant_objects:
            obj_box = obj['bbox']
            
            # Check left hand
            if left_hand_box is not None:
                iou = self._calculate_iou(left_hand_box, obj_box)
                print(f"          L_hand IoU with {obj['class']}: {iou:.3f} (threshold: {self.iou_threshold})")
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = obj
                    best_hand = 'left'
            
            # Check right hand
            if right_hand_box is not None:
                iou = self._calculate_iou(right_hand_box, obj_box)
                print(f"          R_hand IoU with {obj['class']}: {iou:.3f} (threshold: {self.iou_threshold})")
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = obj
                    best_hand = 'right'
        
        if best_match:
            print(f"      └─> ✅ Object-based match: {best_match['class']} (IoU={best_iou:.3f}, hand={best_hand})")
            return {
                'detected': True,
                'confidence': best_iou,
                'method': 'object-based',
                'object_class': best_match.get('class', 'unknown'),
                'hand_used': best_hand
            }
        
        print(f"      └─> ❌ No object match, trying hand-state fallback...")
        return {'detected': False, 'confidence': 0.0, 'method': 'none'}
    
    def _detect_hand_state(self,
                          person_bbox: np.ndarray,
                          keypoints: np.ndarray,
                          state: Dict) -> Dict:
        """Hand state fallback detection."""
        
        score = 0.0
        
        # Extract keypoints
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_elbow = keypoints[self.LEFT_ELBOW]
        right_elbow = keypoints[self.RIGHT_ELBOW]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        
        # Feature 1: Arm angle (bent arm suggests holding)
        if left_wrist[2] >= 0.3 and left_elbow[2] >= 0.3 and left_shoulder[2] >= 0.3:
            angle = self._calculate_angle(left_shoulder[:2], left_elbow[:2], left_wrist[:2])
            if 60 < angle < 150:
                score += 0.3
        
        if right_wrist[2] >= 0.3 and right_elbow[2] >= 0.3 and right_shoulder[2] >= 0.3:
            angle = self._calculate_angle(right_shoulder[:2], right_elbow[:2], right_wrist[:2])
            if 60 < angle < 150:
                score += 0.3
        
        # Feature 2: Hand in holding zone
        person_height = person_bbox[3] - person_bbox[1]
        person_top = person_bbox[1]
        
        zone_y_min = person_top + person_height * self.holding_zone_y_range[0]
        zone_y_max = person_top + person_height * self.holding_zone_y_range[1]
        
        if left_wrist[2] >= 0.3 and zone_y_min <= left_wrist[1] <= zone_y_max:
            score += 0.15
        
        if right_wrist[2] >= 0.3 and zone_y_min <= right_wrist[1] <= zone_y_max:
            score += 0.15
        
        # Feature 3: Two-hand proximity (holding bag with both hands)
        if left_wrist[2] >= 0.3 and right_wrist[2] >= 0.3:
            distance = np.sqrt((left_wrist[0] - right_wrist[0])**2 + 
                             (left_wrist[1] - right_wrist[1])**2)
            y_diff = abs(left_wrist[1] - right_wrist[1])
            
            if distance < self.two_hand_max_distance and y_diff < 50:
                score += 0.2
        
        # Feature 4: Wrist stability
        if left_wrist[2] >= 0.3:
            state['wrist_history'].append(left_wrist[:2])
        elif right_wrist[2] >= 0.3:
            state['wrist_history'].append(right_wrist[:2])
        
        if len(state['wrist_history']) >= 5:
            variance = np.var(state['wrist_history'], axis=0).mean()
            if variance < 20:  # Stable wrist
                score += 0.2
        
        # Determine hand used
        hand_used = 'unknown'
        if left_wrist[2] >= 0.3 and right_wrist[2] >= 0.3:
            hand_used = 'both'
        elif left_wrist[2] >= 0.3:
            hand_used = 'left'
        elif right_wrist[2] >= 0.3:
            hand_used = 'right'
        
        # DEBUG
        print(f"      └─> Hand-state score: {score:.2f} (threshold: {self.min_hand_state_score})")
        
        if score >= self.min_hand_state_score:
            print(f"      └─> ✅ Hand-state match!")
            return {
                'detected': True,
                'confidence': score,
                'method': 'hand-state',
                'object_class': 'unknown',
                'hand_used': hand_used
            }
        
        print(f"      └─> ❌ Hand-state below threshold")
        return {'detected': False, 'confidence': score, 'method': 'none'}
    
    def _create_hand_box(self, wrist_keypoint: np.ndarray, person_height: float) -> np.ndarray:
        """Create adaptive hand bounding box around wrist."""
        wx, wy = wrist_keypoint[0], wrist_keypoint[1]
        
        hand_size = person_height * self.hand_size_ratio
        
        x1 = wx - hand_size
        y1 = wy - hand_size
        x2 = wx + hand_size
        y2 = wy + hand_size * self.hand_extension_vertical
        
        return np.array([x1, y1, x2, y2])
    
    def _filter_relevant_objects(self, 
                                detected_objects: List[Dict],
                                person_bbox: np.ndarray) -> List[Dict]:
        """Filter objects that could be bottles or bags."""
        relevant = []
        
        # Expand person bbox for interaction zone
        interaction_margin = 100
        interaction_box = [
            person_bbox[0] - interaction_margin,
            person_bbox[1],
            person_bbox[2] + interaction_margin,
            person_bbox[3]
        ]
        
        # DEBUG
        print(f"      └─> Filtering from {len(detected_objects)} objects")
        
        for obj in detected_objects:
            obj_bbox = obj['bbox']
            obj_class = obj.get('class', '')
            obj_conf = obj.get('confidence', 0.0)
            obj_area = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
            
            # DEBUG: Show each object
            print(f"          Checking: {obj_class} (conf={obj_conf:.2f}, area={obj_area:.0f})")
            
            # Skip if outside interaction zone
            if not self._boxes_overlap(interaction_box, obj_bbox):
                print(f"            ❌ Outside interaction zone")
                continue
            
            # Check if it's a bottle
            if obj_class in self.bottle_classes and obj_conf >= self.object_confidence_min:
                print(f"            ✓ Bottle match!")
                relevant.append(obj)
                continue
            
            # Check if it's a medium-sized object (could be bag)
            if (self.medium_object_size_range[0] <= obj_area <= self.medium_object_size_range[1] 
                and obj_conf >= 0.2):
                print(f"            ✓ Medium object match!")
                relevant.append(obj)
            else:
                print(f"            ❌ Not bottle and not medium-sized")
        
        return relevant
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection + 1e-8
        
        return intersection / union
    
    def _boxes_overlap(self, box1, box2) -> bool:
        """Quick check if two boxes overlap."""
        return not (box1[2] < box2[0] or box1[0] > box2[2] or
                   box1[3] < box2[1] or box1[1] > box2[3])
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
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
            print(f"      └─> ✅ CONFIRMED HOLDING!")
        elif state['frames_not_holding'] >= self.frames_to_confirm_release:
            state['is_holding'] = False
            detection_result['is_holding'] = False
            print(f"      └─> ❌ CONFIRMED NOT HOLDING")
        else:
            # Keep previous state during transition
            detection_result['is_holding'] = state['is_holding']
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
        
        # Draw hand keypoints
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        
        if left_wrist[2] >= 0.3:
            cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 5, (255, 0, 0), -1)
        
        if right_wrist[2] >= 0.3:
            cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 0, 255), -1)
        
        # Draw hand boxes
        if left_wrist[2] >= 0.3:
            person_height = person_bbox[3] - person_bbox[1]
            hand_box = self._create_hand_box(left_wrist, person_height)
            hx1, hy1, hx2, hy2 = map(int, hand_box)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)
        
        if right_wrist[2] >= 0.3:
            person_height = person_bbox[3] - person_bbox[1]
            hand_box = self._create_hand_box(right_wrist, person_height)
            hx1, hy1, hx2, hy2 = map(int, hand_box)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 1)
        
        # Draw holding status
        status_text = f"Holding: {holding_result.get('is_holding', False)}"
        method_text = f"Method: {holding_result.get('method', 'none')}"
        conf_text = f"Conf: {holding_result.get('confidence', 0.0):.2f}"
        
        cv2.putText(frame, status_text, (x1, y1 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, method_text, (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
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

