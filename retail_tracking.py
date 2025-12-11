import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime

class LightweightReID:
    """Lightweight ReID với features cải tiến: HOG + LAB Color + Spatial Pyramid"""
    
    def __init__(self):
        self.feature_dim = 512  # Tăng từ 256 lên 512 cho chính xác hơn
        
    def extract_features(self, frame, bbox):
        """Extract appearance features từ bbox với nhiều loại features"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None
            
            # Resize to standard size (128x256)
            person_crop = cv2.resize(person_crop, (128, 256))
            
            # Split into 3 regions: head, torso, legs (spatial pyramid)
            h = person_crop.shape[0]
            head = person_crop[:int(h*0.3), :]      # Upper 30%
            torso = person_crop[int(h*0.3):int(h*0.7), :]  # Middle 40%
            legs = person_crop[int(h*0.7):, :]      # Lower 30%
            
            # Extract features from each region
            features_list = []
            
            # 1. LAB Color features (tốt hơn HSV cho ReID)
            head_lab = self.extract_lab_color_features(head)
            torso_lab = self.extract_lab_color_features(torso)
            legs_lab = self.extract_lab_color_features(legs)
            features_list.extend([head_lab, torso_lab, legs_lab])  # 3 * 64 = 192
            
            # 2. HOG features (Histogram of Oriented Gradients) - tốt hơn Sobel
            head_hog = self.extract_hog_features(head)
            torso_hog = self.extract_hog_features(torso)
            legs_hog = self.extract_hog_features(legs)
            features_list.extend([head_hog, torso_hog, legs_hog])  # 3 * 64 = 192
            
            # 3. Texture features (LBP-like)
            head_texture = self.extract_texture_features(head)
            torso_texture = self.extract_texture_features(torso)
            legs_texture = self.extract_texture_features(legs)
            features_list.extend([head_texture, torso_texture, legs_texture])  # 3 * 32 = 96
            
            # 4. Edge density features
            head_edge = self.extract_edge_density(head)
            torso_edge = self.extract_edge_density(torso)
            legs_edge = self.extract_edge_density(legs)
            features_list.extend([head_edge, torso_edge, legs_edge])  # 3 * 16 = 48
            
            # Concatenate all features
            features = np.concatenate(features_list)  # Total: 528 dims
            
            # Trim or pad to exact dimension
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            else:
                features = np.pad(features, (0, self.feature_dim - len(features)), 'constant')
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            return None
    
    def extract_lab_color_features(self, img):
        """Extract LAB color histogram (tốt hơn HSV cho ReID)"""
        if img.size == 0:
            return np.zeros(64)
        
        # Convert to LAB color space (perceptually uniform)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Calculate histograms for L, A, B channels
        hist_l = cv2.calcHist([lab], [0], None, [32], [0, 100])  # L: 0-100
        hist_a = cv2.calcHist([lab], [1], None, [16], [0, 255])  # A: 0-255
        hist_b = cv2.calcHist([lab], [2], None, [16], [0, 255])  # B: 0-255
        
        # Normalize
        hist_l = cv2.normalize(hist_l, hist_l).flatten()
        hist_a = cv2.normalize(hist_a, hist_a).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        
        # Concatenate (32 + 16 + 16 = 64)
        color_feat = np.concatenate([hist_l, hist_a, hist_b])
        
        return color_feat
    
    def extract_hog_features(self, img):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        if img.size == 0:
            return np.zeros(64)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent HOG computation
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            gray = cv2.resize(gray, (16, 16))
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude and direction
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        direction = ((direction + 180) % 360).astype(np.uint8)
        
        # Create histogram of oriented gradients
        # Divide into 8x8 cells, 9 orientation bins
        h, w = magnitude.shape
        cell_size = 8
        n_cells_x = w // cell_size
        n_cells_y = h // cell_size
        
        # Simplified HOG: histogram of gradient directions weighted by magnitude
        hist = np.zeros(64)
        for i in range(0, min(n_cells_y * cell_size, h), cell_size):
            for j in range(0, min(n_cells_x * cell_size, w), cell_size):
                cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                cell_dir = direction[i:i+cell_size, j:j+cell_size]
                
                # Weighted histogram
                for mag, dir_val in zip(cell_mag.flatten(), cell_dir.flatten()):
                    bin_idx = int(dir_val / 360 * 64) % 64
                    hist[bin_idx] += mag
        
        # Normalize
        hist = hist / (np.linalg.norm(hist) + 1e-8)
        
        return hist
    
    def extract_texture_features(self, img):
        """Extract texture features using local binary patterns (simplified)"""
        if img.size == 0:
            return np.zeros(32)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistency
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            gray = cv2.resize(gray, (16, 16))
        
        # Compute local variance (texture measure)
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Histogram of local variance
        hist = cv2.calcHist([local_var.astype(np.uint8)], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def extract_edge_density(self, img):
        """Extract edge density features"""
        if img.size == 0:
            return np.zeros(16)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide into 4x4 grid and compute edge density per cell
        h, w = edges.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h // grid_h, w // grid_w
        
        edge_density = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                density = np.sum(cell > 0) / (cell_h * cell_w)
                edge_density.append(density)
        
        # Pad to 16 if needed
        edge_density = np.array(edge_density[:16])
        if len(edge_density) < 16:
            edge_density = np.pad(edge_density, (0, 16 - len(edge_density)), 'constant')
        
        return edge_density
    
    def compute_similarity(self, feat1, feat2):
        """Compute cosine similarity between two feature vectors"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        try:
            dot_product = np.dot(feat1, feat2)
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            
            similarity = dot_product / (norm1 * norm2 + 1e-8)
            return float(np.clip(similarity, 0, 1))  # Clamp to [0, 1]
        except:
            return 0.0


class CustomerTracker:
    """Tracker nâng cao sử dụng BoT-SORT + Lightweight ReID"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.customers = {}  # {track_id: customer_data}
        self.next_customer_id = 1
        self.active_tracks = set()
        self.previous_tracks = set()
        self.logs = []
        self.device = device
        
        # Track history để phát hiện người quay lại
        self.id_mapping = {}  # {old_track_id: customer_id}
        self.track_buffer = {}  # Buffer lưu tracks vừa mất
        self.max_buffer_time = 5.0  # 5 giây (tăng từ 3s để handle occlusion lâu hơn)
        
        # Lightweight ReID
        print("🔄 Đang khởi tạo Lightweight ReID...")
        self.reid_extractor = LightweightReID()
        print(f"✅ Lightweight ReID đã sẵn sàng (Feature dim: {self.reid_extractor.feature_dim})")
        
        # ReID parameters - Phase 1 Optimized
        self.base_reid_threshold = 0.35  # Base threshold (giảm để giảm false negatives)
        self.reid_threshold = 0.35  # Current threshold (có thể adaptive)
        self.reid_weight = 0.7  # Weight cho ReID score
        self.iou_weight = 0.3  # Weight cho IoU score
        
        # Adaptive threshold parameters
        self.use_adaptive_threshold = True
        self.min_threshold = 0.25  # Minimum threshold
        self.max_threshold = 0.50  # Maximum threshold
        
        # Performance metrics
        self.metrics = {
            'reid_success_count': 0,
            'reid_attempt_count': 0,
            'id_switch_count': 0,
            'total_customers': 0,
            'feature_similarities': [],
            'track_lifespans': [],
            'reid_times': []
        }
        
    def get_or_create_customer(self, track_id, bbox, frame_time, frame=None, current_features=None):
        """Lấy customer_id từ track_id, hoặc tạo mới nếu chưa có
        
        Args:
            track_id: Track ID từ YOLO
            bbox: Bounding box [x1, y1, x2, y2]
            frame_time: Thời gian frame hiện tại
            frame: Frame image để extract features (optional)
            current_features: Pre-extracted features (optional)
        """
        
        # Kiểm tra track_id đã có customer_id chưa
        if track_id in self.id_mapping:
            customer_id = self.id_mapping[track_id]
            if customer_id in self.customers:
                return customer_id, False  # Existing customer
        
        # Extract features cho track mới nếu chưa có
        if current_features is None and frame is not None:
            current_features = self.reid_extractor.extract_features(frame, bbox)
        
        # Tính feature quality
        current_feature_quality = self.calculate_feature_quality(current_features)
        
        # Kiểm tra có phải là người quay lại không (trong buffer)
        best_match = None
        best_score = 0.0
        
        for old_track_id, buffer_data in list(self.track_buffer.items()):
            if frame_time - buffer_data['lost_time'] > self.max_buffer_time:
                # Quá lâu, xóa khỏi buffer
                del self.track_buffer[old_track_id]
                continue
            
            # Tính IoU score với predicted position nếu có velocity
            old_bbox = buffer_data['last_bbox']
            time_since_lost = frame_time - buffer_data['lost_time']
            
            # Nếu có velocity, predict position
            if 'velocity' in buffer_data and buffer_data['velocity'] is not None:
                predicted_bbox = self.predict_bbox_position(
                    old_bbox, buffer_data['velocity'], time_since_lost
                )
                # Use predicted bbox for IoU, but also check original
                iou_score_predicted = self.calculate_iou(bbox, predicted_bbox)
                iou_score_original = self.calculate_iou(bbox, old_bbox)
                iou_score = max(iou_score_predicted, iou_score_original)  # Take better one
            else:
                iou_score = self.calculate_iou(bbox, old_bbox)
            
            # Tính ReID score nếu có features
            reid_score = 0.0
            old_feature_quality = 0.5
            if current_features is not None and 'features' in buffer_data:
                old_features = buffer_data['features']
                if old_features is not None:
                    reid_score = self.reid_extractor.compute_similarity(current_features, old_features)
                    old_feature_quality = self.calculate_feature_quality(old_features)
                    
                    # Track similarity for metrics
                    self.metrics['feature_similarities'].append(reid_score)
            
            # Combined score: weighted average
            combined_score = (self.reid_weight * reid_score + self.iou_weight * iou_score)
            
            # Lưu match tốt nhất
            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    'old_track_id': old_track_id,
                    'buffer_data': buffer_data,
                    'iou_score': iou_score,
                    'reid_score': reid_score,
                    'combined_score': combined_score,
                    'feature_quality': min(current_feature_quality, old_feature_quality)
                }
        
        # Tính adaptive threshold
        if best_match is not None:
            track_age = best_match['buffer_data'].get('track_age', 0.0)
            adaptive_threshold = self.get_adaptive_threshold(
                best_match['feature_quality'], track_age
            )
        else:
            adaptive_threshold = self.base_reid_threshold
        
        # Nếu có match tốt (adaptive threshold)
        if best_match is not None and best_match['combined_score'] >= adaptive_threshold:
            customer_id = best_match['buffer_data']['customer_id']
            self.id_mapping[track_id] = customer_id
            
            # Restore customer data
            if customer_id not in self.customers:
                self.customers[customer_id] = best_match['buffer_data']['customer_data'].copy()
            
            # Update metrics
            self.metrics['reid_success_count'] += 1
            self.metrics['reid_attempt_count'] += 1
            
            # Log với thông tin chi tiết
            print(f"🔄 ReID: {customer_id} (Track {best_match['old_track_id']} → {track_id}) "
                  f"[IoU:{best_match['iou_score']:.2f} ReID:{best_match['reid_score']:.3f} "
                  f"Score:{best_match['combined_score']:.3f} Thresh:{adaptive_threshold:.3f}]")
            
            # Xóa khỏi buffer
            del self.track_buffer[best_match['old_track_id']]
            
            # Log re-entry
            log_entry = {
                'customer_id': customer_id,
                'old_track_id': int(best_match['old_track_id']),
                'new_track_id': int(track_id),
                'event': 'RE_ENTRY',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'iou': float(best_match['iou_score']),
                'reid_score': float(best_match['reid_score']),
                'combined_score': float(best_match['combined_score']),
                'adaptive_threshold': float(adaptive_threshold),
                'feature_quality': float(best_match['feature_quality'])
            }
            self.logs.append(log_entry)
            
            return customer_id, True  # Re-identified
        
        # Track attempt (even if failed)
        if best_match is not None:
            self.metrics['reid_attempt_count'] += 1
        
        # Tạo customer mới
        customer_id = f"CUST_{self.next_customer_id:04d}"
        self.next_customer_id += 1
        self.id_mapping[track_id] = customer_id
        
        self.customers[customer_id] = {
            'id': customer_id,
            'track_id': track_id,
            'entry_time': datetime.now(),
            'bbox_history': deque(maxlen=30),
            'gestures': [],
            'items_detected': set(),
            'exit_time': None,
            'suspicious_count': 0,
            'last_seen': frame_time,
            'feature_history': deque(maxlen=10),  # Keep last 10 features (tăng từ 5)
            'velocity': None,  # Velocity for motion prediction [dx, dy]
            'track_age': 0.0  # Track age in seconds
        }
        
        log_entry = {
            'customer_id': customer_id,
            'track_id': int(track_id),
            'event': 'ENTRY',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bbox': [float(x) for x in bbox]
        }
        self.logs.append(log_entry)
        self.metrics['total_customers'] += 1
        print(f"✅ Khách hàng mới: {customer_id} (Track ID: {track_id})")
        
        return customer_id, True  # New customer
    
    def calculate_feature_quality(self, features):
        """Tính feature quality score dựa trên variance và magnitude"""
        if features is None or len(features) == 0:
            return 0.0
        
        try:
            # Magnitude (features không nên quá nhỏ)
            magnitude = np.linalg.norm(features)
            
            # Variance (features nên có variation, không phải all zeros)
            variance = np.var(features)
            
            # Quality score: combination of magnitude and variance
            # Normalize to [0, 1]
            quality = min(1.0, (magnitude * 0.5 + variance * 100) / 2.0)
            return float(quality)
        except:
            return 0.5  # Default medium quality
    
    def get_adaptive_threshold(self, feature_quality, track_age=0.0):
        """Tính adaptive threshold dựa trên feature quality và track age"""
        if not self.use_adaptive_threshold:
            return self.base_reid_threshold
        
        # Higher quality features → lower threshold (more confident)
        # Older tracks → lower threshold (more stable)
        quality_factor = 1.0 - (feature_quality * 0.2)  # Reduce threshold by up to 20%
        age_factor = min(0.1, track_age / 10.0)  # Reduce threshold for older tracks
        
        adaptive_threshold = self.base_reid_threshold * (1.0 - quality_factor - age_factor)
        
        # Clamp to min/max
        adaptive_threshold = max(self.min_threshold, min(self.max_threshold, adaptive_threshold))
        
        return adaptive_threshold
    
    def calculate_velocity(self, bbox_history, time_delta):
        """Tính velocity từ bbox history"""
        if len(bbox_history) < 2 or time_delta <= 0:
            return None
        
        try:
            # Lấy 2 bbox gần nhất
            bbox1 = bbox_history[-2]
            bbox2 = bbox_history[-1]
            
            # Tính center của mỗi bbox
            center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
            center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
            
            # Velocity = displacement / time
            velocity = (center2 - center1) / time_delta
            
            return velocity
        except:
            return None
    
    def predict_bbox_position(self, last_bbox, velocity, time_delta):
        """Dự đoán vị trí bbox dựa trên velocity"""
        if velocity is None or time_delta <= 0:
            return last_bbox
        
        try:
            # Tính center của last bbox
            center = np.array([(last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2])
            
            # Predict new center
            predicted_center = center + velocity * time_delta
            
            # Tính bbox size
            bbox_width = last_bbox[2] - last_bbox[0]
            bbox_height = last_bbox[3] - last_bbox[1]
            
            # Tạo predicted bbox
            predicted_bbox = [
                predicted_center[0] - bbox_width / 2,
                predicted_center[1] - bbox_height / 2,
                predicted_center[0] + bbox_width / 2,
                predicted_center[1] + bbox_height / 2
            ]
            
            return predicted_bbox
        except:
            return last_bbox
    
    def calculate_iou(self, box1, box2):
        """Tính Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def update_customer(self, track_id, bbox, keypoints, items_held, frame_time, frame=None):
        """Cập nhật thông tin khách hàng
        
        Args:
            track_id: Track ID từ YOLO
            bbox: Bounding box
            keypoints: Pose keypoints
            items_held: List items đang cầm
            frame_time: Thời gian frame
            frame: Frame image để extract features (optional)
        """
        # Extract features
        current_features = None
        if frame is not None:
            current_features = self.reid_extractor.extract_features(frame, bbox)
        
        customer_id, is_new = self.get_or_create_customer(
            track_id, bbox, frame_time, frame=frame, current_features=current_features
        )
        
        customer = self.customers[customer_id]
        
        # Calculate velocity từ bbox history
        if len(customer['bbox_history']) > 0:
            last_time = customer.get('last_seen', frame_time)
            time_delta = frame_time - last_time if last_time > 0 else 0.1
            velocity = self.calculate_velocity(customer['bbox_history'], time_delta)
            customer['velocity'] = velocity
        
        customer['bbox_history'].append(bbox)
        customer['last_seen'] = frame_time
        customer['track_id'] = track_id  # Update current track_id
        
        # Update track age
        if customer.get('entry_time'):
            entry_timestamp = customer['entry_time']
            if isinstance(entry_timestamp, datetime):
                entry_seconds = entry_timestamp.timestamp()
            else:
                entry_seconds = entry_timestamp
            customer['track_age'] = frame_time - entry_seconds
        else:
            customer['track_age'] = 0.0
        
        # Update feature history
        if current_features is not None:
            customer['feature_history'].append(current_features)
        
        # Phát hiện cử chỉ lấy hàng
        gesture = self.detect_grabbing_gesture(keypoints, bbox)
        if gesture:
            gesture_data = {
                'type': gesture['type'] if isinstance(gesture, dict) else gesture,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'confidence': gesture.get('confidence', 0.8) if isinstance(gesture, dict) else 0.8
            }
            customer['gestures'].append(gesture_data)
            customer['suspicious_count'] += 1
        
        # Cập nhật vật phẩm đang cầm
        if items_held:
            for item in items_held:
                if item not in customer['items_detected']:
                    customer['items_detected'].add(item)
                    print(f"🛍️  {customer['id']}: Phát hiện cầm {item}")
        
        self.active_tracks.add(track_id)
        return customer_id
    
    def detect_grabbing_gesture(self, keypoints, bbox):
        """Phát hiện cử chỉ lấy hàng dựa trên keypoints"""
        if keypoints is None or len(keypoints) < 17:
            return None
        
        try:
            # YOLO pose keypoints: 0-Nose, 5-L Shoulder, 6-R Shoulder, 
            # 7-L Elbow, 8-R Elbow, 9-L Wrist, 10-R Wrist
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            
            # Kiểm tra confidence
            if keypoints[9][2] < 0.3 and keypoints[10][2] < 0.3:
                return None
            
            bbox_height = bbox[3] - bbox[1]
            bbox_width = bbox[2] - bbox[0]
            shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            # Gesture 1: Reaching up (lấy hàng trên cao)
            reaching_up = False
            if keypoints[9][2] > 0.3 and left_wrist[1] < shoulder_avg_y - bbox_height * 0.1:
                reaching_up = True
            if keypoints[10][2] > 0.3 and right_wrist[1] < shoulder_avg_y - bbox_height * 0.1:
                reaching_up = True
            
            # Gesture 2: Reaching side (lấy hàng bên cạnh)
            body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            reaching_side = False
            if keypoints[9][2] > 0.3 and left_wrist[0] < body_center_x - bbox_width * 0.3:
                reaching_side = True
            if keypoints[10][2] > 0.3 and right_wrist[0] > body_center_x + bbox_width * 0.3:
                reaching_side = True
            
            # Gesture 3: Holding (tay gập)
            holding = False
            if keypoints[7][2] > 0.3 and keypoints[9][2] > 0.3:
                angle = self.calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
                if 30 < angle < 120:
                    holding = True
            if keypoints[8][2] > 0.3 and keypoints[10][2] > 0.3:
                angle = self.calculate_arm_angle(right_shoulder, right_elbow, right_wrist)
                if 30 < angle < 120:
                    holding = True
            
            if reaching_up:
                return {'type': 'REACHING_UP', 'confidence': 0.85}
            elif reaching_side:
                return {'type': 'REACHING_SIDE', 'confidence': 0.80}
            elif holding:
                return {'type': 'HOLDING_ITEM', 'confidence': 0.75}
                
        except Exception:
            return None
        
        return None
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """Tính góc gập tay"""
        v1 = np.array(elbow) - np.array(shoulder)
        v2 = np.array(wrist) - np.array(elbow)
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)
    
    def cleanup_inactive_tracks(self, current_tracks, frame_time):
        """Xử lý tracks không còn active"""
        inactive = self.previous_tracks - current_tracks
        
        for track_id in inactive:
            if track_id in self.id_mapping:
                customer_id = self.id_mapping[track_id]
                
                if customer_id in self.customers:
                    customer = self.customers[customer_id]
                    last_bbox = list(customer['bbox_history'])[-1] if customer['bbox_history'] else None
                    
                    # Get average features từ feature_history (tốt hơn single feature)
                    avg_features = None
                    if customer.get('feature_history') and len(customer['feature_history']) > 0:
                        avg_features = np.mean(list(customer['feature_history']), axis=0)
                    
                    # Calculate track lifespan
                    track_lifespan = 0.0
                    if customer.get('entry_time'):
                        entry_timestamp = customer['entry_time']
                        if isinstance(entry_timestamp, datetime):
                            entry_seconds = entry_timestamp.timestamp()
                        else:
                            entry_seconds = entry_timestamp
                        track_lifespan = frame_time - entry_seconds
                    self.metrics['track_lifespans'].append(track_lifespan)
                    
                    # Thêm vào buffer thay vì xóa ngay
                    self.track_buffer[track_id] = {
                        'customer_id': customer_id,
                        'customer_data': customer.copy(),
                        'last_bbox': last_bbox,
                        'lost_time': frame_time,
                        'features': avg_features,  # Lưu average features để ReID
                        'velocity': customer.get('velocity'),  # Lưu velocity để predict
                        'track_age': customer.get('track_age', 0.0)  # Track age
                    }
                    
                    print(f"⏸️  {customer_id} (Track {track_id}) tạm thời mất track")
        
        # Xóa các tracks trong buffer quá lâu (đã rời đi thật)
        for track_id in list(self.track_buffer.keys()):
            if frame_time - self.track_buffer[track_id]['lost_time'] > self.max_buffer_time:
                customer_id = self.track_buffer[track_id]['customer_id']
                self.finalize_customer_exit(customer_id, track_id)
                del self.track_buffer[track_id]
        
        self.previous_tracks = current_tracks.copy()
    
    def finalize_customer_exit(self, customer_id, track_id):
        """Xác nhận khách hàng đã rời đi"""
        if customer_id not in self.customers:
            return
        
        customer = self.customers[customer_id]
        customer['exit_time'] = datetime.now()
        
        duration = (customer['exit_time'] - customer['entry_time']).total_seconds()
        
        log_entry = {
            'customer_id': customer_id,
            'track_id': int(track_id),
            'event': 'EXIT',
            'timestamp': customer['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'duration': float(duration),
            'gestures_count': int(len(customer['gestures'])),
            'items_detected': list(customer['items_detected']),
            'suspicious_count': int(customer['suspicious_count'])
        }
        self.logs.append(log_entry)
        
        print(f"🚪 {customer_id} rời đi - Thời gian: {duration:.1f}s, "
              f"Cử chỉ: {len(customer['gestures'])}, "
              f"Vật phẩm: {len(customer['items_detected'])}")
        
        if customer['suspicious_count'] > 3:
            print(f"⚠️  CẢNH BÁO: {customer_id} có {customer['suspicious_count']} hành vi đáng chú ý!")
    
    def get_metrics_summary(self):
        """Lấy summary của metrics"""
        summary = {}
        
        # ReID success rate
        if self.metrics['reid_attempt_count'] > 0:
            summary['reid_success_rate'] = (
                self.metrics['reid_success_count'] / self.metrics['reid_attempt_count']
            )
        else:
            summary['reid_success_rate'] = 0.0
        
        # Average feature similarity
        if len(self.metrics['feature_similarities']) > 0:
            summary['avg_feature_similarity'] = np.mean(self.metrics['feature_similarities'])
            summary['min_feature_similarity'] = np.min(self.metrics['feature_similarities'])
            summary['max_feature_similarity'] = np.max(self.metrics['feature_similarities'])
        else:
            summary['avg_feature_similarity'] = 0.0
            summary['min_feature_similarity'] = 0.0
            summary['max_feature_similarity'] = 0.0
        
        # Track lifespan statistics
        if len(self.metrics['track_lifespans']) > 0:
            summary['avg_track_lifespan'] = np.mean(self.metrics['track_lifespans'])
            summary['min_track_lifespan'] = np.min(self.metrics['track_lifespans'])
            summary['max_track_lifespan'] = np.max(self.metrics['track_lifespans'])
        else:
            summary['avg_track_lifespan'] = 0.0
            summary['min_track_lifespan'] = 0.0
            summary['max_track_lifespan'] = 0.0
        
        # Counts
        summary['total_customers'] = self.metrics['total_customers']
        summary['reid_success_count'] = self.metrics['reid_success_count']
        summary['reid_attempt_count'] = self.metrics['reid_attempt_count']
        summary['id_switch_count'] = self.metrics['id_switch_count']
        
        return summary
    
    def print_metrics(self):
        """In metrics summary"""
        summary = self.get_metrics_summary()
        
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE METRICS SUMMARY")
        print("=" * 70)
        print(f"   Total Customers: {summary['total_customers']}")
        print(f"   ReID Attempts: {summary['reid_attempt_count']}")
        print(f"   ReID Successes: {summary['reid_success_count']}")
        if summary['reid_attempt_count'] > 0:
            print(f"   ReID Success Rate: {summary['reid_success_rate']*100:.1f}%")
        print(f"   ID Switches: {summary['id_switch_count']}")
        print(f"   Avg Track Lifespan: {summary['avg_track_lifespan']:.1f}s")
        if len(self.metrics['feature_similarities']) > 0:
            print(f"   Avg Feature Similarity: {summary['avg_feature_similarity']:.3f}")
            print(f"   Min/Max Similarity: {summary['min_feature_similarity']:.3f} / {summary['max_feature_similarity']:.3f}")
        print("=" * 70 + "\n")
    
    def save_logs(self, filename='customer_logs.json'):
        """Lưu logs vào file với JSON serialization đúng"""
        logs_serializable = []
        for log in self.logs:
            log_copy = {}
            for key, value in log.items():
                # Convert numpy/torch types to Python native types
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    log_copy[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    log_copy[key] = float(value)
                elif isinstance(value, np.ndarray):
                    log_copy[key] = value.tolist()
                elif isinstance(value, set):
                    log_copy[key] = list(value)
                elif isinstance(value, (datetime,)):
                    log_copy[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    log_copy[key] = value
            logs_serializable.append(log_copy)
        
        # Add metrics to logs
        metrics_summary = self.get_metrics_summary()
        metrics_serializable = {}
        for key, value in metrics_summary.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                metrics_serializable[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        output = {
            'logs': logs_serializable,
            'metrics': metrics_serializable,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu {len(logs_serializable)} logs và metrics vào {filename}")
    
    def is_near(self, person_box, object_box, threshold=150):
        """Kiểm tra vật phẩm có gần người không"""
        person_center = [(person_box[0] + person_box[2])/2, (person_box[1] + person_box[3])/2]
        object_center = [(object_box[0] + object_box[2])/2, (object_box[1] + object_box[3])/2]
        
        distance = np.sqrt((person_center[0] - object_center[0])**2 + 
                          (person_center[1] - object_center[1])**2)
        
        return distance < threshold


def main():
    # Khởi tạo
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Đang sử dụng: {'GPU' if device == 0 else 'CPU'}")
    
    # Load models
    print("📦 Đang load YOLO models...")
    
    # Sử dụng BoT-SORT tracker (tốt nhất cho occlusion handling)
    # Các tracker khả dụng: bytetrack, botsort
    pose_model = YOLO('yolov8n-pose.pt')
    detect_model = YOLO('yolov8n.pt')
    
    # Khởi tạo tracker
    tracker = CustomerTracker()
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    print("\n" + "=" * 70)
    print("🎯 SMART RETAIL TRACKING SYSTEM - PHASE 1 OPTIMIZED")
    print("   Tracker: BoT-SORT (Optimized for Occlusion)")
    print("   ReID: Lightweight (HOG + LAB + Texture)")
    print("   Features: ReID, Motion Prediction, Track Persistence")
    print("   Phase 1: Adaptive Threshold, Velocity Prediction, Metrics")
    print("=" * 70)
    print("\n📹 Camera đã sẵn sàng!")
    print("💡 Phím tắt:")
    print("   q - Thoát")
    print("   l - Xem 10 logs gần nhất")
    print("   s - Lưu logs vào file")
    print("   i - Xem thông tin tracker")
    print("   m - Xem performance metrics")
    print("=" * 70 + "\n")
    
    frame_count = 0
    fps_list = []
    start_time = time.time()
    reid_count = 0
    
    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # Detect poses với BoT-SORT tracking
            # tracker_config: bytetrack.yaml hoặc botsort.yaml
            pose_results = pose_model.track(
                frame,
                persist=True,
                tracker='botsort.yaml',  # Dùng BoT-SORT cho occlusion tốt hơn
                conf=0.5,
                iou=0.7,  # IoU threshold cao hơn để tránh ID switch
                device=device,
                verbose=False
            )
            
            # Detect objects
            object_results = detect_model(
                frame,
                conf=0.4,
                device=device,
                verbose=False
            )
            
            # Vẽ kết quả (tắt labels để chỉ hiển thị Customer ID)
            annotated_frame = pose_results[0].plot(labels=False)
            
            # Xử lý tracking
            current_tracks = set()
            if pose_results[0].boxes is not None and pose_results[0].boxes.id is not None:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                track_ids = pose_results[0].boxes.id.cpu().numpy().astype(int)
                keypoints = pose_results[0].keypoints.data.cpu().numpy() if pose_results[0].keypoints is not None else None
                
                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    current_tracks.add(track_id)
                    
                    person_keypoints = keypoints[i] if keypoints is not None else None
                    
                    # Phát hiện vật phẩm gần người
                    items_held = []
                    if object_results[0].boxes is not None:
                        obj_boxes = object_results[0].boxes.xyxy.cpu().numpy()
                        obj_classes = object_results[0].boxes.cls.cpu().numpy().astype(int)
                        obj_names = object_results[0].names
                        
                        shop_items = ['bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 
                                     'sandwich', 'cake', 'carrot', 'backpack', 'handbag', 
                                     'suitcase', 'cell phone', 'book']
                        
                        for obj_box, obj_cls in zip(obj_boxes, obj_classes):
                            obj_name = obj_names[obj_cls]
                            if obj_name in shop_items:
                                if tracker.is_near(box, obj_box):
                                    items_held.append(obj_name)
                                    # Vẽ box cho vật phẩm
                                    cv2.rectangle(annotated_frame, 
                                                (int(obj_box[0]), int(obj_box[1])), 
                                                (int(obj_box[2]), int(obj_box[3])), 
                                                (0, 255, 255), 2)
                                    cv2.putText(annotated_frame, obj_name,
                                              (int(obj_box[0]), int(obj_box[1]-10)),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Cập nhật tracker (truyền frame để extract ReID features)
                    customer_id = tracker.update_customer(track_id, box, person_keypoints, 
                                                         items_held, current_time, frame=frame)
                    
                    # Hiển thị Customer ID và thông tin
                    customer = tracker.customers.get(customer_id)
                    if customer:
                        gesture_count = len(customer['gestures'])
                        items_count = len(customer['items_detected'])
                        
                        # Màu: xanh lá nếu bình thường, vàng nếu có hành vi đáng chú ý
                        color = (0, 255, 255) if customer['suspicious_count'] > 2 else (0, 255, 0)
                        
                        info_text = f"{customer_id} | G:{gesture_count} I:{items_count}"
                        cv2.putText(annotated_frame, info_text,
                                  (int(box[0]), int(box[1]-10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Vẽ track history (trajectory)
                        if len(customer['bbox_history']) > 1:
                            points = []
                            for bbox in customer['bbox_history']:
                                center = (int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2))
                                points.append(center)
                            
                            for j in range(1, len(points)):
                                cv2.line(annotated_frame, points[j-1], points[j], color, 2)
            
            # Cleanup inactive tracks
            tracker.cleanup_inactive_tracks(current_tracks, current_time)
            
            # Tính FPS
            fps = 1 / (time.time() - frame_start)
            fps_list.append(fps)
            
            # Vẽ thông tin tổng quan
            info_y = 30
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            cv2.putText(annotated_frame, f"Active: {len(current_tracks)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            cv2.putText(annotated_frame, f"Buffered: {len(tracker.track_buffer)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            info_y += 30
            cv2.putText(annotated_frame, f"Total: {tracker.next_customer_id - 1}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị tracker type
            cv2.putText(annotated_frame, "BoT-SORT + Lightweight ReID", 
                       (10, annotated_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Hiển thị
            cv2.imshow('Smart Retail Tracking (BoT-SORT)', annotated_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                print("\n" + "=" * 70)
                print("📋 10 LOGS GẦN NHẤT:")
                for log in tracker.logs[-10:]:
                    print(json.dumps(log, indent=2, ensure_ascii=False))
                print("=" * 70 + "\n")
            elif key == ord('s'):
                tracker.save_logs()
            elif key == ord('i'):
                print("\n" + "=" * 70)
                print("📊 TRACKER INFO:")
                print(f"   Active customers: {len([c for c in tracker.customers.values() if c['exit_time'] is None])}")
                print(f"   Buffered tracks: {len(tracker.track_buffer)}")
                print(f"   Total customers served: {tracker.next_customer_id - 1}")
                print(f"   Total logs: {len(tracker.logs)}")
                print(f"   ID mappings: {len(tracker.id_mapping)}")
                print("=" * 70)
                tracker.print_metrics()
            elif key == ord('m'):
                # Show metrics only
                tracker.print_metrics()
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                avg_fps = sum(fps_list[-30:]) / min(30, len(fps_list))
                print(f"Frame {frame_count}: FPS = {avg_fps:.1f}, "
                      f"Active: {len(current_tracks)}, Buffered: {len(tracker.track_buffer)}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Dừng bởi người dùng")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Finalize tất cả customers còn lại
        for track_id, buffer_data in tracker.track_buffer.items():
            customer_id = buffer_data['customer_id']
            tracker.finalize_customer_exit(customer_id, track_id)
        
        tracker.save_logs()
        
        if fps_list:
            total_time = time.time() - start_time
            print("\n" + "=" * 70)
            print("📊 THỐNG KÊ CUỐI:")
            print(f"   Tổng frames: {frame_count}")
            print(f"   FPS trung bình: {sum(fps_list)/len(fps_list):.1f}")
            print(f"   Tổng thời gian: {total_time:.1f}s")
            print(f"   Tổng khách hàng: {tracker.next_customer_id - 1}")
            print(f"   Tổng logs: {len(tracker.logs)}")
            re_entries = len([log for log in tracker.logs if log.get('event') == 'RE_ENTRY'])
            print(f"   Số lần ReID thành công: {re_entries}")
            print("=" * 70)
            # Print final metrics
            tracker.print_metrics()
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã đóng camera")

if __name__ == '__main__':
    main()