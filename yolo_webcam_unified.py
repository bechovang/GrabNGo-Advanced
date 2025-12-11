import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from scipy.spatial.distance import cosine

class FeatureExtractor:
    """Trích xuất đặc điểm ngoại hình để ReID"""
    def __init__(self):
        self.feature_dim = 128
        
    def extract_color_histogram(self, image):
        """Trích xuất histogram màu sắc (HSV)"""
        if image.size == 0:
            return np.zeros(self.feature_dim)
        
        # Chuyển sang HSV để robust hơn với ánh sáng
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Tính histogram cho 3 kênh
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Kết hợp (96 features)
        color_features = np.concatenate([hist_h, hist_s, hist_v])
        
        return color_features
    
    def extract_texture_features(self, image):
        """Trích xuất đặc điểm texture đơn giản"""
        if image.size == 0:
            return np.zeros(32)
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tính gradient để capture texture
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Histogram của gradient
        hist_gx = cv2.calcHist([np.uint8(np.abs(grad_x))], [0], None, [16], [0, 256])
        hist_gy = cv2.calcHist([np.uint8(np.abs(grad_y))], [0], None, [16], [0, 256])
        
        hist_gx = cv2.normalize(hist_gx, hist_gx).flatten()
        hist_gy = cv2.normalize(hist_gy, hist_gy).flatten()
        
        texture_features = np.concatenate([hist_gx, hist_gy])
        
        return texture_features
    
    def extract_features(self, frame, bbox):
        """Trích xuất toàn bộ features từ bbox"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Đảm bảo bbox trong frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return np.zeros(self.feature_dim)
        
        # Resize để consistency
        person_img = cv2.resize(person_img, (64, 128))
        
        # Chia thành 3 vùng: đầu, thân, chân
        h = person_img.shape[0]
        upper = person_img[0:h//3, :]
        middle = person_img[h//3:2*h//3, :]
        lower = person_img[2*h//3:, :]
        
        # Trích xuất color từ mỗi vùng
        upper_color = self.extract_color_histogram(upper)[:32]
        middle_color = self.extract_color_histogram(middle)[:32]
        lower_color = self.extract_color_histogram(lower)[:32]
        
        # Trích xuất texture tổng thể
        texture = self.extract_texture_features(person_img)
        
        # Kết hợp features (32+32+32+32 = 128)
        features = np.concatenate([upper_color, middle_color, lower_color, texture])
        
        return features

class CustomerTracker:
    def __init__(self):
        self.customers = {}  # {track_id: customer_data}
        self.next_customer_id = 1
        self.active_tracks = set()
        self.gesture_history = defaultdict(list)
        self.logs = []
        
        # ReID components
        self.feature_extractor = FeatureExtractor()
        self.lost_customers = {}  # Lưu khách hàng bị mất track tạm thời
        self.customer_features = {}  # {customer_id: list of features}
        self.reid_threshold = 0.35  # Ngưỡng similarity để match
        self.max_lost_frames = 90  # 3 giây @ 30fps
        
    def add_customer(self, track_id, bbox, frame):
        """Ghi nhận khách hàng mới vào cửa hàng"""
        customer_id = f"CUST_{self.next_customer_id:04d}"
        self.next_customer_id += 1
        
        # Trích xuất features ngay từ đầu
        features = self.feature_extractor.extract_features(frame, bbox)
        
        self.customers[track_id] = {
            'id': customer_id,
            'entry_time': datetime.now(),
            'bbox_history': [bbox],
            'gestures': [],
            'items_detected': [],
            'exit_time': None,
            'suspicious_count': 0,
            'last_seen': time.time()
        }
        
        # Lưu features để ReID
        self.customer_features[customer_id] = [features]
        
        log_entry = {
            'customer_id': customer_id,
            'track_id': track_id,
            'event': 'ENTRY',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox
        }
        self.logs.append(log_entry)
        print(f"✅ Khách hàng mới: {customer_id} (Track ID: {track_id})")
        
        return customer_id
    
    def try_reid_customer(self, track_id, bbox, frame):
        """Thử nhận diện lại khách hàng bị mất track"""
        # Trích xuất features của người hiện tại
        current_features = self.feature_extractor.extract_features(frame, bbox)
        
        best_match_id = None
        best_similarity = 0
        
        # So sánh với các khách hàng bị mất gần đây
        for lost_track_id, lost_data in list(self.lost_customers.items()):
            customer_id = lost_data['customer_id']
            
            # Kiểm tra đã mất quá lâu chưa
            frames_lost = lost_data['frames_lost']
            if frames_lost > self.max_lost_frames:
                continue
            
            # Lấy features của khách hàng này
            if customer_id in self.customer_features:
                # So sánh với nhiều features gần đây nhất
                recent_features = self.customer_features[customer_id][-5:]  # 5 features gần nhất
                
                similarities = []
                for stored_features in recent_features:
                    # Tính cosine similarity (1 = giống nhau, 0 = khác nhau)
                    similarity = 1 - cosine(current_features, stored_features)
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_match_id = (lost_track_id, customer_id)
        
        # Nếu tìm thấy match tốt
        if best_match_id and best_similarity > self.reid_threshold:
            old_track_id, customer_id = best_match_id
            print(f"🔄 ReID Success: {customer_id} (Track {old_track_id} → {track_id}) "
                  f"[Similarity: {best_similarity:.2f}]")
            
            # Chuyển data từ old track sang new track
            self.customers[track_id] = self.customers[old_track_id]
            self.customers[track_id]['last_seen'] = time.time()
            del self.customers[old_track_id]
            
            # Xóa khỏi lost list
            if old_track_id in self.lost_customers:
                del self.lost_customers[old_track_id]
            
            return customer_id
        
        return None
    
    def update_customer(self, track_id, bbox, keypoints, items_held, frame):
        """Cập nhật thông tin khách hàng"""
        # Thử ReID nếu là track_id mới
        if track_id not in self.customers:
            reid_customer_id = self.try_reid_customer(track_id, bbox, frame)
            if not reid_customer_id:
                # Không match được, tạo mới
                self.add_customer(track_id, bbox, frame)
        
        customer = self.customers[track_id]
        customer['bbox_history'].append(bbox)
        customer['last_seen'] = time.time()
        
        # Cập nhật features định kỳ (mỗi 10 frames)
        if len(customer['bbox_history']) % 10 == 0:
            features = self.feature_extractor.extract_features(frame, bbox)
            customer_id = customer['id']
            
            if customer_id in self.customer_features:
                self.customer_features[customer_id].append(features)
                # Giữ tối đa 30 features gần nhất
                if len(self.customer_features[customer_id]) > 30:
                    self.customer_features[customer_id] = self.customer_features[customer_id][-30:]
        
        # Giữ lại 30 frames gần nhất
        if len(customer['bbox_history']) > 30:
            customer['bbox_history'] = customer['bbox_history'][-30:]
        
        # Phát hiện cử chỉ lấy hàng
        gesture = self.detect_grabbing_gesture(keypoints, bbox)
        if gesture:
            customer['gestures'].append({
                'type': gesture['type'] if isinstance(gesture, dict) else gesture,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'confidence': gesture['confidence'] if isinstance(gesture, dict) else 0.8
            })
            customer['suspicious_count'] += 1
        
        # Cập nhật vật phẩm đang cầm
        if items_held:
            for item in items_held:
                if item not in customer['items_detected']:
                    customer['items_detected'].append(item)
                    print(f"🛍️  {customer['id']}: Phát hiện cầm {item}")
        
        self.active_tracks.add(track_id)
    
    def detect_grabbing_gesture(self, keypoints, bbox):
        """Phát hiện cử chỉ lấy hàng"""
        if keypoints is None or len(keypoints) < 17:
            return None
        
        try:
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]
            left_shoulder = keypoints[5][:2]
            right_shoulder = keypoints[6][:2]
            
            if (keypoints[9][2] < 0.3 and keypoints[10][2] < 0.3):
                return None
            
            bbox_height = bbox[3] - bbox[1]
            bbox_width = bbox[2] - bbox[0]
            shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            # Cử chỉ 1: Tay với lên cao
            reaching_up = False
            if keypoints[9][2] > 0.3:
                if left_wrist[1] < shoulder_avg_y - bbox_height * 0.1:
                    reaching_up = True
            if keypoints[10][2] > 0.3:
                if right_wrist[1] < shoulder_avg_y - bbox_height * 0.1:
                    reaching_up = True
            
            # Cử chỉ 2: Tay với ra ngoài
            reaching_side = False
            body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            
            if keypoints[9][2] > 0.3:
                if left_wrist[0] < body_center_x - bbox_width * 0.3:
                    reaching_side = True
            if keypoints[10][2] > 0.3:
                if right_wrist[0] > body_center_x + bbox_width * 0.3:
                    reaching_side = True
            
            # Cử chỉ 3: Tay gập
            holding = False
            if keypoints[7][2] > 0.3 and keypoints[9][2] > 0.3:
                left_arm_angle = self.calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
                if 30 < left_arm_angle < 120:
                    holding = True
            if keypoints[8][2] > 0.3 and keypoints[10][2] > 0.3:
                right_arm_angle = self.calculate_arm_angle(right_shoulder, right_elbow, right_wrist)
                if 30 < right_arm_angle < 120:
                    holding = True
            
            if reaching_up:
                return {'type': 'REACHING_UP', 'confidence': 0.85}
            elif reaching_side:
                return {'type': 'REACHING_SIDE', 'confidence': 0.80}
            elif holding:
                return {'type': 'HOLDING_ITEM', 'confidence': 0.75}
                
        except Exception as e:
            return None
        
        return None
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """Tính góc gập tay"""
        v1 = np.array(elbow) - np.array(shoulder)
        v2 = np.array(wrist) - np.array(elbow)
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)
    
    def remove_customer(self, track_id):
        """Ghi nhận khách hàng rời khỏi cửa hàng"""
        if track_id in self.customers:
            customer = self.customers[track_id]
            customer['exit_time'] = datetime.now()
            
            duration = (customer['exit_time'] - customer['entry_time']).total_seconds()
            
            log_entry = {
                'customer_id': customer['id'],
                'track_id': track_id,
                'event': 'EXIT',
                'timestamp': customer['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'duration': f"{duration:.1f}s",
                'gestures_count': len(customer['gestures']),
                'items_detected': customer['items_detected'],
                'suspicious_count': customer['suspicious_count']
            }
            self.logs.append(log_entry)
            
            print(f"🚪 {customer['id']} rời đi - Thời gian: {duration:.1f}s, "
                  f"Cử chỉ: {len(customer['gestures'])}, "
                  f"Vật phẩm: {len(customer['items_detected'])}")
            
            if customer['suspicious_count'] > 3:
                print(f"⚠️  CẢNH BÁO: {customer['id']} có {customer['suspicious_count']} hành vi đáng chú ý!")
    
    def cleanup_inactive_tracks(self, current_tracks):
        """Xóa các track không còn active, nhưng lưu vào lost_customers trước"""
        inactive = self.active_tracks - current_tracks
        
        for track_id in inactive:
            if track_id in self.customers:
                # Chuyển sang lost_customers thay vì xóa ngay
                if track_id not in self.lost_customers:
                    self.lost_customers[track_id] = {
                        'customer_id': self.customers[track_id]['id'],
                        'lost_time': time.time(),
                        'frames_lost': 0
                    }
                    print(f"⏸️  {self.customers[track_id]['id']} (Track {track_id}) tạm mất track")
        
        # Tăng frame count cho các lost customers
        for track_id in list(self.lost_customers.keys()):
            self.lost_customers[track_id]['frames_lost'] += 1
            
            # Nếu mất quá lâu, xóa hẳn
            if self.lost_customers[track_id]['frames_lost'] > self.max_lost_frames:
                customer_id = self.lost_customers[track_id]['customer_id']
                print(f"❌ {customer_id} (Track {track_id}) mất track quá lâu, xác nhận rời đi")
                self.remove_customer(track_id)
                del self.lost_customers[track_id]
                if track_id in self.customers:
                    del self.customers[track_id]
        
        self.active_tracks = current_tracks.copy()
    
    def save_logs(self, filename='customer_logs.json'):
        """Lưu logs vào file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu logs vào {filename}")
    
    def is_near(self, person_box, object_box, threshold=100):
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
    pose_model = YOLO('yolov8n-pose.pt')
    detect_model = YOLO('yolov8n.pt')
    
    # Khởi tạo tracker với ReID
    tracker = CustomerTracker()
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    print("📹 Camera đã sẵn sàng!")
    print("💡 Nhấn 'q' để thoát, 'l' để xem logs, 's' để lưu logs, 'r' để xem ReID stats")
    print("=" * 70)
    
    frame_count = 0
    fps_list = []
    reid_success_count = 0
    
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect poses với tracking
            pose_results = pose_model.track(
                frame,
                persist=True,
                conf=0.5,
                iou=0.5,
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
            
            # Vẽ kết quả
            annotated_frame = pose_results[0].plot()
            
            # Xử lý tracking
            current_tracks = set()
            if pose_results[0].boxes is not None and pose_results[0].boxes.id is not None:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                track_ids = pose_results[0].boxes.id.cpu().numpy().astype(int)
                keypoints = pose_results[0].keypoints.data.cpu().numpy() if pose_results[0].keypoints is not None else None
                
                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    current_tracks.add(track_id)
                    
                    person_keypoints = keypoints[i] if keypoints is not None else None
                    
                    # Phát hiện vật phẩm
                    items_held = []
                    if object_results[0].boxes is not None:
                        obj_boxes = object_results[0].boxes.xyxy.cpu().numpy()
                        obj_classes = object_results[0].boxes.cls.cpu().numpy().astype(int)
                        obj_names = object_results[0].names
                        
                        shop_items = ['bottle', 'cup', 'bowl', 'banana', 'apple', 'orange', 
                                     'sandwich', 'cake', 'carrot', 'backpack', 'handbag', 'suitcase']
                        
                        for obj_box, obj_cls in zip(obj_boxes, obj_classes):
                            obj_name = obj_names[obj_cls]
                            if obj_name in shop_items:
                                if tracker.is_near(box, obj_box):
                                    items_held.append(obj_name)
                                    cv2.rectangle(annotated_frame, 
                                                (int(obj_box[0]), int(obj_box[1])), 
                                                (int(obj_box[2]), int(obj_box[3])), 
                                                (0, 255, 255), 2)
                                    cv2.putText(annotated_frame, obj_name,
                                              (int(obj_box[0]), int(obj_box[1]-10)),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Cập nhật tracker (pass frame để extract features)
                    tracker.update_customer(track_id, box, person_keypoints, items_held, frame)
                    
                    # Hiển thị Customer ID
                    if track_id in tracker.customers:
                        customer_id = tracker.customers[track_id]['id']
                        gesture_count = len(tracker.customers[track_id]['gestures'])
                        
                        # Vẽ với màu khác nếu là ReID
                        color = (0, 255, 0)  # Xanh lá
                        
                        cv2.putText(annotated_frame, 
                                  f"{customer_id} | G:{gesture_count}",
                                  (int(box[0]), int(box[1]-10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Cleanup inactive tracks
            tracker.cleanup_inactive_tracks(current_tracks)
            
            # Tính FPS
            fps = 1 / (time.time() - start_time)
            fps_list.append(fps)
            
            # Vẽ thông tin
            info_y = 30
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            info_y += 35
            cv2.putText(annotated_frame, f"Active: {len(current_tracks)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            info_y += 35
            cv2.putText(annotated_frame, f"Lost: {len(tracker.lost_customers)}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            info_y += 35
            cv2.putText(annotated_frame, f"Total: {tracker.next_customer_id - 1}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Hiển thị
            cv2.imshow('Smart Retail Tracking with ReID', annotated_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                print("\n" + "=" * 70)
                print("📋 LOGS GẦN ĐÂY:")
                for log in tracker.logs[-10:]:
                    print(json.dumps(log, indent=2, ensure_ascii=False))
                print("=" * 70 + "\n")
            elif key == ord('s'):
                tracker.save_logs()
            elif key == ord('r'):
                print("\n" + "=" * 70)
                print("🔄 ReID STATISTICS:")
                print(f"   Customers tracked: {len(tracker.customers)}")
                print(f"   Lost customers: {len(tracker.lost_customers)}")
                print(f"   Features stored: {sum(len(v) for v in tracker.customer_features.values())}")
                print("=" * 70 + "\n")
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                avg_fps = sum(fps_list[-30:]) / min(30, len(fps_list))
                print(f"Frame {frame_count}: FPS = {avg_fps:.1f}, "
                      f"Active: {len(current_tracks)}, Lost: {len(tracker.lost_customers)}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Dừng bởi người dùng")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tracker.save_logs()
        
        if fps_list:
            print("\n" + "=" * 70)
            print("📊 THỐNG KÊ CUỐI:")
            print(f"   Tổng frames: {frame_count}")
            print(f"   FPS trung bình: {sum(fps_list)/len(fps_list):.1f}")
            print(f"   Tổng khách hàng: {tracker.next_customer_id - 1}")
            print(f"   Tổng logs: {len(tracker.logs)}")
            print("=" * 70)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã đóng camera")

if __name__ == '__main__':
    main()