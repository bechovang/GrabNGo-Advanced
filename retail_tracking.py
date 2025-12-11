import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import defaultdict, deque
import json
from datetime import datetime

class CustomerTracker:
    """Tracker nâng cao sử dụng BoT-SORT/ByteTrack có sẵn trong Ultralytics"""
    def __init__(self):
        self.customers = {}  # {track_id: customer_data}
        self.next_customer_id = 1
        self.active_tracks = set()
        self.previous_tracks = set()
        self.logs = []
        
        # Track history để phát hiện người quay lại
        self.id_mapping = {}  # {old_track_id: customer_id}
        self.track_buffer = {}  # Buffer lưu tracks vừa mất
        self.max_buffer_time = 3.0  # 3 giây
        
    def get_or_create_customer(self, track_id, bbox, frame_time):
        """Lấy customer_id từ track_id, hoặc tạo mới nếu chưa có"""
        
        # Kiểm tra track_id đã có customer_id chưa
        if track_id in self.id_mapping:
            customer_id = self.id_mapping[track_id]
            if customer_id in self.customers:
                return customer_id, False  # Existing customer
        
        # Kiểm tra có phải là người quay lại không (trong buffer)
        for old_track_id, buffer_data in list(self.track_buffer.items()):
            if frame_time - buffer_data['lost_time'] > self.max_buffer_time:
                # Quá lâu, xóa khỏi buffer
                del self.track_buffer[old_track_id]
                continue
            
            # Kiểm tra vị trí có gần không (người quay lại thường ở gần đó)
            old_bbox = buffer_data['last_bbox']
            iou = self.calculate_iou(bbox, old_bbox)
            
            # Nếu IoU > 0.3 trong vòng 3 giây → có thể là người quay lại
            if iou > 0.3:
                customer_id = buffer_data['customer_id']
                self.id_mapping[track_id] = customer_id
                
                # Restore customer data
                if customer_id not in self.customers:
                    self.customers[customer_id] = buffer_data['customer_data']
                
                print(f"🔄 Nhận diện lại: {customer_id} (Track {old_track_id} → {track_id}) [IoU: {iou:.2f}]")
                del self.track_buffer[old_track_id]
                
                # Log re-entry
                log_entry = {
                    'customer_id': customer_id,
                    'old_track_id': old_track_id,
                    'new_track_id': track_id,
                    'event': 'RE_ENTRY',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'iou': f"{iou:.2f}"
                }
                self.logs.append(log_entry)
                
                return customer_id, True  # Re-identified
        
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
            'last_seen': frame_time
        }
        
        log_entry = {
            'customer_id': customer_id,
            'track_id': track_id,
            'event': 'ENTRY',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bbox': [float(x) for x in bbox]
        }
        self.logs.append(log_entry)
        print(f"✅ Khách hàng mới: {customer_id} (Track ID: {track_id})")
        
        return customer_id, True  # New customer
    
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
    
    def update_customer(self, track_id, bbox, keypoints, items_held, frame_time):
        """Cập nhật thông tin khách hàng"""
        customer_id, is_new = self.get_or_create_customer(track_id, bbox, frame_time)
        
        customer = self.customers[customer_id]
        customer['bbox_history'].append(bbox)
        customer['last_seen'] = frame_time
        customer['track_id'] = track_id  # Update current track_id
        
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
                    
                    # Thêm vào buffer thay vì xóa ngay
                    self.track_buffer[track_id] = {
                        'customer_id': customer_id,
                        'customer_data': customer.copy(),
                        'last_bbox': last_bbox,
                        'lost_time': frame_time
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
            'track_id': track_id,
            'event': 'EXIT',
            'timestamp': customer['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'duration': f"{duration:.1f}s",
            'gestures_count': len(customer['gestures']),
            'items_detected': list(customer['items_detected']),
            'suspicious_count': customer['suspicious_count']
        }
        self.logs.append(log_entry)
        
        print(f"🚪 {customer_id} rời đi - Thời gian: {duration:.1f}s, "
              f"Cử chỉ: {len(customer['gestures'])}, "
              f"Vật phẩm: {len(customer['items_detected'])}")
        
        if customer['suspicious_count'] > 3:
            print(f"⚠️  CẢNH BÁO: {customer_id} có {customer['suspicious_count']} hành vi đáng chú ý!")
    
    def save_logs(self, filename='customer_logs.json'):
        """Lưu logs vào file"""
        logs_serializable = []
        for log in self.logs:
            log_copy = log.copy()
            # Convert set to list if needed
            if 'items_detected' in log_copy and isinstance(log_copy['items_detected'], set):
                log_copy['items_detected'] = list(log_copy['items_detected'])
            logs_serializable.append(log_copy)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(logs_serializable, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu {len(logs_serializable)} logs vào {filename}")
    
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
    print("🎯 SMART RETAIL TRACKING SYSTEM")
    print("   Tracker: BoT-SORT (Optimized for Occlusion)")
    print("   Features: ReID, Motion Prediction, Track Persistence")
    print("=" * 70)
    print("\n📹 Camera đã sẵn sàng!")
    print("💡 Phím tắt:")
    print("   q - Thoát")
    print("   l - Xem 10 logs gần nhất")
    print("   s - Lưu logs vào file")
    print("   i - Xem thông tin tracker")
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
                    
                    # Cập nhật tracker
                    customer_id = tracker.update_customer(track_id, box, person_keypoints, 
                                                         items_held, current_time)
                    
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
            cv2.putText(annotated_frame, "Tracker: BoT-SORT", 
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
                print("=" * 70 + "\n")
            
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
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã đóng camera")

if __name__ == '__main__':
    main()