"""
GrabNGo Vision Pipeline với SORT Tracker
Đơn giản và dễ cài đặt hơn StrongSORT
"""

import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms import functional as F
from datetime import datetime
from filterpy.kalman import KalmanFilter

# ===== SORT TRACKER (Nhúng trực tiếp, không cần clone repo) =====
class Sort:
    """
    SORT: A Simple, Online and Realtime Tracker
    Đơn giản hóa từ: https://github.com/abewley/sort
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        """
        Params:
          detections - [[x1,y1,x2,y2,score], ...] numpy array
        Returns:
          [[x1,y1,x2,y2,id], ...] numpy array
        """
        self.frame_count += 1
        
        # Lấy predicted locations từ các trackers hiện có
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :])
        
        # Tạo tracker mới cho unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            
            # Xóa dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Gán detections với tracked objects (cả hai đều là bounding boxes)
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self._linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Lọc matched với IoU thấp
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def _iou(self, bb_test, bb_gt):
        """Tính Intersection over Union (IoU)"""
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o
    
    def _linear_assignment(self, cost_matrix):
        """Hungarian algorithm"""
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))


class KalmanBoxTracker:
    """
    Kalman Filter cho tracking bounding box
    """
    count = 0
    
    def __init__(self, bbox):
        """Khởi tạo tracker với bbox [x1, y1, x2, y2, score]"""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """Update với observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """Predict location"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """Return current bounding box"""
        return self._convert_x_to_bbox(self.kf.x)
    
    def _convert_bbox_to_z(self, bbox):
        """Convert [x1,y1,x2,y2] to [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x, score=None):
        """Convert [x,y,s,r] back to [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# ===== MAIN PIPELINE =====
print("🚀 GrabNGo Vision Pipeline với SORT")
print("=" * 60)

# Kiểm tra device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Thiết bị: {device}")

# Tải Detection Model
print("\n📦 Đang tải Keypoint R-CNN...")
weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
detection_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
detection_model.to(device)
detection_model.eval()
print("✓ Detection model sẵn sàng!")

# Khởi tạo SORT Tracker
print("\n🎯 Khởi tạo SORT Tracker...")
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
print("✓ SORT sẵn sàng!")

# Mở video
print("\n" + "=" * 60)
print("CHỌN NGUỒN VIDEO:")
print("1. Webcam")
print("2. File video")
choice = input("Chọn (1/2): ").strip() or "1"

if choice == "2":
    video_path = input("Nhập đường dẫn video: ").strip()
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit(1)

print("✓ Video sẵn sàng!")
print("=" * 60)
print("🎮 ĐIỀU KHIỂN: Q=Thoát | SPACE=Pause | D=Debug")
print("=" * 60)

# Biến
frame_count = 0
paused = False
show_debug = True
track_colors = {}

def get_color(track_id):
    if track_id not in track_colors:
        np.random.seed(int(track_id))
        track_colors[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
    return track_colors[track_id]

# Vòng lặp chính
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = datetime.now()
        
        # Resize nếu cần
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        
        # Detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).to(device)
        
        with torch.no_grad():
            outputs = detection_model([image_tensor])
        
        # Chuẩn bị detections cho SORT: [[x1,y1,x2,y2,score], ...]
        detections = []
        keypoints_map = {}
        
        for i in range(len(outputs[0]['scores'])):
            score = outputs[0]['scores'][i].item()
            if score > 0.7:
                box = outputs[0]['boxes'][i].cpu().numpy()
                kp = outputs[0]['keypoints'][i].cpu().numpy()
                kp_scores = outputs[0]['keypoints_scores'][i].cpu().numpy()
                
                detections.append([box[0], box[1], box[2], box[3], score])
                keypoints_map[tuple(box.astype(int))] = {'points': kp, 'scores': kp_scores}
        
        # Update tracker
        if len(detections) > 0:
            tracks = tracker.update(np.array(detections))
        else:
            tracks = np.empty((0, 5))
        
        # Vẽ kết quả
        output_data = []
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            # Tìm keypoints
            box_key = (x1, y1, x2, y2)
            kp_data = keypoints_map.get(box_key)
            
            # Lưu output
            output_data.append({
                "timestamp": timestamp.isoformat(),
                "track_id": track_id,
                "box": [x1, y1, x2, y2],
                "keypoints": kp_data['points'].tolist() if kp_data else None
            })
            
            # Vẽ
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id}"
            cv2.rectangle(frame, (x1, y1-20), (x1+60, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Vẽ keypoints (cổ tay)
            if kp_data and show_debug:
                kp = kp_data['points']
                kp_scores = kp_data['scores']
                # 9=left_wrist, 10=right_wrist
                for idx in [9, 10]:
                    if kp_scores[idx] > 0.5:
                        cv2.circle(frame, (int(kp[idx][0]), int(kp[idx][1])), 
                                 5, (0, 255, 255), -1)
        
        # Debug info
        if show_debug:
            cv2.putText(frame, f"Frame: {frame_count}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # In ra console
        if output_data:
            print(f"Frame {frame_count}: {len(output_data)} tracks")
    
    # Hiển thị
    if paused:
        cv2.putText(frame, "PAUSED", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("GrabNGo - SORT Tracking", frame)
    
    # Xử lý phím
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('d'):
        show_debug = not show_debug

cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Hoàn tất! Đã xử lý {frame_count} frames")