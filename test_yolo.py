from ultralytics import YOLO 
model = YOLO('yolov8n-pose.pt') 
model.predict(source='https://ultralytics.com/images/bus.jpg', device=0, save=True, show=True) 
