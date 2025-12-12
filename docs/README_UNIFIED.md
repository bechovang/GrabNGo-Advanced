# YOLO Webcam Unified Script

This project now uses a single unified script that replaces both `yolo_webcam.py` and `yolo_webcam_for_cpu.py`.

## Features

- Supports both GPU and CPU inference
- Adjustable resolution, FPS, and model parameters
- Real-time FPS display
- Statistics logging
- Screenshot capability
- Command-line arguments for customization

## Usage

### Basic usage (auto-detects GPU if available):
```bash
python yolo_webcam_unified.py
```

### Using GPU specifically:
```bash
python yolo_webcam_unified.py --device gpu
```

### Using CPU specifically:
```bash
python yolo_webcam_unified.py --device cpu
```

### With custom parameters:
```bash
python yolo_webcam_unified.py --device gpu --camera 1 --width 1280 --height 720 --fps 30 --model yolov8m-pose.pt --conf 0.7
```

### Available arguments:
- `--device`: Device to use (auto, gpu, cpu)
- `--camera`: Camera index (default: 0)
- `--width`: Camera frame width (default: 640)
- `--height`: Camera frame height (default: 480)
- `--fps`: Camera frame rate (default: 30)
- `--model`: YOLO model to use (default: yolov8n-pose.pt)
- `--conf`: Confidence threshold (default: 0.5)
- `--imgsz`: Image size for inference (default: 640)
- `--display-fps`: Show FPS counter on screen