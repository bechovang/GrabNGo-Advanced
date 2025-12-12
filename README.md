# Smart Retail Tracking System

A production-ready retail customer tracking system using YOLO pose estimation, BoT-SORT tracking, and lightweight ReID (Re-Identification) for robust customer tracking across occlusions.

## Features

- **BoT-SORT Tracking**: Advanced multi-object tracking with occlusion handling
- **Lightweight ReID**: Appearance-based re-identification using LAB color, HOG, texture, and edge features
- **Manual Confirmation System**: Intelligent validation and manual confirmation for new customer IDs
- **Holding Detection**: Detects when customers are holding objects (bottles, bags, etc.)
- **Real-time Visualization**: Live tracking with trajectory visualization and status indicators

## Project Structure

```
GrabNGo-Advanced/
├── src/                    # Core source code
│   ├── main_tracker.py     # Main tracking system
│   └── holding_detector.py # Holding detection module
├── config/                 # Configuration files
│   └── botsort_reid.yaml   # BoT-SORT tracker configuration
├── docs/                   # Detailed documentation
│   ├── HOLDING_DETECTION.md
│   └── MANUAL_CONFIRMATION.md
├── main.py                 # Main entry point
└── requirements.txt        # Python dependencies
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download YOLO models** (if not already present):
   - `yolo11n-pose.pt` - Pose estimation model
   - `yolo11n-cls.pt` - Classification model (for ReID)

## Usage

### Basic Usage

Run the main tracking system:
```bash
python main.py
```

### Keyboard Controls

- `q` - Quit the application
- `c` - Confirm selected pending track (if validated)
- `1-9` - Select pending track by number
- `s` - Save tracking logs to file
- `i` - Display tracker statistics

### Configuration

Edit `config/botsort_reid.yaml` to adjust tracking parameters:
- `track_buffer`: Frames to keep lost tracks (default: 300 frames ≈ 10s at 30fps)
- `appearance_thresh`: ReID similarity threshold (default: 0.1)
- `proximity_thresh`: Minimum IoU for ReID consideration (default: 0.2)

## System Overview

### Tracking Flow

1. **Detection**: YOLO pose model detects people in frame
2. **Tracking**: BoT-SORT assigns track IDs and handles occlusions
3. **ReID**: Lightweight ReID extracts appearance features (LAB, HOG, texture, edge)
4. **Validation**: New tracks are validated (samples, confidence, feature quality)
5. **Confirmation**: Validated tracks can be manually confirmed to create customer IDs
6. **Re-identification**: Lost tracks are matched using appearance features

### Manual Confirmation System

New detections start as **PENDING** tracks and must pass validation before confirmation:

**Validation Requirements:**
- ≥5 feature samples collected
- ≥30% valid features
- Average confidence ≥0.5
- Feature consistency check
- Upper body visible (head/torso keypoints)
- Legs visible (ankle keypoints)

**Visual Indicators:**
- **Orange box**: Collecting data (< 80% validation)
- **Green box**: Ready for confirmation (≥ 80% validation)
- **Green box with ID**: Confirmed customer (e.g., `CUST_0001`)

### Holding Detection

The system can detect when customers are holding objects using:
- MediaPipe Hands (finger state detection)
- Dominant color analysis
- Color variance analysis

See `docs/HOLDING_DETECTION.md` for detailed information.

## Technical Details

### ReID Features

The lightweight ReID system extracts 512-dimensional features from:
- **LAB Color Histograms**: Perceptual color space (head, torso, legs regions)
- **HOG Features**: Histogram of Oriented Gradients
- **Texture Features**: Local variance analysis
- **Edge Density**: Canny edge detection in grid cells

### Tracking Algorithm

- **BoT-SORT**: State-of-the-art tracker with motion compensation
- **Track Buffer**: Maintains lost tracks for 10 seconds (300 frames @ 30fps)
- **Two-stage Matching**: High confidence (≥0.5) or low confidence (≥0.3) with IoU (≥0.2)

## Output Files

- `tracking_events.json`: All tracking events (entry, exit, re-identification)
- `customer_logs.json`: Detailed customer logs (if using older tracking system)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- OpenCV 4.8+
- Ultralytics YOLO 8.0+
- MediaPipe (for holding detection)

## Documentation

- **Holding Detection**: See `docs/HOLDING_DETECTION.md`
- **Manual Confirmation**: See `docs/MANUAL_CONFIRMATION.md`
- **Installation**: See `docs/INSTALL.md` (if present)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

