# Architecture: Smart Retail Tracking System (Main Application)

**Part ID:** `main`
**Project Type:** Backend (Computer Vision)
**Technology Stack:** Python, YOLO, PyTorch, OpenCV, MediaPipe, MQTT, Flask

---

## Executive Summary

The Smart Retail Tracking System is a production-ready computer vision application that performs real-time customer tracking using YOLO pose estimation, BoT-SORT multi-object tracking, and lightweight re-identification (ReID). The system integrates with MQTT-connected weight sensors and provides a web-based dashboard for monitoring and QR code confirmation.

**Key Capabilities:**
- Real-time person detection and tracking (30 FPS)
- Appearance-based re-identification across occlusions
- Manual customer confirmation via QR codes
- Weight-based pickup detection via MQTT
- Object holding detection (MediaPipe-based)
- Multi-process shared statistics

---

## Technology Stack

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Language** | Python | 3.8+ | Industry standard for CV/ML |
| **CV Framework** | Ultralytics YOLO | 8.0+ | State-of-the-art pose estimation |
| **Deep Learning** | PyTorch | 2.0+ | YOLO backend, CUDA support |
| **Image Processing** | OpenCV | 4.8+ | Frame capture and processing |
| **Hand Detection** | MediaPipe | 0.10.0+ | Hand/pose for holding detection |
| **Web Framework** | Flask | Latest | Lightweight HTTP server |
| **MQTT Client** | paho-mqtt | 1.6.0+ | IoT sensor integration |
| **Numerical** | NumPy | 1.24.0+ | Array operations |
| **Image Utils** | Pillow | 9.5.0+ | Image handling |

---

## Architecture Pattern

**Pattern:** Event-Driven Pipeline with State Machine

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Camera     │───▶│   YOLO      │───▶│  BoT-SORT   │───▶│   ReID      │
│  Capture    │    │  Detection  │    │  Tracking   │    │  Matching    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                    ┌─────────────┐    ┌─────────────┐           │
                    │   MQTT      │◀───│  State      │◀──────────┘
                    │  Events     │    │  Machine    │
                    └─────────────┘    └─────────────┘
                            │                  │
                    ┌───────┴─────┐    ┌──────┴──────┐
                    │   Stats     │    │   Web       │
                    │  Manager    │    │  Dashboard  │
                    └─────────────┘    └─────────────┘
```

### Component Responsibilities

1. **YOLO Detection** - Person pose detection (keypoints + bboxes)
2. **BoT-SORT** - Multi-object tracking with Kalman filter
3. **ReID Module** - Appearance feature extraction and matching
4. **State Machine** - Track lifecycle: PENDING → VALIDATED → CONFIRMED
5. **MQTT Client** - Subscribe to weight change events
6. **Stats Manager** - Inter-process data sharing
7. **Web Server** - Flask API and dashboard

---

## Component Overview

### 1. RetailCustomerTracker (Core)

**File:** `src/main_tracker.py`

**Responsibilities:**
- Coordinate all tracking pipeline stages
- Manage track state (pending, validated, confirmed)
- Handle MQTT weight events
- QR zone detection and confirmation
- Holding detection integration

**Key Attributes:**
```python
class RetailCustomerTracker:
    customers: Dict[int, Dict]              # Confirmed customers
    pending_tracks: Dict[int, Dict]         # Tracks awaiting confirmation
    events: List[Dict]                      # All tracking events
    zone_active_pending: Optional[int]      # Track in QR zone
    mqtt_connected: bool                    # MQTT connection status
```

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `process_frame(frame)` | Main processing pipeline |
| `confirm_pending_with_customer_id(...)` | QR confirmation |
| `_check_qr_zone()` | Check QR zone occupancy |
| `_handle_weight_event(...)` | MQTT weight event handler |

---

### 2. LightweightReID (Re-Identification)

**File:** `src/tracker/reid.py`

**Feature Extraction (512-dimensional):**
- LAB Color Histograms (192 dims) - Head, Torso, Legs regions
- HOG Features (192 dims) - Gradient orientation
- Texture Features (96 dims) - Local variance
- Edge Density (48 dims) - Canny edge grid

**Algorithm:**
```python
def extract_features(frame, bbox):
    1. Crop and resize person region (128x256)
    2. Divide into 3 regions: head, torso, legs
    3. Extract LAB, HOG, texture, edge for each region
    4. Concatenate and normalize to 512-dim vector
    5. Return feature vector
```

---

### 3. MQTTClient (Weight Events)

**File:** `src/mqtt/client.py`

**Protocol:**
- **Broker:** `test.mosquitto.org` (configurable)
- **Topic:** `my-shop/shelf-1/events`
- **Message:** `"CHANGE:-480"` (grams)

**Callbacks:**
```python
def on_weight_event(weight_change_g: int, timestamp: datetime):
    # Called when weight change detected
    # Negative value = item removed
    # Positive value = item returned
```

---

### 4. HoldingDetector (Optional)

**File:** `src/holding_detector.py`

**Methods:**
1. **MediaPipe Hands** - Finger state detection (fist vs open)
2. **Dominant Color** - K-means clustering to detect non-skin colors
3. **Color Variance** - High variance indicates object presence

**Combined Score:**
```
score = 0.3 × finger_score + 0.4 × dominant_score + 0.3 × variance_score
```

---

### 5. StatsManager (Inter-Process Communication)

**File:** `src/utils/stats_manager.py`

**Shared Files:**
- `data/shared_stats.json` - Statistics
- `data/shared_customers.json` - Customer data
- `data/shared_mqtt_events.json` - Recent MQTT events

**Purpose:** Allow dashboard to access data without direct tracker access

---

### 6. Flask Web Server

**Files:** `src/web/server.py`, `src/web/routes.py`

**Endpoints:**
| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Mobile QR scanner |
| GET | `/dashboard` | Dashboard interface |
| GET | `/dashboard/data` | JSON API for dashboard |
| GET | `/qr_zone_status` | QR zone status |
| POST | `/confirm` | Confirm customer via QR |
| GET | `/pending` | Debug: list pending tracks |

---

## Data Architecture

### Track State Machine

```
┌───────────┐     Validation      ┌──────────────┐
│  PENDING  │─────────────────────▶│  VALIDATED   │
│           │  (samples ≥ 5,       │              │
│           │   confidence ≥ 0.5)  │              │
└───────────┘                     └──────────────┘
     │                                  │
     │ QR Confirmation                   │
     │ (customer_id)                    │
     ▼                                  │
┌───────────┐                          │
│ CONFIRMED │◀─────────────────────────┘
│           │     Manual Confirm
└───────────┘
```

### Data Models

#### Customer Record
```python
{
    "customer_id": "CUST_0001",
    "state": TrackState.CONFIRMED,
    "entry_time": datetime,
    "last_detection_time": datetime,
    "shopping_cart": [],
    "pickup_count": 0,
    "features": np.array(512),  # ReID features
    "last_box": [x1, y1, x2, y2]
}
```

#### Pending Track
```python
{
    "pending_id": "PENDING_0001",
    "track_id": 1,
    "first_seen": datetime,
    "last_seen": datetime,
    "features": np.array(512),
    "validation_score": 0.85,
    "last_box": [x1, y1, x2, y2]
}
```

#### Tracking Event
```python
{
    "type": "entry" | "exit" | "reid" | "item_picked_up",
    "track_id": int,
    "customer_id": str | None,
    "timestamp": datetime,
    "details": dict
}
```

---

## Source Tree

```
src/
├── __init__.py
├── main_tracker.py              # RetailCustomerTracker (main class)
├── holding_detector.py          # HoldingDetector (optional)
├── mqtt/
│   ├── __init__.py
│   └── client.py                # MQTTClient
├── tracker/
│   ├── __init__.py
│   └── reid.py                  # LightweightReID
├── utils/
│   ├── __init__.py
│   └── stats_manager.py         # StatsManager
└── web/
    ├── __init__.py
    ├── server.py                # Flask app
    ├── routes.py                # API endpoints
    └── static/
        ├── dashboard.html       # Dashboard UI
        ├── mobile_qr_scanner.html
        └── ...
```

---

## Development Workflow

### Prerequisites and Dependencies

**Install:**
```bash
pip install -r requirements.txt
```

**Models Required:**
- `models/yolo11n-pose.pt` - Pose estimation
- `models/yolo11n-cls.pt` - Classification (for ReID)

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Local Development Commands

**Run dashboard only:**
```bash
python run_dashboard.py
# Access: http://localhost:8081/dashboard
```

**Run full system (if main.py exists):**
```bash
python main.py
# Access: http://localhost:8080/dashboard
```

### Testing Approach

**Unit Tests:**
- Mock YOLO, MQTT, MediaPipe
- Test components in isolation

**Integration Tests:**
- MQTT flow with real broker
- QR confirmation flow
- Weight event correlation

---

## Deployment Architecture

### Infrastructure Requirements

**Hardware:**
- Camera (USB or RTSP stream)
- GPU (NVIDIA, optional) for YOLO acceleration
- Network connection for MQTT

**Software:**
- Python 3.8+
- CUDA 11+ (if using GPU)
- MQTT broker (test.mosquitto.org or local)

### Environment Configuration

**Config Files:**
- `config/botsort_reid.yaml` - Tracker parameters
- `config/zone_config.json` - Zone definitions

**MQTT Settings:**
- Broker: Configurable in `main_tracker.py`
- Topic: `my-shop/shelf-1/events`

---

## Testing Strategy

### Test Coverage

**Components to Test:**
1. ReID feature extraction
2. MQTT client connection/events
3. QR zone detection
4. State transitions (PENDING → VALIDATED → CONFIRMED)
5. Weight event correlation

### Test Commands

**Test MQTT connection:**
```bash
# Subscribe to test topic
mosquitto_sub -h test.mosquitto.org -t "my-shop/shelf-1/events"
```

**Test camera:**
```bash
# If test script exists
python test_camera.py
```

---

## Known Limitations

1. **Occlusion Duration:** Track buffer = 300 frames (10s @ 30fps)
2. **ReID Accuracy:** Lower confidence for similar clothing
3. **MQTT Reliability:** Public broker may be unreliable
4. **Lighting Sensitivity:** YOLO performance affected by poor lighting
5. **Holding Detection:** Optional feature, may be disabled

---

*Architecture document generated: 2026-01-14*
*Scan Level: Exhaustive*
*Part: main (Smart Retail Tracking System)*
