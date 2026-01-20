# Component Inventory - GrabNGo-Advanced

## Overview

This document catalogs all components in the GrabNGo-Advanced project, categorized by type, reusability, and purpose.

---

## Part 1: Main Application (Python Backend)

### Core Tracking Components

| Component | File | Type | Purpose | Reusability |
|-----------|------|------|---------|-------------|
| **RetailCustomerTracker** | `src/main_tracker.py` | Class | Main tracking system | Project-specific |
| **LightweightReID** | `src/tracker/reid.py` | Class | Re-identification (512-dim features) | ✅ Reusable |
| **TrackState** | `src/main_tracker.py` | Enum | PENDING/CONFIRMED states | Project-specific |
| **HoldingDetector** | `src/holding_detector.py` | Class | Object holding detection | ✅ Reusable |

### Communication Components

| Component | File | Type | Purpose | Reusability |
|-----------|------|------|---------|-------------|
| **MQTTClient** | `src/mqtt/client.py` | Class | MQTT wrapper for weight events | ✅ Reusable |
| **StatsManager** | `src/utils/stats_manager.py` | Class | Inter-process data sharing | ✅ Reusable |

### Web Components

| Component | File | Type | Purpose | Reusability |
|-----------|------|------|---------|-------------|
| **Flask App** | `src/web/server.py` | Module | Web server initialization | ✅ Reusable |
| **Routes** | `src/web/routes.py` | Module | API endpoints | Project-specific |

### UI Components (Static Assets)

| Component | File | Type | Purpose | Reusability |
|-----------|------|------|---------|-------------|
| **Dashboard** | `src/web/static/dashboard.html` | HTML | Main dashboard UI | Project-specific |
| **Mobile QR Scanner** | `src/web/static/mobile_qr_scanner.html` | HTML | QR code confirmation | Project-specific |

---

## Part 2: Embedded Component (ESP32)

### ESP32 Components

| Component | File | Type | Purpose | Reusability |
|-----------|------|------|---------|-------------|
| **HX711 Driver** | `hx711.py` | Class | Load cell amplifier driver | ✅ Reusable |
| **Main Loop** | `main.py` | Script | Weight monitoring + MQTT | Project-specific |
| **Boot Script** | `boot.py` | Script | Device initialization | ✅ Reusable |
| **Calibration** | `calibrate.py` | Script | Sensor calibration utility | ✅ Reusable |

---

## Reusable Components Detail

### 1. LightweightReID (`src/tracker/reid.py`)

**Purpose:** Extract 512-dimensional appearance features from person crops

**Key Features:**
- LAB color histograms (192 dims)
- HOG features (192 dims)
- Texture features (96 dims)
- Edge density (48 dims)

**Methods:**
```python
def extract_features(frame, bbox) -> np.ndarray  # Returns 512-dim vector
@staticmethod
def similarity(f1, f2) -> float  # Cosine similarity
```

**Reusability:** Can be extracted for other tracking projects

---

### 2. MQTTClient (`src/mqtt/client.py`)

**Purpose:** Wrapper for paho-mqtt with connection handling

**Key Features:**
- Async connection with timeout
- Network connectivity test
- Auto-reconnect on disconnect
- Event callback registration

**Usage:**
```python
def on_event(weight_change, timestamp):
    print(f"Weight changed: {weight_change}")

client = MQTTClient(
    broker="test.mosquitto.org",
    topic="my-shop/shelf-1/events",
    on_weight_event=on_event
)
client.connect()
```

**Reusability:** Can be used for any MQTT subscription use case

---

### 3. StatsManager (`src/utils/stats_manager.py`)

**Purpose:** Share data between Python processes via JSON files

**Key Features:**
- Thread-safe file writing
- Silent error handling (no crash on write failure)
- Auto-timestamp for all data

**Methods:**
```python
save_stats(stats_data)
load_stats() -> dict
save_customers_data(customers_data)
load_customers_data() -> dict
save_mqtt_events(mqtt_events)
load_mqtt_events() -> list
```

**Reusability:** Useful for any multi-process Python application

---

### 4. HoldingDetector (`src/holding_detector.py`)

**Purpose:** Detect if person is holding an object using MediaPipe

**Key Features:**
- MediaPipe Hands (finger state detection)
- Dominant color detection (K-means)
- Color variance analysis
- Temporal smoothing (reduce flickering)

**Methods:**
```python
detect_holding(customer_id, person_bbox, keypoints, frame) -> dict
reset_customer(customer_id)
get_holding_status(customer_id) -> bool
```

**Reusability:** Can be used for any hand-held object detection

---

### 5. HX711 Driver (`code-weight-sensor/hx711.py`)

**Purpose:** MicroPython driver for HX711 load cell amplifier

**Key Features:**
- 24-bit ADC reading
- Configurable gain (32, 64, 128)
- Averaging for stable readings
- Tare function
- Power down/up modes

**Methods:**
```python
read() -> int  # Raw 24-bit value
read_average(times=16) -> float
tare(times=16)  # Set zero offset
get_weight(times=16) -> float  # Convert to weight
```

**Reusability:** Can be used with any HX711-based scale project

---

## Project-Specific Components

### RetailCustomerTracker

**Why Project-Specific:**
- Tightly coupled to YOLO pose detection
- Uses specific zone configuration
- Implements QR confirmation workflow
- Integrates with MQTT weight events

**Key Customizations:**
- `zone_config.json` integration
- Manual confirmation state machine
- Weight event correlation
- Customer lifecycle management

---

## Component Dependencies

```
RetailCustomerTracker
├── LightweightReID (appearance matching)
├── MQTTClient (weight events)
├── HoldingDetector (optional)
└── StatsManager (data sharing)

Flask Web Server
├── Routes (API endpoints)
└── StatsManager (read data)

ESP32 Main Loop
├── HX711 Driver (weight readings)
└── MQTT (publish events)
```

---

## Design System Patterns

### Error Handling Pattern

**Silent Fail for Non-Critical:**
```python
try:
    risky_operation()
except Exception as e:
    pass  # Don't break main process
```

**Used In:** StatsManager, MQTTClient callbacks

---

### Callback Pattern

**MQTT Weight Events:**
```python
def on_weight_event(weight_change_g, timestamp):
    # Handle event
    pass

client = MQTTClient(on_weight_event=on_weight_event)
```

**Used In:** MQTTClient, HoldingDetector

---

### State Machine Pattern

**Track Lifecycle:**
```
PENDING → VALIDATED → CONFIRMED
```

**Used In:** RetailCustomerTracker

---

## Component Modularity

### High Modularity (Easy to Extract)

1. **LightweightReID** - No external dependencies except OpenCV/NumPy
2. **MQTTClient** - Standalone wrapper
3. **StatsManager** - Pure Python, no dependencies
4. **HX711 Driver** - Pure MicroPython

### Medium Modularity (Some Coupling)

1. **HoldingDetector** - Uses MediaPipe (optional dependency)
2. **Flask App** - Routes depend on tracker structure
3. **Boot Script** - Hardware-specific configuration

### Low Modularity (Tightly Coupled)

1. **RetailCustomerTracker** - Core business logic, highly specialized
2. **Main Loop (ESP32)** - Hardware + WiFi + MQTT tightly integrated

---

## Extension Points

### Adding New ReID Methods

**File:** `src/tracker/reid.py`

**Extend:** Add new feature extraction method
```python
def _new_feature_method(self, img):
    # Extract custom features
    return feature_vector
```

### Adding New MQTT Topics

**File:** `src/mqtt/client.py`

**Extend:** Add multiple topic subscriptions
```python
class MQTTClient:
    def __init__(self, topics=[]):
        self.topics = topics
```

### Adding New Zones

**File:** `config/zone_config.json`

**Extend:** Add new zone definition
```json
{
  "new_zone": {
    "x1_percent": 0.0,
    "y1_percent": 0.0,
    "x2_percent": 0.2,
    "y2_percent": 1.0
  }
}
```

---

*Component inventory generated: 2026-01-14*
*Scan Level: Exhaustive*
