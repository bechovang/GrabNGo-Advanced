# GrabNGo-Advanced - Project Context

**CRITICAL RULES AND PATTERNS FOR AI AGENTS**

This document is the authoritative source of truth for all AI-assisted development. When implementing features, fixing bugs, or making architectural decisions, **ALWAYS** reference this document first.

## Project Overview

**GrabNGo-Advanced** is a smart retail tracking system that combines computer vision, embedded hardware, and IoT technologies to automatically detect customer shopping behavior without manual checkout.

### Core Value Proposition
- Autonomous retail shopping experience (grab items and walk out)
- Real-time customer tracking using YOLO pose estimation
- Weight-based pickup detection using MQTT-connected sensors
- QR code confirmation for customer identification

## Repository Structure

### Type: Multi-Part Project (2 Parts)

1. **Main Application** (`C:\Users\Admin\Desktop\GIT CLONE\GrabNGo-Advanced`)
   - Project Type: Backend (Computer Vision)
   - Language: Python 3.8+
   - Primary Framework: Ultralytics YOLO + Flask

2. **Embedded Component** (`code-weight-sensor/weight_sensor_esp32/`)
   - Project Type: Embedded (MicroPython)
   - Hardware: ESP32
   - Primary Function: Weight sensor with MQTT publishing

---

# PART 1: MAIN APPLICATION (Python Backend)

## Technology Stack

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Language** | Python | 3.8+ | Core language |
| **Computer Vision** | Ultralytics YOLO | 8.0+ | Pose estimation (yolo11n-pose.pt) |
| **Deep Learning** | PyTorch | 2.0+ | Backend for YOLO |
| **Image Processing** | OpenCV | 4.8+ | Frame processing |
| **Hand Detection** | MediaPipe | 0.10.0+ | Holding detection |
| **Web Framework** | Flask | 3.x | Dashboard and API |
| **Authentication** | Flask-HTTPAuth | 4.x | Dashboard authentication (NEW) |
| **Encryption** | cryptography | 41.x | AES-256 file encryption (NEW) |
| **MQTT Client** | paho-mqtt | 1.6.0+ | IoT communication |
| **MQTT Broker (Prod)** | Mosquitto | 2.x LTS | Self-hosted broker (NEW) |
| **Numerical Computing** | NumPy | 1.24.0+ | Array operations |
| **Image Handling** | Pillow | 9.5.0+ | Image utilities |

## Directory Structure

```
GrabNGo-Advanced/
├── src/                          # Core source code (ALL NEW CODE GOES HERE)
│   ├── __init__.py
│   ├── main_tracker.py            # Main tracking system (RetailCustomerTracker)
│   ├── holding_detector.py        # Object holding detection (MediaPipe-based)
│   ├── mqtt/                      # MQTT integration
│   │   ├── __init__.py
│   │   └── client.py              # MQTTClient wrapper
│   ├── tracker/                   # ReID module
│   │   ├── __init__.py
│   │   └── reid.py                # LightweightReID (LAB+HOG+texture+edge)
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── stats_manager.py       # Shared stats between processes
│   │   ├── encryption.py          # AES-256 file encryption (NEW)
│   │   └── archival.py             # JSON archival scripts (NEW)
│   └── web/                       # Web dashboard
│       ├── __init__.py
│       ├── server.py              # Flask app initialization
│       ├── routes.py              # API endpoints
│       └── static/                # Static assets (HTML/JS/CSS)
├── config/                        # Configuration files
│   ├── botsort_reid.yaml          # BoT-SORT tracker config
│   └── zone_config.json           # QR zone and shelf zone definitions
├── data/                          # Runtime data (gitignored)
│   ├── customer_logs.json         # Customer logs
│   ├── tracking_events.json       # All tracking events
│   ├── shared_stats.json          # Shared stats for multi-process
│   ├── shared_customers.json      # Shared customer data
│   └── shared_mqtt_events.json    # Shared MQTT events
├── models/                        # YOLO models (gitignored)
│   ├── yolo11n-pose.pt            # Pose estimation model
│   └── yolo11n-cls.pt             # Classification model for ReID
├── scripts/                       # Operational scripts (NEW)
│   ├── health_check.py            # System health monitoring
│   ├── deploy.sh                  # Deployment automation
│   └── setup_mosquitto.sh         # Mosquitto broker setup
├── systemd/                       # Production service files (NEW)
│   ├── grabngo-tracker.service    # Tracker process service
│   └── grabngo-dashboard.service  # Dashboard process service
├── docs/                          # Documentation
├── run_dashboard.py               # Dashboard-only entry point
└── requirements.txt               # Python dependencies
```

## Critical Architecture Patterns

### 1. RetailCustomerTracker (Main Class)

**Location:** `src/main_tracker.py`

**Responsibilities:**
- YOLO-based person detection and tracking
- BoT-SORT multi-object tracking with occlusion handling
- Lightweight ReID (Re-Identification) using LAB+HOG+texture+edge features
- Manual confirmation system for new customer IDs
- MQTT weight event processing
- QR zone detection for customer confirmation
- Holding detection (optional, MediaPipe-based)

**Key Methods:**
```python
class RetailCustomerTracker:
    def __init__(self, detection_model, tracker_config):
        # Initialize YOLO, BoT-SORT, MQTT, ReID, HoldingDetector

    def process_frame(self, frame):
        # Main processing pipeline:
        # 1. Detect people (YOLO pose)
        # 2. Track (BoT-SORT)
        # 3. ReID (appearance matching)
        # 4. Validate new tracks
        # 5. Update holding state

    def confirm_pending_with_customer_id(self, customer_id, pending_id=None):
        # Confirm a pending track with QR-scanned customer ID

    def _check_qr_zone(self):
        # Check if any pending track is in QR confirmation zone
        # Returns: (zone_active, pending_id, pending_count)
```

**State Management:**
- `self.customers`: Dict of confirmed customers (track_id → customer data)
- `self.pending_tracks`: Dict of pending tracks awaiting confirmation
- `self.events`: List of all tracking events
- `self.zone_active_pending`: Track ID currently in QR zone

### 2. LightweightReID (Re-Identification)

**Location:** `src/tracker/reid.py`

**Feature Extraction (512-dimensional):**
- LAB Color Histograms (64 dims × 3 regions = 192)
- HOG Features (64 dims × 3 regions = 192)
- Texture Features (32 dims × 3 regions = 96)
- Edge Density (16 dims × 3 regions = 48)

**Key Methods:**
```python
class LightweightReID:
    def extract_features(self, frame, bbox):
        # Extract 512-dim feature vector from person crop

    @staticmethod
    def similarity(f1, f2):
        # Cosine similarity between feature vectors
```

### 3. MQTTClient (Weight Events)

**Location:** `src/mqtt/client.py`

**Purpose:** Subscribe to weight change events from ESP32 sensors

**Message Format:** `"CHANGE:-480"` (weight change in grams)

**Callback Integration:**
```python
def on_weight_event(weight_change_g, timestamp):
    # Called when weight event received
    # Integrates with tracker to detect item pickup
```

### 4. Flask Web Server

**Location:** `src/web/server.py`, `src/web/routes.py`

**Endpoints:**
- `GET /` - Mobile QR scanner
- `GET /dashboard` - Dashboard interface
- `GET /dashboard/data` - Dashboard JSON data
- `GET /qr_zone_status` - QR zone status for mobile
- `POST /confirm` - Confirm customer with QR code
- `GET /pending` - Debug: list pending tracks

**Shared Stats:**
Uses `StatsManager` to share data between tracker process and dashboard:
- `data/shared_stats.json`
- `data/shared_customers.json`
- `data/shared_mqtt_events.json`

## Configuration Files

### BoT-SORT Configuration (`config/botsort_reid.yaml`)

**Critical Parameters:**
```yaml
track_buffer: 300              # Keep lost tracks for 10s @ 30fps
with_reid: True
proximity_thresh: 0.2          # Min IoU for ReID consideration
appearance_thresh: 0.1         # Min cosine similarity for ReID
```

### Zone Configuration (`config/zone_config.json`)

**Zones Defined:**
1. `qr_zone` - Area for QR code confirmation (right side of frame)
2. `shelf_zone` - Area where shelf is located (for weight sensor correlation)

**Format:** Percentage of frame (0.0 to 1.0)

## Data Flow

```
Camera Frame
    ↓
YOLO Pose Detection (people)
    ↓
BoT-SORT Tracking (track IDs)
    ↓
LightweightReID (appearance features)
    ↓
Validation & Confirmation
    ↓
MQTT Weight Events (optional)
    ↓
Dashboard Updates (via shared stats)
```

## Entry Points

1. **Full System:** `python main.py` (if exists)
2. **Dashboard Only:** `python run_dashboard.py`

---

# PART 2: EMBEDDED COMPONENT (ESP32)

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Platform** | ESP32 | Microcontroller |
| **Language** | MicroPython | Firmware |
| **Sensor** | HX711 | Load cell amplifier |
| **Communication** | umqtt.simple | MQTT client |
| **Networking** | network.WLAN | WiFi |

## Directory Structure

```
code-weight-sensor/
└── weight_sensor_esp32/
    ├── main.py          # Main entry point
    ├── boot.py          # Boot script
    ├── hx711.py         # HX711 driver library
    ├── calibrate.py     # Calibration utility
    └── test_weight.py   # Testing utility
```

## Hardware Configuration

**Pin Assignments:**
- `DT_PIN = 25` - HX711 DOUT (Data Out)
- `SCK_PIN = 26` - HX711 SCK (Serial Clock)

**MQTT Configuration:**
- Broker: `test.mosquitto.org`
- Topic: `my-shop/shelf-1/events`
- Client ID: `esp32-shelf-1`

**Calibration Values:**
- `TARE_VALUE = 471778`
- `VALUE_WITH_WEIGHT = 256326`
- `KNOWN_WEIGHT_G = 480`
- `RATIO = (VALUE_WITH_WEIGHT - TARE_VALUE) / KNOWN_WEIGHT_G`

## Key Features

1. **WiFi Connection** - Auto-connect to configured network
2. **Weight Reading** - Stable readings using median filter
3. **Change Detection** - Publishes only when weight changes > 50g
4. **MQTT Publishing** - Format: `"CHANGE:-480"` (negative = removed)
5. **Error Recovery** - Auto-reconnect on WiFi/MQTT failure

## Data Flow

```
HX711 Sensor → Raw Reading → Convert to Weight → Detect Change → MQTT Publish
```

---

# INTEGRATION BETWEEN PARTS

## Communication Protocol

**MQTT Topic:** `my-shop/shelf-1/events`

**Message Format:** `"CHANGE:<weight_change_g>"`

**Example:**
- `"CHANGE:-480"` - 480g removed from shelf (item picked up)
- `"CHANGE:480"` - 480g added to shelf (item returned)

## Correlation Logic

1. ESP32 detects weight change → publishes MQTT event
2. Python tracker receives event via `MQTTClient`
3. Tracker correlates with active tracks in `shelf_zone`
4. If customer in zone + weight decreases → item picked up
5. Event logged to `tracking_events.json`

---

# CRITICAL IMPLEMENTATION RULES

## For All New Code

1. **Module Structure:** ALL new code goes in `src/` directory with proper `__init__.py`
2. **Type Hints:** Use type hints for function signatures
3. **Error Handling:** Use try-except with silent fail for non-critical paths
4. **Logging:** Use print statements with prefixes: `[INFO]`, `[WARN]`, `[ERROR]`
5. **Configuration:** Store config in `config/` directory (JSON/YAML)

## When Adding Features

1. **Tracker Features:** Extend `RetailCustomerTracker` class
2. **New Modules:** Create in appropriate `src/` subdirectory
3. **Web Endpoints:** Add to `src/web/routes.py`
4. **Config Changes:** Update `config/` files
5. **Shared Data:** Use `StatsManager` for inter-process communication

## When Modifying Tracking Logic

1. **Validation Thresholds:** Modify in `main_tracker.py`
2. **ReID Parameters:** Modify `config/botsort_reid.yaml`
3. **Zone Definitions:** Modify `config/zone_config.json`
4. **Holding Detection:** Modify `holding_detector.py`

## When Adding New Hardware Sensors

1. Create new ESP32 firmware in `code-weight-sensor/`
2. Add MQTT topic subscription in `src/mqtt/client.py`
3. Add event handler in `main_tracker.py`
4. Update zone configuration if needed

---

# TESTING GUIDELINES

## Unit Tests
- Mock YOLO, MQTT, MediaPipe dependencies
- Test individual components in isolation

## Integration Tests
- Test MQTT flow with real broker
- Test QR confirmation flow
- Test weight event correlation

## Hardware Tests
- Use `test_weight.py` for ESP32 testing
- Use `calibrate.py` for sensor calibration

---

# DEPENDENCIES

## Python (Main Application)

**Install:**
```bash
pip install -r requirements.txt
```

**Contents:**
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=9.5.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
mediapipe>=0.10.0
paho-mqtt>=1.6.0
flask-httpauth>=4.0.0      # NEW: Dashboard authentication
cryptography>=41.0.0        # NEW: AES-256 file encryption
```

## MicroPython (ESP32)

**Required Libraries:**
- `umqtt.simple` - MQTT client (built-in to most MicroPython builds)

---

# ENVIRONMENT SETUP

## Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## ESP32 Flashing
1. Install MicroPython on ESP32
2. Copy files: `boot.py`, `hx711.py`, `main.py`
3. Configure WiFi credentials in `main.py`

---

# GIT IGNORE PATTERNS

```
venv/
__pycache__/
*.pyc
data/*.json
models/*.pt
.env
.DS_Store
```

---

# PRODUCTION SECURITY & DEPLOYMENT (NEW from Architecture)

## Security Architecture

### Dashboard Authentication
**Technology:** Flask-HTTPAuth 4.x
**Pattern:** HTTP Basic Auth with role-based access

**Roles:**
- **admin** - Full access to all endpoints including `/api/export/transactions`
- **manager** - Dashboard viewing, basic operations
- **viewer** - Read-only dashboard access

**Implementation Location:** `src/web/routes.py`

**Example Pattern:**
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Check credentials (hardcoded for MVP, database in Phase 2)
    # Role mapping: admin/manager/viewer

@app.route('/dashboard')
@auth.login_required
def dashboard():
    role = get_user_role(auth.username())
    if role not in ['admin', 'manager', 'viewer']:
        return jsonify({'error': 'Unauthorized'}), 403
```

### Data Encryption at Rest
**Technology:** cryptography 41.x (AES-256 via Fernet)
**Pattern:** File-level encryption for sensitive data only

**Encrypted Files:**
- `data/customer_logs.json` - Customer transaction records
- `data/tracking_events.json` - Tracking metadata

**NOT Encrypted:**
- `data/shared_stats.json` - IPC performance (unencrypted for speed)

**Implementation Location:** `src/utils/encryption.py` (NEW)

**Key Management:**
- Environment variable: `ENCRYPTION_KEY` or
- Key file: `data/.encryption_key` (gitignored)

**Example Pattern:**
```python
from cryptography.fernet import Fernet

def encrypt_file(filepath, key):
    f = Fernet(key)
    with open(filepath, 'rb') as file:
        data = file.read()
    encrypted = f.encrypt(data)
    with open(filepath, 'wb') as file:
        file.write(encrypted)

def generate_key():
    return Fernet.generate_key()
```

### MQTT Broker Configuration
**Development:** `test.mosquitto.org` (public broker)
**Production:** Mosquitto 2.x LTS (self-hosted)

**Production Setup:**
- TLS certificate from Let's Encrypt
- Username/password authentication
- Configuration files: `/etc/mosquitto/mosquitto.conf`

**Configuration Update Required:**
- ESP32 `code-weight-sensor/weight_sensor_esp32/main.py`
- Python `src/mqtt/client.py`

**Connection Pattern:**
```python
# Python (src/mqtt/client.py)
broker = "localhost"  # Production self-hosted
port = 8883  # TLS port
username = "grabngo_client"
password = os.getenv("MQTT_PASSWORD")

# ESP32 (main.py)
mqtt_broker = "192.168.1.100"  # Local network IP
mqtt_port = 8883  # TLS
mqtt_user = "esp32_client"
mqtt_password = "your_password"
```

## Deployment Architecture

### systemd Services
Production deployment uses systemd for auto-restart and logging:

**Tracker Service:** `/etc/systemd/system/grabngo-tracker.service`
```ini
[Unit]
Description=GrabNGo Tracking System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/home/user/GrabNGo-Advanced
ExecStart=/home/user/GrabNGo-Advanced/venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Dashboard Service:** `/etc/systemd/system/grabngo-dashboard.service`
```ini
[Unit]
Description=GrabNGo Dashboard
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/home/user/GrabNGo-Advanced
ExecStart=/home/user/GrabNGo-Advanced/venv/bin/python run_dashboard.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Commands:**
```bash
sudo systemctl enable grabngo-tracker
sudo systemctl start grabngo-tracker
sudo systemctl status grabngo-tracker
journalctl -u grabngo-tracker -f  # View logs
```

### Health Monitoring
**Script:** `scripts/health_check.py` (NEW)

**Features:**
- Check camera connection every 30s
- Check MQTT broker connection every 30s
- Send email alerts when offline >60s
- Log to journalctl for debugging

**Cron Job:**
```bash
# /etc/cron.d/grabngo-health
*/1 * * * * /path/to/venv/bin/python /path/to/scripts/health_check.py
```

**Email Alert Pattern:**
```python
import smtplib
def send_alert_email(service_name):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('alerts@yourstore.com', 'password')
    server.sendmail('alerts@yourstore.com', 'admin@yourstore.com',
                   f'Subject: ALERT: {service_name} is offline')
```

## Data Retention & Archival

### JSON with Daily Archival
**Active Data:** 7 days in `data/customer_logs.json`
**Archived Data:** `data/archive/customers_YYYY-MM-DD.json.gz`

**Implementation:** `src/utils/archival.py` (NEW)

**Retention Policy:**
- Active data: Last 7 days
- Archived data: Compressed by date
- GDPR deletion: Customer data purge on request

**Archival Pattern:**
```python
import gzip
import json
from datetime import datetime, timedelta

def archive_old_customers():
    cutoff_date = datetime.now() - timedelta(days=7)
    # Move customers older than cutoff to archive
    # Compress with gzip
    # Save to data/archive/customers_YYYY-MM-DD.json.gz
```

**GDPR Deletion Workflow:**
```python
def delete_customer_data(customer_id):
    # Remove from customer_logs.json
    # Remove from archive files
    # Log deletion for audit trail
    print(f"[INFO] Customer {customer_id} data deleted per GDPR request")
```

**Cron Job:**
```bash
# /etc/cron.d/grabngo-archive
0 2 * * * /path/to/venv/bin/python /path/to/src/utils/archival.py
```

## Customer ID Format
**Pattern:** `CUST_XXXX` (e.g., CUST_0001, CUST_0245)
- Pseudonymized, no real names
- Generated upon QR zone entry or manual confirmation
- Sequential numbering starting from 0001

**Implementation Location:** `src/main_tracker.py`

---

*Last Updated: 2026-01-14*
*Workflow: document-project (exhaustive scan) + Architecture Integration*

---

# LANGUAGE-SPECIFIC RULES (Python)

## Silent Fail Pattern (CRITICAL for 30 FPS)

**Rule:** Non-critical operations MUST NOT crash the main tracking loop

```python
# CORRECT: Silent fail for non-critical paths
try:
    stats_manager.save_stats(data)
except Exception as e:
    pass  # Don't break main process

# CORRECT: Log but don't crash
try:
    mqtt_client.connect()
except Exception as e:
    print(f"[WARN] MQTT connection failed: {e}")

# WRONG: Crashing on non-critical error
try:
    stats_manager.save_stats(data)
except Exception as e:
    raise  # This breaks the 30 FPS loop!
```

**Apply to:**
- File I/O operations (StatsManager)
- MQTT connection (non-critical for tracking)
- Optional feature failures (HoldingDetector)

## Print with Prefixes (No Logging Framework)

**Rule:** Use print statements with `[PREFIX]` for all logging

```python
# Prefix hierarchy:
print("[ERROR] Camera not found")      # Critical failures
print("[WARN] MQTT disconnected")      # Non-critical issues
print("[INFO] System started")         # Normal operations
print("[DEBUG] Track updated")         # Debugging (with DEBUG flag)
```

**DO NOT USE:**
- `logging` module (not in MVP)
- `logger.info()`, `logger.error()` etc.
- Unadorned `print()` without prefix

## PEP 8 Naming Conventions (Strict)

**Classes:** `CapWords` (e.g., `RetailCustomerTracker`, `MQTTClient`)
**Functions/Variables:** `snake_case` (e.g., `process_frame()`, `track_id`)
**Constants:** `UPPER_CASE` (e.g., `MAX_TRACKS`, `FRAME_BUFFER`)
**Files:** `snake_case.py` (e.g., `main_tracker.py`, `mqtt_client.py`)

## Type Hints Required

**Rule:** All function signatures MUST have type hints

```python
# CORRECT:
def extract_features(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
    pass

# WRONG:
def extract_features(self, frame, bbox):  # Missing type hints
    pass
```

## MQTT Payload Format (Simple Strings)

**Rule:** MQTT payloads are comma-separated strings, NOT JSON

```python
# CORRECT (backward compatible with ESP32):
payload = f"{weight_change_g},{timestamp_unix}"
# Example: "+250,1736848000"

# WRONG (breaks ESP32 parsing):
payload = json.dumps({"weight": 250, "timestamp": 1736848000})
```

---

# FRAMEWORK-SPECIFIC RULES

## Flask Authentication (Flask-HTTPAuth 4.x)

**Pattern:** HTTP Basic Auth with role-based access

```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

# MVP: Hardcoded users dictionary
USERS = {
    'admin': 'admin_password',     # Full access
    'manager': 'manager_password', # Dashboard + export
    'viewer': 'viewer_password'    # Read-only
}

@auth.verify_password
def verify_password(username, password):
    if username in USERS and USERS[username] == password:
        return username

@app.route('/api/export/transactions')
@auth.login_required
def export_transactions():
    if auth.username() != 'admin':
        return jsonify({'error': 'Admin only'}), 403
```

## Flask API Pattern (Minimal)

**Rule:** Don't over-engineer - add endpoints as needed

**CORRECT:**
```python
@app.route('/api/stats')
@auth.login_required
def get_stats():
    return jsonify(stats_manager.load_stats())
```

**WRONG (for MVP):**
- OpenAPI/Swagger documentation
- Hypermedia HATEOAS
- Versioned URLs (/api/v1/)

## YOLO Model Loading (Performance Critical)

**Rule:** Load YOLO model ONCE at initialization

```python
# CORRECT: Load once in __init__
class RetailCustomerTracker:
    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')  # Load once

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)  # Use cached model
        # ...

# WRONG: Reload every frame (30x slower!)
def process_frame(self, frame):
    model = YOLO('yolo11n-pose.pt')  # This kills FPS
    results = model(frame)
```

## MQTT Callback Pattern

**Rule:** Callbacks must use silent fail pattern

```python
# CORRECT:
def _on_message(self, client, userdata, msg):
    try:
        payload = msg.payload.decode()
        weight_change, timestamp = payload.split(',')
        self.on_weight_event(int(weight_change), int(timestamp))
    except Exception as e:
        print(f"[WARN] MQTT parse failed: {e}")  # Don't crash
```

---

# USAGE GUIDELINES

## For AI Agents

**Before implementing any code:**
1. Read this file in its entirety
2. Follow ALL rules exactly as documented
3. When in doubt, prefer the more restrictive option
4. Never skip authentication or encryption for production code

**Critical Rules to Remember:**
- **ALL new code goes in `src/` directory**
- **Silent fail for non-critical operations** (don't break 30 FPS loop)
- **Print with prefixes** `[ERROR]`, `[WARN]`, `[INFO]`, `[DEBUG]`
- **PEP 8 naming** strictly enforced
- **Type hints required** on all function signatures
- **MQTT payloads are simple strings** (not JSON)

## For Humans

**Keeping this file optimized:**
- Keep content lean and focused on agent needs
- Update when technology stack changes
- Review quarterly for outdated rules
- Remove rules that become obvious over time

**When to update:**
- New dependencies added to requirements.txt
- New architectural patterns emerge
- Security requirements change
- Performance patterns are discovered

---

## SUMMARY

**Total Critical Rules:** 25+ implementation rules
**Sections:** 7 comprehensive sections
**Optimized for:** LLM context efficiency

**File Location:** `project-context.md` (root of repository)

---

*Project Context completed: 2026-01-14*
*Workflows: document-project + Architecture Integration*
*Status: Complete and optimized for AI agents*
