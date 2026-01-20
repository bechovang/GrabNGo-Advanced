# Source Tree Analysis - GrabNGo-Advanced

## Complete Directory Structure

```
GrabNGo-Advanced/
├── project-context.md                    # ⭐ CRITICAL - AI agent "bible"
│
├── src/                                  # Core source code - ALL NEW CODE GOES HERE
│   ├── __init__.py
│   ├── main_tracker.py                   # RetailCustomerTracker class
│   ├── holding_detector.py               # HoldingDetector (MediaPipe-based)
│   │
│   ├── mqtt/                             # MQTT integration
│   │   ├── __init__.py
│   │   └── client.py                     # MQTTClient wrapper
│   │
│   ├── tracker/                          # ReID module
│   │   ├── __init__.py
│   │   └── reid.py                       # LightweightReID class
│   │
│   ├── utils/                            # Utilities
│   │   ├── __init__.py
│   │   └── stats_manager.py              # StatsManager (inter-process data)
│   │
│   └── web/                              # Web dashboard
│       ├── __init__.py
│       ├── server.py                     # Flask app initialization
│       ├── routes.py                     # API endpoints
│       └── static/                       # Static assets
│           ├── dashboard.html            # Dashboard UI
│           ├── mobile_qr_scanner.html    # Mobile QR app
│           └── ...
│
├── config/                               # Configuration files
│   ├── botsort_reid.yaml                 # BoT-SORT tracker + ReID config
│   └── zone_config.json                  # QR zone & shelf zone definitions
│
├── data/                                 # Runtime data (gitignored)
│   ├── customer_logs.json                # Customer logs
│   ├── tracking_events.json              # All tracking events
│   ├── shared_stats.json                 # Shared stats (multi-process)
│   ├── shared_customers.json             # Shared customer data
│   ├── shared_mqtt_events.json           # Shared MQTT events
│   ├── logs/                             # Log files
│   │   └── tracking_events.json
│   └── qr_codes/                         # Generated QR codes
│
├── models/                               # YOLO models (gitignored)
│   ├── yolo11n-pose.pt                   # Pose estimation model
│   └── yolo11n-cls.pt                    # Classification model (ReID)
│
├── scripts/                              # Utility scripts
│
├── docs/                                 # Documentation
│   ├── index.md                          # ⭐ Master documentation index
│   ├── architecture-main.md              # Main application architecture
│   ├── architecture-esp32.md             # ESP32 architecture
│   ├── source-tree-analysis.md           # This file
│   ├── component-inventory.md            # Component inventory
│   ├── development-guide.md              # Development guide
│   │
│   ├── setup/
│   │   ├── INSTALL.md                    # Installation instructions
│   │   └── SYSTEM_SETUP.md               # Complete system setup
│   │
│   ├── features/
│   │   ├── DASHBOARD_README.md           # Dashboard documentation
│   │   ├── MANUAL_CONFIRMATION.md        # Manual confirmation system
│   │   ├── QR_CONFIRMATION_GUIDE.md      # QR code confirmation
│   │   ├── README_QR_SYSTEM.md           # QR system overview
│   │   ├── WEIGHT_PICKUP_DETECTION.md    # Weight-based detection
│   │   └── ZONE_CONTROLS_README.md       # Zone control system
│   │
│   ├── troubleshooting/
│   │   └── TROUBLESHOOTING_WEB.md        # Web dashboard troubleshooting
│   │
│   ├── HOLDING_DETECTION.md              # Object holding detection
│   ├── README_UNIFIED.md                 # Unified system docs
│   ├── QUICK_TEST_GUIDE.md               # Quick testing guide
│   ├── BAO_CAO_TIEN_DO.md                # Progress report (Vietnamese)
│   └── readme_yolo.md                    # YOLO-specific docs
│
├── code-weight-sensor/                   # ⭐ PART 2: Embedded component
│   └── weight_sensor_esp32/              # ESP32/MicroPython code
│       ├── main.py                       # Main entry point
│       ├── boot.py                       # Boot script
│       ├── hx711.py                      # HX711 driver library
│       ├── calibrate.py                  # Calibration utility
│       ├── test_weight.py                # Testing utility
│       └── HUONG_DAN_NAP_CODE_ESP32.md   # Code loading guide (Vietnamese)
│
├── run_dashboard.py                      # Dashboard-only entry point
│
├── requirements.txt                      # Python dependencies
├── README.md                             # Project overview
│
├── venv/                                 # Virtual environment (gitignored)
├── _bmad/                                # BMAD workflow files
├── _bmad-output/                         # BMAD generated output
└── .git/                                 # Git repository
```

---

## Critical Folders Summary

### Entry Points

| File | Purpose | Access |
|------|---------|--------|
| `run_dashboard.py` | Dashboard only (MQTT monitoring) | http://localhost:8081/dashboard |
| `main.py` | Full system with camera | http://localhost:8080/dashboard |

### Core Source (`src/`)

**All new code MUST go in this directory with proper module structure.**

| Subdirectory | Purpose | Key Files |
|--------------|---------|-----------|
| `src/mqtt/` | MQTT integration | `client.py` |
| `src/tracker/` | ReID module | `reid.py` |
| `src/utils/` | Shared utilities | `stats_manager.py` |
| `src/web/` | Flask dashboard | `server.py`, `routes.py` |

### Configuration (`config/`)

| File | Purpose | Critical Settings |
|------|---------|-------------------|
| `botsort_reid.yaml` | Tracker + ReID config | `track_buffer`, `proximity_thresh`, `appearance_thresh` |
| `zone_config.json` | Zone definitions | `qr_zone`, `shelf_zone` (percentages) |

### Data (`data/`)

**All files here are gitignored** - runtime generated data only.

| File | Purpose | Shared Via |
|------|---------|------------|
| `shared_stats.json` | Statistics | StatsManager |
| `shared_customers.json` | Customer data | StatsManager |
| `shared_mqtt_events.json` | MQTT events (last 20) | StatsManager |

---

## Multi-Part Organization

### Part 1: Main Application (Python Backend)

**Root:** Project root directory
**Type:** Backend (Computer Vision)
**Entry Points:** `run_dashboard.py`, `main.py`

**Critical Directories:**
- `src/` - All Python source code
- `config/` - YAML/JSON configuration
- `models/` - YOLO model files
- `data/` - Runtime data (gitignored)

### Part 2: Embedded Component (ESP32)

**Root:** `code-weight-sensor/weight_sensor_esp32/`
**Type:** Embedded (MicroPython)
**Entry Point:** `main.py`

**Files:**
- `main.py` - Main application
- `boot.py` - Boot script
- `hx711.py` - HX711 driver library

---

## Integration Points

### MQTT Communication (Main ← ESP32)

```
ESP32 Publisher:
  File: code-weight-sensor/weight_sensor_esp32/main.py
  Topic: my-shop/shelf-1/events
  Format: "CHANGE:-480"

Python Subscriber:
  File: src/mqtt/client.py
  Callback: on_weight_event(weight_change_g, timestamp)
  Integrates: src/main_tracker.py
```

### Shared Stats (Tracker → Dashboard)

```
Tracker Process:
  Writes: data/shared_stats.json
  Writes: data/shared_customers.json
  Writes: data/shared_mqtt_events.json

Dashboard Process:
  Reads: data/shared_stats.json
  Reads: data/shared_customers.json
  Reads: data/shared_mqtt_events.json

Via: src/utils/stats_manager.py
```

---

## File Naming Conventions

| Pattern | Purpose | Example |
|---------|---------|---------|
| `*_config.json` | Configuration files | `zone_config.json` |
| `shared_*.json` | Inter-process data | `shared_stats.json` |
| `test_*.py` | Test scripts | `test_weight.py` |
| `*.md` | Documentation | `README.md` |
| `*_template.md` | Document templates | `project-context-template.md` |

---

## Critical Paths for Development

### When Adding Tracker Features

1. Edit `src/main_tracker.py` - Main tracker logic
2. Update `config/botsort_reid.yaml` - If changing thresholds
3. Add to `src/` subdirectory - If new module needed

### When Adding Web Endpoints

1. Add route in `src/web/routes.py`
2. Update HTML in `src/web/static/` if needed
3. Add stats sharing via `StatsManager` if needed

### When Adding Hardware Sensors

1. Create new ESP32 firmware in `code-weight-sensor/`
2. Add MQTT topic to `src/mqtt/client.py`
3. Handle events in `src/main_tracker.py`
4. Update zone config if spatial correlation needed

---

*Source tree analysis generated: 2026-01-14*
*Scan Level: Exhaustive*
