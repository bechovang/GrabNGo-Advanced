# GrabNGo-Advanced - Project Documentation Index

## Project Overview

- **Type:** Multi-part project with 2 parts
- **Primary Language:** Python (Backend) + MicroPython (Embedded)
- **Architecture:** Computer Vision + IoT Integration

---

## Quick Reference

### Part 1: Smart Retail Tracking System (Python Backend)

| Attribute | Value |
|-----------|-------|
| **Type** | Backend (Computer Vision) |
| **Tech Stack** | Python, YOLO, PyTorch, OpenCV, MediaPipe, MQTT, Flask |
| **Root** | `C:\Users\Admin\Desktop\GIT CLONE\GrabNGo-Advanced` |
| **Entry Point** | `run_dashboard.py` or `python main.py` |
| **Purpose** | Real-time customer tracking with YOLO pose estimation and MQTT integration |

### Part 2: Weight Sensor (ESP32/MicroPython)

| Attribute | Value |
|-----------|-------|
| **Type** | Embedded (MicroPython) |
| **Tech Stack** | MicroPython, HX711, MQTT |
| **Root** | `code-weight-sensor/weight_sensor_esp32/` |
| **Entry Point** | `main.py` |
| **Purpose** | Weight-based pickup detection with MQTT publishing |

---

## Generated Documentation

- [Project Context](../project-context.md) - **CRITICAL** - Authoritative rules and patterns for AI agents
- [Architecture - Main Application](./architecture-main.md) - Detailed architecture for Python backend
- [Architecture - ESP32](./architecture-esp32.md) - Detailed architecture for embedded component
- [Source Tree Analysis](./source-tree-analysis.md)
- [Component Inventory](./component-inventory.md)
- [Development Guide](./development-guide.md)

---

## Existing Documentation

### Main Documentation

- [README.md](../README.md) - Project overview and usage instructions
- [docs/README_UNIFIED.md](./README_UNIFIED.md) - Unified system documentation

### Setup & Installation

- [docs/setup/INSTALL.md](./setup/INSTALL.md) - Installation instructions
- [docs/setup/SYSTEM_SETUP.md](./setup/SYSTEM_SETUP.md) - Complete system setup guide

### Features

- [docs/features/DASHBOARD_README.md](./features/DASHBOARD_README.md) - Dashboard feature documentation
- [docs/features/MANUAL_CONFIRMATION.md](./features/MANUAL_CONFIRMATION.md) - Manual confirmation system
- [docs/features/QR_CONFIRMATION_GUIDE.md](./features/QR_CONFIRMATION_GUIDE.md) - QR code confirmation
- [docs/features/README_QR_SYSTEM.md](./features/README_QR_SYSTEM.md) - QR system overview
- [docs/features/WEIGHT_PICKUP_DETECTION.md](./features/WEIGHT_PICKUP_DETECTION.md) - Weight-based detection
- [docs/features/ZONE_CONTROLS_README.md](./features/ZONE_CONTROLS_README.md) - Zone control system
- [docs/HOLDING_DETECTION.md](./HOLDING_DETECTION.md) - Object holding detection

### Troubleshooting & Testing

- [docs/troubleshooting/TROUBLESHOOTING_WEB.md](./troubleshooting/TROUBLESHOOTING_WEB.md) - Web dashboard troubleshooting
- [docs/QUICK_TEST_GUIDE.md](./QUICK_TEST_GUIDE.md) - Quick testing guide

### Progress & Other

- [docs/BAO_CAO_TIEN_DO.md](./BAO_CAO_TIEN_DO.md) - Progress report (Vietnamese)
- [docs/readme_yolo.md](./readme_yolo.md) - YOLO-specific documentation

### ESP32 Documentation

- [code-weight-sensor/HUONG_DAN_NAP_CODE_ESP32.md](../code-weight-sensor/HUONG_DAN_NAP_CODE_ESP32.md) - ESP32 code loading guide (Vietnamese)

---

## Getting Started

### For the Main Application (Python)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download YOLO models** (if not already present):
   - `yolo11n-pose.pt` - Pose estimation model
   - `yolo11n-cls.pt` - Classification model (for ReID)

3. **Run the system:**
   ```bash
   # Dashboard only (MQTT monitoring)
   python run_dashboard.py

   # Full system with camera (if main.py exists)
   python main.py
   ```

4. **Access dashboard:**
   - Dashboard: http://localhost:8081/dashboard
   - Mobile app: http://localhost:8081/

### For the ESP32 Component

1. **Install MicroPython** on ESP32
2. **Copy files** to ESP32:
   - `boot.py`
   - `hx711.py`
   - `main.py`
3. **Configure WiFi** in `main.py`
4. **Flash and run** - ESP32 will auto-connect to MQTT broker

---

## Key Concepts

### Tracking Pipeline

1. **Detection**: YOLO pose model detects people in frame
2. **Tracking**: BoT-SORT assigns track IDs and handles occlusions
3. **ReID**: Lightweight ReID extracts appearance features (LAB, HOG, texture, edge)
4. **Validation**: New tracks are validated (samples, confidence, feature quality)
5. **Confirmation**: Validated tracks can be manually confirmed via QR code
6. **Re-identification**: Lost tracks are matched using appearance features

### Integration Between Parts

The Python backend and ESP32 communicate via MQTT:

| Component | Role | MQTT Topic |
|-----------|------|------------|
| **ESP32** | Publisher (weight events) | `my-shop/shelf-1/events` |
| **Python** | Subscriber (weight events) | `my-shop/shelf-1/events` |

**Message Format:** `"CHANGE:-480"` (weight change in grams)

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/botsort_reid.yaml` | BoT-SORT tracker and ReID configuration |
| `config/zone_config.json` | QR zone and shelf zone definitions |

---

## Project Parts Summary

### Repository Type: Multi-Part

| Part | Name | Technology | Purpose |
|------|------|------------|---------|
| **main** | Smart Retail Tracking System | Python, YOLO, MQTT, Flask | CV-based customer tracking |
| **esp32** | Weight Sensor | MicroPython, HX711 | Weight-based pickup detection |

---

## AI Agent Usage

**When implementing new features:**

1. **FIRST** - Read `project-context.md` for authoritative rules
2. **THEN** - Read relevant architecture document
3. **FINALLY** - Follow the patterns and conventions defined

**Critical:**
- ALL new code goes in `src/` directory
- Use type hints for function signatures
- Follow existing error handling patterns
- Update documentation when adding features

---

*Documentation generated: 2026-01-14*
*Scan Level: Exhaustive (all source files)*
*Workflow: document-project v1.2.0*
