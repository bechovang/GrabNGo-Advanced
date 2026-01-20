# Development Guide - GrabNGo-Advanced

## Overview

This guide provides comprehensive instructions for setting up, developing, testing, and deploying the GrabNGo-Advanced system.

---

## Prerequisites and Dependencies

### System Requirements

**For Main Application (Python):**
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- GPU with CUDA support (optional, for YOLO acceleration)
- Camera (USB or RTSP stream)
- Network connection for MQTT

**For ESP32 Component:**
- ESP32 development board
- HX711 load cell amplifier
- Load cell (5kg typical)
- Micro USB cable
- WiFi network (2.4GHz only)

### Python Dependencies

**Install:**
```bash
pip install -r requirements.txt
```

**Contents:**
```
ultralytics>=8.0.0      # YOLO pose estimation
opencv-python>=4.8.0    # Image processing
numpy>=1.24.0           # Numerical operations
pillow>=9.5.0          # Image utilities
torch>=2.0.0            # Deep learning backend
torchvision>=0.15.0     # Torch vision utilities
mediapipe>=0.10.0       # Hand/pose detection
paho-mqtt>=1.6.0        # MQTT client
```

---

## Environment Setup

### 1. Python Virtual Environment

**Create Virtual Environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Verify Installation:**
```bash
python --version  # Should show Python 3.8+
pip list          # Should show installed packages
```

### 2. YOLO Models

**Download Models:**
```bash
# YOLO Pose Estimation
# Download from: https://github.com/ultralytics/assets/releases
# Save to: models/yolo11n-pose.pt

# YOLO Classification (for ReID)
# Save to: models/yolo11n-cls.pt
```

**Or Auto-Download (First Run):**
The models will be auto-downloaded on first run by Ultralytics.

### 3. ESP32 MicroPython

**Flash MicroPython:**
```bash
# Download firmware: https://micropython.org/download/ESP32/
# Flash using esptool:
esptool.py --chip esp32 --port COMX write_flash -z 0x1000 esp32-micropython.bin
```

**Upload Files:**
```bash
# Using Thonny IDE
# Or using ampy:
ampy --port COMX put boot.py
ampy --port COMX put hx711.py
ampy --port COMX put main.py
```

---

## Local Development Commands

### Running the System

**Dashboard Only (No Camera):**
```bash
python run_dashboard.py
# Access: http://localhost:8081/dashboard
# Access: http://localhost:8081/ (mobile QR)
```

**Full System (With Camera):**
```bash
python main.py
# Access: http://localhost:8080/dashboard
```

### Testing Components

**Test Camera:**
```bash
# If test script exists
python test_camera.py
```

**Test MQTT Connection:**
```bash
# Subscribe to topic
mosquitto_sub -h test.mosquitto.org -t "my-shop/shelf-1/events"
```

**Test HX711:**
```python
# On ESP32, run:
exec(open('test_weight.py').read())
```

---

## Build Process

### No Build Required

This project uses interpreted languages (Python, MicroPython), so no compilation is required.

**For Distribution:**
1. Create Python wheel for main application
2. Package ESP32 firmware as `.bin` files

---

## Testing Approach

### Unit Tests

**Test Individual Components:**
```python
# Test ReID feature extraction
from src.tracker.reid import LightweightReID
reid = LightweightReID()
features = reid.extract_features(frame, bbox)

# Test MQTT client
from src.mqtt.client import MQTTClient
client = MQTTClient(broker="test.mosquitto.org")
client.connect()
```

### Integration Tests

**Test MQTT Flow:**
1. Start ESP32 sensor
2. Subscribe to MQTT topic on host
3. Add/remove weight on sensor
4. Verify messages received

**Test QR Confirmation:**
1. Start dashboard
2. Stand in QR zone with camera
3. Scan QR code with mobile
4. Verify customer confirmed

### Manual Testing Checklist

- [ ] Camera connects and displays frames
- [ ] People detected and tracked
- [ ] Tracks show PENDING → VALIDATED → CONFIRMED
- [ ] MQTT weight events received
- [ ] Dashboard shows real-time data
- [ ] QR code confirmation works
- [ ] Mobile web interface accessible

---

## Common Development Tasks

### Adding a New Tracker Feature

1. **Edit `src/main_tracker.py`:**
   ```python
   class RetailCustomerTracker:
       def new_feature(self, params):
           # Implementation
           pass
   ```

2. **Update config if needed:**
   ```yaml
   # config/botsort_reid.yaml
   new_feature_param: value
   ```

3. **Add shared stats if needed:**
   ```python
   from src.utils.stats_manager import StatsManager
   stats_manager = StatsManager()
   stats_manager.save_stats(new_data)
   ```

### Adding a New Web Endpoint

1. **Add route in `src/web/routes.py`:**
   ```python
   @app.route('/new-endpoint', methods=['GET'])
   def new_endpoint():
       # Implementation
       return jsonify(data)
   ```

2. **Update HTML if needed:**
   ```html
   <!-- src/web/static/dashboard.html -->
   <script>
   fetch('/new-endpoint')
       .then(r => r.json())
       .then(data => console.log(data));
   </script>
   ```

### Adding a New Hardware Sensor

1. **Create ESP32 firmware:**
   ```python
   # code-weight-sensor/new_sensor/main.py
   # Read sensor → Publish MQTT
   ```

2. **Add MQTT subscription:**
   ```python
   # src/mqtt/client.py
   class MQTTClient:
       def __init__(self, topics=["new-sensor/events"]):
           # Subscribe to new topic
   ```

3. **Handle events in tracker:**
   ```python
   # src/main_tracker.py
   def _handle_new_sensor_event(self, data):
       # Process sensor data
   ```

### Modifying Zone Definitions

**Edit `config/zone_config.json`:**
```json
{
  "qr_zone": {
    "x1_percent": 0.6,
    "y1_percent": 0.0,
    "x2_percent": 1.0,
    "y2_percent": 1.0
  }
}
```

**Values are percentages of frame (0.0 to 1.0)**

---

## Code Style and Conventions

### Python Code Style

**Follow PEP 8:**
- 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints for function signatures
- Document classes and functions with docstrings

**Example:**
```python
class MyClass:
    """Brief description of the class."""

    def my_method(self, param: str) -> bool:
        """
        Brief description of the method.

        Args:
            param: Description of param

        Returns:
            bool: Description of return value
        """
        return True
```

### Module Structure

**ALL new code goes in `src/` directory:**
```
src/
├── __init__.py
├── new_module/
│   ├── __init__.py
│   └── feature.py
```

### Error Handling Pattern

**Silent fail for non-critical paths:**
```python
try:
    risky_operation()
except Exception as e:
    pass  # Don't break main process
```

**Logging with prefixes:**
```python
print("[INFO] System started")
print("[WARN] MQTT not connected")
print("[ERROR] Camera not found")
```

---

## Configuration Management

### Configuration Files

| File | Format | Purpose |
|------|--------|---------|
| `config/botsort_reid.yaml` | YAML | Tracker + ReID parameters |
| `config/zone_config.json` | JSON | Zone definitions (percentages) |

### Environment Variables

**Currently hardcoded in files:**
- MQTT broker address
- WiFi credentials (ESP32)
- Zone coordinates

**TODO:** Consider moving to `.env` file for production

---

## Debugging

### Enable Debug Logging

**In `src/main_tracker.py`:**
```python
DEBUG = True
# Add print statements for debugging
if DEBUG:
    print(f"[DEBUG] Track {track_id} updated")
```

### Common Issues

**Issue: Camera not detected**
- Check USB connection
- Try different camera index (0, 1, 2...)
- Verify driver installed

**Issue: MQTT connection timeout**
- Check network connectivity
- Verify broker is online: `ping test.mosquitto.org`
- Try local MQTT broker

**Issue: YOLO model not found**
- Check `models/` directory exists
- Verify model file names match code
- Check file permissions

---

## Deployment

### Production Deployment

**For Main Application:**
1. Set up headless server (Ubuntu recommended)
2. Install dependencies: `pip install -r requirements.txt`
3. Configure auto-start (systemd service)
4. Use GPU server for better performance

**For ESP32:**
1. Calibrate sensor in actual environment
2. Configure WiFi credentials
3. Set up local MQTT broker (Mosquitto)
4. Power with reliable USB supply

### Systemd Service Example

**Create `/etc/systemd/system/grabngo.service`:**
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

[Install]
WantedBy=multi-user.target
```

**Enable service:**
```bash
sudo systemctl enable grabngo
sudo systemctl start grabngo
sudo systemctl status grabngo
```

---

## Performance Optimization

### GPU Acceleration

**Install CUDA-enabled PyTorch:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU available:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Resolution Optimization

**Lower resolution for faster FPS:**
```python
frame = cv2.resize(frame, (640, 480))  # Instead of 1920x1080
```

### Skip Frames for Performance

```python
frame_skip = 2
if frame_count % frame_skip == 0:
    process_frame(frame)
```

---

*Development guide generated: 2026-01-14*
*Scan Level: Exhaustive*
