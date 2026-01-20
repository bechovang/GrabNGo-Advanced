# Architecture: Weight Sensor (ESP32/MicroPython)

**Part ID:** `esp32`
**Project Type:** Embedded (MicroPython)
**Technology Stack:** MicroPython, HX711, MQTT

---

## Executive Summary

The Weight Sensor is an embedded system built on ESP32 running MicroPython. It continuously monitors a load cell via the HX711 amplifier and publishes weight change events to an MQTT broker. The system is designed for retail shelf monitoring to detect when items are picked up or returned.

**Key Capabilities:**
- Real-time weight monitoring (5 Hz sampling)
- Stable readings using median filter
- WiFi auto-connect and reconnection
- MQTT publishing for weight changes
- Configurable change threshold (50g default)
- Error recovery and auto-reconnect

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Platform** | ESP32 | Microcontroller with WiFi |
| **Language** | MicroPython | Firmware |
| **Sensor Driver** | HX711 | Load cell amplifier (24-bit ADC) |
| **Communication** | umqtt.simple | MQTT client library |
| **Networking** | network.WLAN | WiFi connectivity |

---

## Architecture Pattern

**Pattern:** Event-Driven Publisher with State Machine

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   HX711     │───▶│   Weight    │───▶│   Change    │───▶│    MQTT     │
│  Sensor     │    │  Reading    │    │  Detection  │    │   Publish   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                            │                  │                  │
                    ┌───────┴─────┐    ┌──────┴──────┐    ┌─────┴─────┐
                    │   Median    │    │  Threshold  │    │   Broker   │
                    │   Filter    │    │  Check      │    │           │
                    └─────────────┘    └─────────────┘    └───────────┘
```

---

## Hardware Configuration

### Pin Assignments

| Pin | Function | Description |
|-----|----------|-------------|
| GPIO 25 | DT (DOUT) | HX711 Data Out |
| GPIO 26 | SCK (PD_SCK) | HX711 Serial Clock |

### Load Cell Specifications

| Parameter | Value |
|-----------|-------|
| **Max Capacity** | Typically 5kg-10kg (depends on hardware) |
| **ADC Resolution** | 24-bit (HX711) |
| **Gain Setting** | 128 (default) |
| **Update Rate** | 10 SPS (samples per second) |

---

## Calibration Values

**Current Configuration:**
```python
TARE_VALUE = 471778          # Raw reading with no load
VALUE_WITH_WEIGHT = 256326   # Raw reading with known weight
KNOWN_WEIGHT_G = 480         # Known weight in grams
RATIO = (VALUE_WITH_WEIGHT - TARE_VALUE) / KNOWN_WEIGHT_G
```

**Recalibration:**
1. Run without load → record `TARE_VALUE`
2. Run with known weight → record `VALUE_WITH_WEIGHT`
3. Update values in `main.py`
4. Use `calibrate.py` utility script

---

## Component Overview

### 1. HX711 Driver

**File:** `hx711.py`

**Class:** `HX711`

**Key Methods:**
```python
class HX711:
    def __init__(self, d_out, pd_sck, gain=128):
        # Initialize HX711 with pin assignments

    def read(self):
        # Read single raw 24-bit value (signed)

    def read_average(self, times=16):
        # Read multiple times and return average

    def tare(self, times=16):
        # Set current reading as zero offset

    def get_weight(self, times=16):
        # Convert reading to weight using calibration
```

**Features:**
- 24-bit signed integer output
- Configurable gain (32, 64, 128)
- Power down/up modes for energy saving

---

### 2. Main Application

**File:** `main.py`

**Workflow:**
```
1. Initialize HX711
2. Connect to WiFi
3. Connect to MQTT broker
4. Read initial weight (baseline)
5. Loop:
   a. Read stable weight
   b. Calculate change from baseline
   c. If |change| > threshold:
      - Publish MQTT event
      - Update baseline
   d. Sleep 200ms
```

**Key Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `WEIGHT_CHANGE_THRESHOLD` | 50g | Minimum change to trigger event |
| `SAMPLE_COUNT` | 10 | Median filter samples |
| `LOOP_DELAY` | 0.2s | Main loop sleep time |

---

### 3. Boot Script

**File:** `boot.py`

**Purpose:** Executes on device boot/reset before main.py

**Typical Content:**
```python
# Optional boot configuration
import machine
# machine.freq(240000000)  # Overclock to 240MHz
```

---

## Data Architecture

### MQTT Message Format

**Topic:** `my-shop/shelf-1/events`

**Payload:** `"CHANGE:<weight_change_g>"`

**Examples:**
- `"CHANGE:-480"` - 480g removed (item picked up)
- `"CHANGE:480"` - 480g added (item returned)
- `"CHANGE:-120"` - 120g removed (small item)

### Weight Reading Algorithm

**Median Filter (Stable Reading):**
```python
def read_weight_stable(samples=10):
    readings = []
    for _ in range(samples):
        readings.append(hx.read())
    return sorted(readings)[len(readings) // 2]  # Median
```

**Change Detection:**
```python
weight_change = current_weight - last_known_weight
if abs(weight_change) > WEIGHT_CHANGE_THRESHOLD:
    publish(f"CHANGE:{round(weight_change)}")
    last_known_weight = current_weight
```

---

## Source Tree

```
code-weight-sensor/
└── weight_sensor_esp32/
    ├── main.py          # Main entry point
    ├── boot.py          # Boot script
    ├── hx711.py         # HX711 driver library
    ├── calibrate.py     # Calibration utility
    └── test_weight.py   # Testing utility
```

---

## Network Configuration

### WiFi Settings

**Current Configuration:**
```python
WIFI_SSID = "Hshop Guest"
WIFI_PASSWORD = "dienturobot"
```

**To Configure:**
1. Edit `main.py`
2. Update `WIFI_SSID` and `WIFI_PASSWORD`
3. Re-flash to ESP32

### WiFi Connection Flow

```
1. Initialize WLAN in STA mode
2. Scan for networks (debug)
3. Connect to configured SSID
4. Wait for connection (max 20s)
5. Print IP address
6. Continue to MQTT connection
```

---

## MQTT Configuration

**Broker Settings:**
```python
MQTT_BROKER = "test.mosquitto.org"
MQTT_CLIENT_ID = "esp32-shelf-1"
MQTT_TOPIC = "my-shop/shelf-1/events"
```

**Connection Flow:**
```
1. Create MQTTClient
2. Connect to broker
3. On connection success → start main loop
4. On connection failure → retry every 5s
```

---

## Development Workflow

### Prerequisites

**Hardware Required:**
- ESP32 development board
- HX711 load cell amplifier
- Load cell (typically 5kg capacity)
- Jumper wires
- Breadboard or PCB

**Software Required:**
- MicroPython firmware for ESP32
- Thonny IDE or ampy for file transfer

### Flashing MicroPython

1. **Download MicroPython:**
   - Visit: https://micropython.org/download/ESP32/

2. **Flash to ESP32:**
   ```bash
   esptool.py --chip esp32 --port COMX write_flash -z 0x1000 esp32-micropython.bin
   ```

3. **Verify:**
   - Open serial console (115200 baud)
   - Should see MicroPython REPL

### Uploading Files

**Using Thonny:**
1. Connect to ESP32
2. Open files in Thonny
3. Upload to device (Ctrl+Shift+U)

**Using ampy:**
```bash
ampy --port COMX put boot.py
ampy --port COMX put hx711.py
ampy --port COMX put main.py
```

### Testing

**Test HX711:**
```python
# Run test_weight.py on device
# Should show raw readings
```

**Test MQTT:**
```bash
# Subscribe on host
mosquitto_sub -h test.mosquitto.org -t "my-shop/shelf-1/events"
# Add/remove weight on sensor
# Should see messages
```

---

## Deployment Architecture

### Physical Setup

```
┌─────────────────┐
│   Shelf Unit    │
│                 │
│  ┌──────────┐   │
│  │ Load     │   │     HX711      ESP32
│  │ Cell     ├───┼─────┬──────────┬─────┬─────▶ MQTT
│  └──────────┘   │     │          │     │      Broker
│                 │     │          │     │
│  ┌──────────┐   │     │          │     │
│  │ Items    │   │   ┌──┴──┐   ┌──┴──┐ │
│  │ to       │   │   │HX711│   │ESP32│ │
│  │ monitor  │   │   └─────┘   └─────┘ │
│  └──────────┘   │                    │
└─────────────────┘                    │
                                        │
                                    WiFi
```

### Network Requirements

- **WiFi:** 2.4GHz (ESP32 doesn't support 5GHz)
- **Protocol:** TCP/IP
- **Firewall:** Allow outbound to MQTT broker port 1883
- **Power:** 5V DC (via USB or external supply)

---

## Known Limitations

1. **Public MQTT Broker:** `test.mosquitto.org` may be unreliable
2. **WiFi Range:** Limited by ESP32 antenna
3. **Power:** Requires continuous power (no battery mode)
4. **Calibration:** Manual recalibration required for new sensors
5. **Load Cell:** Drift over temperature changes

---

## Troubleshooting

### HX711 Reading Zero

**Causes:**
- Wrong pin connections
- HX711 not powered
- Load cell not connected
- Faulty HX711 module

**Solutions:**
1. Check DT (GPIO 25) and SCK (GPIO 26) wiring
2. Verify HX711 VCC/GND connected
3. Check load cell wiring (E+, E-, A+, A-)

### WiFi Connection Failed

**Causes:**
- Wrong SSID/password
- WiFi out of range
- 5GHz network (ESP32 only supports 2.4GHz)

**Solutions:**
1. Verify SSID and password in main.py
2. Move closer to router
3. Check for 2.4GHz WiFi availability

### MQTT Connection Timeout

**Causes:**
- Broker offline
- Network/firewall blocking port 1883
- Wrong broker address

**Solutions:**
1. Test broker availability: `ping test.mosquitto.org`
2. Try local MQTT broker
3. Check firewall settings

---

*Architecture document generated: 2026-01-14*
*Scan Level: Exhaustive*
*Part: esp32 (Weight Sensor)*
