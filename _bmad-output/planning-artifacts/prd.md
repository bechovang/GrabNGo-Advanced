---
stepsCompleted: ["step-01-init", "step-02-discovery", "step-03-success", "step-04-journeys", "step-05-domain", "step-06-innovation:skipped", "step-07-project-type", "step-08-scoping", "step-09-functional", "step-10-nonfunctional", "step-11-polish"]
inputDocuments: ["project-context.md", "docs/index.md", "docs/architecture-main.md", "docs/architecture-esp32.md", "docs/source-tree-analysis.md", "docs/component-inventory.md", "docs/development-guide.md"]
workflowType: 'prd'
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 21
projectType: 'brownfield'
classification:
  projectType: 'iot_embedded_multi_part'
  domain: 'retail_commerce_cv_iot'
  complexity: 'high'
  projectContext: 'brownfield'
---

# Product Requirements Document - GrabNGo-Advanced

**Author:** Admin
**Date:** 2026-01-14

---

## Executive Summary

GrabNGo-Advanced is an autonomous retail checkout system using computer vision and IoT sensors to enable customers to shop and exit without manual checkout.

**Core Differentiator:** Real-time sensor fusion of YOLO-based person tracking with ESP32 weight sensors enables accurate cart detection without requiring customer app interaction or shelf-mounted cameras.

**Target Users:**
- Retail store owners seeking labor cost reduction
- Customers wanting faster, frictionless shopping experience
- Store managers needing real-time operational visibility

**Value Proposition:**
- 75% reduction in checkout staffing
- 50% faster shopping trips for customers
- ≥95% cart accuracy through lightweight ReID (≥90% after occlusion)

**Technology Edge:**
- Multi-part IoT/Embedded system with edge computing
- 512-dim appearance features for re-identification
- Zone-based spatial correlation for sensor fusion
- Real-time MQTT weight event integration

---

## Success Criteria

### User Success

**The "Aha!" Moment:** When a customer picks up an item, walks out of the store, and receives an accurate receipt without ever stopping at a checkout line.

**Completion Scenarios:**
- Customer completes shopping trip in 50% less time than traditional checkout
- Shopping cart accuracy ≥95% (correct items detected)
- Zero false accusations (no customer charged for items they didn't take)
- QR confirmation completes within 10 seconds

**Emotional Success States:**
- **Delighted:** "I just walked out and it knew exactly what I got!"
- **Relieved:** "I don't have to wait in line anymore"
- **Empowered:** "I can shop at my own pace"

**Failure Moments to Avoid:**
- False positive charge (customer disputes item)
- System timeout at QR zone
- Tracking lost mid-shopping trip
- Receipt shows wrong items

### Business Success

**3-Month Success (Proof of Concept):**
- Process ≥100 customers/day with ≤5% error rate
- Customer satisfaction score ≥4.0/5.0
- Reduce checkout wait time by 50% vs manual
- System uptime ≥95% during business hours

**12-Month Success (Production Validated):**
- Process ≥500 customers/day with ≤2% error rate
- Customer satisfaction score ≥4.5/5.0
- 75% reduction in checkout staffing needs
- System uptime ≥99% during business hours
- Positive ROI vs traditional checkout (labor savings - system costs)

**Key Business Metrics:**
- **Error Rate:** False positives ≤1%, False negatives ≤2%
- **Throughput:** 1 customer/minute sustained
- **Adoption:** 80% of customers use autonomous checkout (vs opting out)

### Technical Success

**Real-Time Tracking:**
- Process video at ≥30 FPS with no dropped frames
- Track ≥5 simultaneous customers with ≤5% ID confusion
- ReID re-identification accuracy ≥90% after 10-second occlusion
- MQTT weight event latency ≤2 seconds from sensor to dashboard

**System Reliability:**
- Uptime ≥99% during business hours
- Auto-recovery from WiFi/MQTT disconnection within 30 seconds
- Camera fallback: If primary camera fails, alert staff within 10 seconds

**Accuracy Thresholds:**
- Person detection confidence ≥0.6 (from YOLO)
- ReID similarity threshold: 0.1 cosine similarity
- Weight change detection: ±50g threshold (prevents false positives)

**Edge Case Handling:**
- Multiple customers in QR zone: System requires single occupant or displays "Please wait"
- Occlusion handling: Track buffer = 300 frames (10 seconds @ 30fps)
- Sensor failure: Dashboard alerts when ESP32 weight sensor offline >60 seconds

### Measurable Outcomes

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Checkout Time** | ≤30 seconds/customer | Dashboard timestamp (entry → exit) |
| **Cart Accuracy** | ≥95% | Dispute rate <5% |
| **False Positive Rate** | ≤1% | Manual audit of flagged events |
| **System Uptime** | ≥99% | MQTT connection + tracking events logs |
| **Processing Speed** | ≥30 FPS | Frame processing timestamps |
| **Customer Satisfaction** | ≥4.5/5.0 | Post-visit survey (optional) |
| **Simultaneous Tracking** | ≥5 customers | Active tracks counter |
| **ReID After Occlusion** | ≥90% | Track re-identification events |

---

## Product Scope

GrabNGo-Advanced is a **brownfield project** with a production-ready core already implemented. The system consists of a Python computer vision backend (YOLO pose estimation, BoT-SORT tracking, Lightweight ReID) and an ESP32 embedded weight sensor (HX711 + MQTT).

**Current State:**
- YOLO11n-pose person detection with 300-frame track buffer for occlusion handling
- 512-dim appearance features (LAB + HOG + texture + edge) for re-identification
- MQTT weight sensor integration with zone-based spatial correlation
- QR code confirmation workflow with Flask dashboard
- Multi-process architecture (tracker + dashboard with shared stats)

**Development Phases:**
- **MVP:** Validate CV accuracy (≥95% cart accuracy, ≥90% ReID after occlusion) in single-store deployment
- **Growth:** Multi-camera support, holding detection, advanced ReID, payment integration
- **Vision:** Chain-wide deployment with AI shopping assistant and predictive restocking

Detailed scoping decisions, risk mitigations, and phase-specific feature sets are documented in the Project Scoping & Phased Development section below.

---

## User Journeys

### 1. Shopper Journey - Happy Path (Autonomous Shopping)

**Persona:** Sarah Chen, 32, busy professional

**Situation:** Sarah just finished work and needs to grab a few items before heading home. She's tired, hungry, and dreading the checkout line at the grocery store.

**Goal:** Get in, get what she needs, and get home quickly—without the stress of waiting in line.

**Obstacle:** Traditional checkout means standing in line for 10-15 minutes, fumbling with payment, dealing with slow cashiers.

---

**Opening Scene - Store Entry:**
Sarah walks into the store at 6:15 PM. She's been here before, so she knows the drill. She pulls out her phone and opens the store's mobile web app. The camera system detects her as she enters the shopping area. She appears on the dashboard as "Track #1 - PENDING".

**Rising Action - Shopping Experience:**
Sarah heads to the beverage aisle first. As she picks up a bottled coffee, the ESP32 weight sensor on the shelf detects the weight change (-480g) and publishes an MQTT event. The system correlates this with Track #1's position in the shelf zone. Her shopping cart is updated: `[Bottled Coffee - $4.99]`.

She moves to the snack aisle, picks up some granola bars. Another MQTT event, another update: `[Bottled Coffee - $4.99, Granola Bars - $3.99]`.

She's in and out in 8 minutes total.

**Climax - QR Confirmation:**
Sarah heads to the exit. The system directs her to the QR confirmation zone on the right side of the store. She stands alone in the zone—no other customers are nearby. Her phone shows a green "Ready to Scan" button.

She scans the QR code displayed on the confirmation screen. Her phone prompts: "Confirm your identity: CUST_0042". She taps "Confirm". The dashboard updates: "Track #1 - CONFIRMED as CUST_0042". Her shopping cart is locked and finalized.

**Resolution - Exit and Receipt:**
Sarah walks out the door. The system logs her exit time: 6:23 PM. Total shopping time: 8 minutes. Total items: 2. Total: $8.98.

An email arrives in her inbox: "Thank you for shopping with us! Your receipt: 2 items, $8.98."

Sarah smiles. She just saved 15 minutes of her life. No line. No waiting. No fumbling with payment. She got what she needed and got on with her day.

---

### 2. Shopper Journey - Edge Case (Multiple Customers in QR Zone)

**Persona:** Mike Rodriguez, 45, father of three

**Situation:** It's Saturday afternoon and Mike has his two kids with him. The store is busy. He just needs to grab some formula for the baby.

**Goal:** Get in, get the formula, get out before the kids start acting up.

**Obstacle:** The store is crowded. When he gets to the QR zone, another customer is already there confirming their purchase.

---

**Opening Scene - Entry and Shopping:**
Mike enters with his kids. The system detects three people: "Track #1, #2, #3 - PENDING". He grabs the formula canister—weight sensor detects it. His cart updates: `[Baby Formula - $24.99]`.

**Rising Action - The QR Zone Conflict:**
Mike heads to exit, formula in hand. He enters the QR zone, but another shopper is already there, scanning their QR code. The system detects two tracks in the zone simultaneously.

Mike's phone shows: "Please wait—another customer is confirming. We'll call you next."

**Climax - The Wait:**
Mike stands there for 30 seconds with two restless kids. The first customer finishes and leaves. The system detects Mike is now alone in the zone. His phone updates: "Ready to Scan."

**Resolution - Success After Wait:**
Mike scans the QR code, confirms his identity, and walks out. Total shopping time: 12 minutes (including the 30-second wait). The system worked, but the edge case exposed a pain point—during busy times, customers might wait.

**Journey Requirements Revealed:**
- Queue management for QR zone
- Clear communication when waiting
- Priority system for families/customers with needs?

---

### 3. Store Manager Journey (Dashboard Monitoring)

**Persona:** David Kim, 38, store manager

**Situation:** David is responsible for store operations, inventory, and customer satisfaction. He needs to know what's happening in the store in real-time.

**Goal:** Keep operations running smoothly, catch issues before they become problems, ensure customers are happy.

**Obstacle:** Traditional stores lack visibility—you don't know there's a problem until a customer complains.

---

**Opening Scene - Morning Setup:**
David arrives at 8:00 AM. He opens the dashboard on his tablet: `http://store-dashboard.local:8081/dashboard`. He sees:
- System Status: ✅ Online
- Camera: ✅ Connected
- MQTT: ✅ Connected to broker
- Active Customers: 0

**Rising Action - Mid-Day Monitoring:**
At 2:00 PM, David checks the dashboard during the busy period:
- Active Customers: 3
- Today's Total: 47
- Accuracy Rate: 96% (1 false positive detected earlier)
- MQTT Events: 142 weight changes logged

He notices a discrepancy: Customer CUST_0042 was charged for an item they dispute. David pulls up the tracking events log, sees the weight event correlation, and realizes the customer actually picked up two items but was only charged for one.

**Climax - Real-Time Intervention:**
David sees the alert: "ESP32 weight sensor offline—last seen 2 minutes ago." He checks the shelf—the sensor's WiFi connection dropped. He power-cycles the ESP32, and it reconnects within 30 seconds.

**Resolution - End of Day Review:**
At closing, David reviews the day's metrics:
- Total Customers: 127
- Accuracy Rate: 94.5% (within acceptable range)
- System Uptime: 99.2% (dip due to sensor reboot)
- Revenue: $2,847

He exports the customer logs and prepares the bank deposit. The system worked—and he caught the sensor issue before it affected more customers.

**Journey Requirements Revealed:**
- Real-time dashboard with key metrics
- Alert system for offline sensors
- Exportable transaction logs
- Historical data review capabilities

---

### 4. System Administrator Journey (Initial Setup & Calibration)

**Persona:** Alex Turner, 28, IT support specialist

**Situation:** The store is deploying the GrabNGo-Advanced system for the first time. Alex is responsible for getting everything installed and calibrated.

**Goal:** Get the system up and running with accurate weight detection and reliable tracking.

**Obstacle:** This is complex hardware + software integration. Calibration is finicky. WiFi signals are unreliable in the store environment.

---

**Opening Scene - Hardware Installation:**
Alex mounts the ESP32 and HX711 amplifier under the shelving unit. Connects the DT and SCK pins to GPIO 25 and 26. Powers it via USB. The ESP32 boots and shows the heartbeat on the serial console.

**Rising Action - Calibration Process:**
Alex runs the calibration script. The system reads the raw HX711 value: `471778`. That's the tare (zero) value. He places a known 480g weight on the shelf. The system reads: `256326`. The script calculates the ratio: `-452.4` grams per raw unit.

He tests it with different items—a water bottle (500g), a bag of chips (120g). The readings are within ±10g. Acceptable.

**Climax - Network Integration:**
Alex configures the WiFi credentials in `main.py`. The ESP32 scans, connects, and shows the IP address. He tests the MQTT connection: `mosquitto_sub -h test.mosquitto.org -t "my-shop/shelf-1/events"`.

He places a water bottle on the shelf. Within 2 seconds, the message appears: `"CHANGE:-500"`. The system is working.

**Resolution - Production Deployment:**
Alex documents the calibration values, backs up the firmware, and creates a runbook for troubleshooting. The system goes live the next day.

**Journey Requirements Revealed:**
- Calibration utilities and scripts
- Network testing tools
- MQTT verification commands
- Documentation and runbooks
- Configuration management

---

### Journey Requirements Summary

Based on these narratives, the system requires these key capabilities:

**For Shoppers:**
- QR zone detection and queue management
- Clear status communication (ready/wait/error)
- Fast confirmation flow (<10 seconds)
- Accurate item detection and correlation
- Email/digital receipt delivery

**For Store Managers:**
- Real-time dashboard with metrics
- Alert system for sensor failures
- Transaction logs and export
- Historical data review
- Dispute resolution tools

**For System Administrators:**
- Calibration workflows and utilities
- Network testing and diagnostics
- MQTT message inspection tools
- Configuration backup/restore
- Troubleshooting guides

**For Support Staff:**
- System health monitoring
- Error logs and debugging tools
- Remote access capabilities
- Maintenance scheduling

---

## Domain-Specific Requirements

### Compliance & Regulatory

**Data Privacy:**
- **Customer Consent:** Clear opt-in mechanism for tracking when customers enter the store. Signs at entrance + mobile app consent screen
- **Data Retention:** Video logs: 7-30 days (configurable), Transaction logs: 2-7 years (tax/audit requirements)
- **Right to Deletion:** Customers can request deletion of their tracking data under GDPR/CCPA
- **Biometric Data:** ReID feature vectors (512-dim) are pseudonymized, not stored with customer identity

**Payment Processing:**
- **PCI DSS:** When payment integration is added, compliance with Payment Card Industry Data Security Standard
- **Receipt Regulations:** Transaction records must include itemized list, timestamp, customer ID, total amount

**Jurisdiction-Specific:**
- **GDPR (EU):** Right to access, right to erasure, data portability, privacy by design
- **CCPA (California):** Do not sell my data, right to opt-out, right to deletion
- **Biometric Laws:** Some jurisdictions restrict facial recognition without explicit consent

### Technical Constraints

**Security Requirements:**
- **MQTT Encryption:** Production deployment requires TLS/SSL, authentication (username/password or client certificates), private MQTT broker (not public test.mosquitto.org)
- **Data in Transit:** All camera feeds, MQTT messages, web traffic encrypted (HTTPS, WSS, MQTTS)
- **Access Control:** Dashboard authentication (username/password, role-based access: admin, manager, viewer)
- **Audit Logging:** All administrative actions logged (who, what, when)

**Privacy Protection:**
- **Anonymization:** Video faces blurred/anonymized in stored logs (optional, configurable)
- **ReID Data:** Feature vectors deleted after customer exits + receipt sent (no long-term biometric storage)
- **Pseudonymization:** Customer IDs (CUST_XXXX) used internally instead of real names
- **Data Minimization:** Only store what's necessary for transaction processing

**Performance Requirements:**
- **Real-Time Processing:** 30 FPS = ~33ms per frame for detection + tracking + ReID
- **Latency Budget:** Weight event → MQTT → Dashboard: <2 seconds end-to-end
- **Concurrent Customers:** System handles 5+ simultaneous customers without performance drop
- **Frame Processing:** No dropped frames, stable 30 FPS throughout business day

**Availability Requirements:**
- **Business Hours:** 99% uptime during store operating hours (8-hour day = ~5 minutes downtime acceptable)
- **Graceful Degradation:** If camera fails, system falls back to manual checkout mode; if MQTT fails, weight events queued
- **Data Backup:** Customer transaction logs backed up daily, retained for audit
- **Disaster Recovery:** System can recover from crash within 5 minutes, resume tracking

### Integration Requirements

**Store Systems:**
- **POS Integration:** Future payment integration requires connection to existing point-of-sale system
- **Inventory System:** Real-time inventory updates based on weight sensor events (items picked up = stock decrease)
- **Employee Access:** Store staff need dashboard access for monitoring, no system modification access

**Hardware Vendors:**
- **ESP32:** OTA firmware update mechanism for remote sensor updates
- **Cameras:** Support standard USB (UVC) and RTSP cameras, auto-detection on startup
- **Weight Sensors:** HX711 calibration data stored in config, accessible via dashboard

### Risk Mitigations

**Domain-Specific Risks:**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **False Positive Charge** | High - Customer disputes, legal liability | Medium | Manual review process, weight sensor correlation, dispute workflow, human approval on first visit |
| **Privacy Violation** | High - Regulatory fines, reputational damage | Medium | Clear consent mechanism, data retention policy, anonymization, legal review |
| **System Downtime** | Medium - Lost revenue, customer frustration | Medium | Graceful degradation to manual checkout, alert system, backup power |
| **Camera Tampering** | Medium - Blind spots in tracking | Low | Tamper detection, backup cameras, staff monitoring, alert on camera disconnect |
| **WiFi Interference** | Medium - MQTT packet loss, sensor offline | High | Local MQTT broker, message queuing, retry logic, dual-band WiFi support |
| **Biometric Data Breach** | Critical - Legal liability, customer trust loss | Low | Encrypt ReID features at rest, strict access controls, minimal biometric storage |
| **Regulatory Non-Compliance** | High - Fines, shutdown | Medium | Regular compliance audits, legal review of privacy policy, jurisdiction-specific configs |

**Additional Mitigation Strategies:**
- **Dispute Resolution Workflow:** Customer can flag incorrect charges, staff review video + weight logs, issue refund if confirmed
- **Data Encryption:** All sensitive data encrypted at rest (AES-256) and in transit (TLS 1.3)
- **Security Audits:** Annual penetration testing, quarterly security reviews
- **Privacy Policy:** Clear, accessible privacy policy explaining data collection, usage, retention
- **Legal Review:** Regular legal review of data handling practices, updates as regulations evolve

---

## IoT/Embedded Specific Requirements

### Project-Type Overview

GrabNGo-Advanced is a **multi-part IoT/Embedded system** combining:
- **Part 1:** Python computer vision backend (YOLO pose estimation + tracking)
- **Part 2:** ESP32 embedded weight sensor (MicroPython + HX711 + MQTT)

The system operates at the **edge** (in-store) with real-time CV processing and sensor fusion for autonomous retail checkout.

### Hardware Requirements

**Backend Server (CV Processing):**
- **CPU:** 4+ cores recommended for real-time YOLO inference @ 30 FPS
- **RAM:** 8GB minimum (16GB recommended for concurrent multi-customer tracking)
- **GPU:** Optional but recommended—CUDA-enabled GPU for 2-3x YOLO speedup
- **Storage:** 10GB+ for models, logs, and cached video frames
- **Camera:** USB or RTSP camera (640x480 minimum, 1920x1080 recommended)

**ESP32 Weight Sensor:**
- **Microcontroller:** ESP32-WROOM or ESP32-WROVER module
- **Load Cell:** HX711 amplifier with 5kg capacity (±50g accuracy)
- **Connectivity:** 2.4GHz WiFi only (ESP32 limitation)
- **Power:** USB-powered (5V 1A) or battery with sleep mode

**Critical CV Hardware Consideration:**
- Higher camera resolution (1080p+) improves ReID feature extraction accuracy
- GPU acceleration directly impacts ability to sustain 30 FPS with 5+ simultaneous tracks
- RAM capacity determines track buffer size (300 frames = 10 seconds @ 30 FPS)

### Connectivity Protocol

**MQTT Communication (ESP32 → Python Backend):**
- **Protocol:** MQTT v3.1.1 over TCP
- **Broker:** Public test broker (test.mosquitto.org) for development
- **Topic:** `my-shop/shelf-1/events`
- **Message Format:** `"CHANGE:{weight_g}"` (e.g., `"CHANGE:-480"`)
- **QoS:** QoS 0 (fire-and-forget) for development; QoS 1 for production
- **Latency Budget:** Weight event → MQTT → Dashboard: <2 seconds end-to-end

**Web Dashboard (Flask → Browser):**
- **Protocol:** HTTP/1.1 (upgrade to HTTPS for production)
- **Real-time Updates:** Server-Sent Events (SSE) for live tracking data
- **Mobile QR Scanner:** Responsive web interface (no native app required)

**CV Data Flow:**
```
Camera → OpenCV → YOLO Pose → BoT-SORT → Lightweight ReID → StatsManager → Dashboard
                                                              ↓
                                    MQTT Weight Events → Correlation Engine
```

### Power Profile

**ESP32 Power Consumption:**
- **Active Mode:** ~160mA during weight measurement and MQTT publish
- **Deep Sleep:** ~10µA (not used—continuous monitoring required)
- **Duty Cycle:** 100% (always-on during business hours)
- **Power Source:** USB power supply recommended for reliability

**Backend Server Power:**
- **Idle (no customers):** ~40W (CPU + camera)
- **Active (tracking):** ~80-150W depending on CPU/GPU utilization
- **Operational Hours:** 8-12 hours/day (typical retail schedule)

**Power Failure Recovery:**
- ESP32 auto-reconnects to WiFi/MQTT on power restoration
- Python backend auto-recovers camera connection
- Track buffer (300 frames) provides 10-second occlusion tolerance

### Security Model

**Current State (Development):**
- ⚠️ **Public MQTT Broker:** test.mosquitto.org (NO encryption, NO authentication)
- ⚠️ **HTTP:** Dashboard on port 8081 (NO HTTPS)
- ⚠️ **No Authentication:** Dashboard is publicly accessible
- ⚠️ **No Data Encryption:** Video logs and tracking data stored in plaintext

**Production Requirements (MUST implement before deployment):**

| Security Layer | Current | Production Requirement |
|----------------|---------|------------------------|
| **MQTT** | Public broker, no auth | Private broker with TLS/SSL, username/password or client certificates |
| **Web Dashboard** | HTTP, no auth | HTTPS with authentication (username/password, role-based access) |
| **Data at Rest** | Plaintext | AES-256 encryption for logs and customer data |
| **Data in Transit** | Plaintext | TLS 1.3 for all web traffic, MQTTS for sensor data |
| **Privacy** | No anonymization | Face blurring in stored video (optional), pseudonymized customer IDs |

**Compliance Requirements:**
- **GDPR (EU):** Right to access, right to erasure, data minimization
- **CCPA (California):** Do not sell my data, right to opt-out
- **Biometric Data:** ReID feature vectors (512-dim) deleted after customer exit

### OTA Update Mechanism

**ESP32 Firmware Updates:**
- **Method:** MicroPython `.py` file replacement via serial (Thonny IDE or ampy)
- **Future Enhancement:** Over-the-air (OTA) updates using MicroPython's network boot capability
- **Calibration Data:** Stored in `main.py` (tare value, scale ratio)
- **Update Frequency:** Rare (firmware is stable, updates only for bug fixes)

**Python Backend Updates:**
- **Method:** git pull + pip install -r requirements.txt
- **Models:** YOLO models auto-download on first run via Ultralytics
- **Configuration:** YAML/JSON configs hot-reload (restart required for changes)
- **Zero-Downtime Deployment:** Use systemd service with graceful restart

### CV Accuracy Infrastructure Requirements

**To Achieve ≥90% ReID Accuracy:**

1. **Computational Budget:**
   - **Frame Budget:** ~33ms per frame @ 30 FPS target
   - **YOLO Inference:** ~20ms (GPU) to ~50ms (CPU) per frame
   - **ReID Feature Extraction:** ~5ms per track update
   - **Total per frame:** YOLO + tracking + ReID must fit within 33ms window

2. **Track Buffer (Occlusion Handling):**
   - **Buffer Size:** 300 frames = 10 seconds @ 30 FPS
   - **Purpose:** Store ReID features during occlusion for post-occlusion matching
   - **Memory Impact:** ~5MB per track (512-dim float32 × 300 frames)
   - **5 Simultaneous Tracks:** ~25MB RAM for track buffers

3. **ReID Similarity Threshold:**
   - **Cosine Similarity:** 0.1 threshold for track re-identification
   - **Feature Dimensions:** 512 (LAB color 192 + HOG 192 + texture 96 + edge 48)
   - **Match Strategy:** Nearest neighbor in feature space with threshold gating

4. **Zone-Based Correlation (Cart Accuracy):**
   - **Shelf Zone:** Weight events only correlated with tracks in spatial zone
   - **QR Zone:** Single-occupancy requirement for accurate confirmation
   - **Zone Configuration:** Percentage-based (0.0-1.0) for camera resolution independence

5. **Performance Metrics & Monitoring:**
   - **FPS Monitoring:** Dashboard displays real-time processing rate
   - **Active Track Counter:** Track ≥5 simultaneous customers
   - **ReID Event Logging:** Track re-identification after occlusion
   - **MQTT Latency:** Timestamp from weight event to dashboard display

### Implementation Considerations

**Hardware Setup:**
- ESP32 and HX711 require careful calibration (tare + scale ratio)
- WiFi signal strength impacts MQTT reliability (use local broker in production)
- Camera positioning critical for zone definition accuracy

**Software Architecture:**
- Multi-process architecture: Tracker process + Dashboard process (shared stats via JSON)
- Silent failure pattern: Non-critical errors (MQTT disconnect) shouldn't crash main process
- Frame skipping optional: Can skip frames to maintain FPS at cost of reduced tracking precision

**Environmental Factors:**
- Lighting changes affect YOLO detection confidence
- Occlusion from shelves/fixtures requires robust track buffer
- WiFi interference causes MQTT packet loss (message queuing recommended)

**Testing & Validation:**
- Unit tests for ReID feature extraction and similarity matching
- Integration tests for MQTT weight event correlation
- Manual testing: Walk through zones with known items, verify cart accuracy
- Field testing: Real retail environment with actual customers

---

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Problem-Solving MVP focused on **CV accuracy validation**
- **Core Problem:** Can we accurately track customers and correlate item pickups to achieve ≥95% cart accuracy?
- **Validated Learning:** What is the real-world dispute rate with autonomous checkout?
- **Success Definition:** ≤10% error rate (MVP) → ≤2% error rate (Production)

**Resource Requirements:**
- **Minimum Team:** 1 CV engineer (Python/PyTorch/OpenCV) + 1 embedded/IT generalist
- **Hardware:** 1 server (CPU or GPU), 1 camera, 1 ESP32+HX711 sensor per shelf
- **Timeline:** 2-4 weeks for CV accuracy optimization + field testing

### MVP Feature Set (Phase 1)

**Core User Journeys Supported:**
- ✅ Shopper autonomous journey (Sarah Chen - happy path)
- ✅ Store manager monitoring (David Kim - dashboard)
- ⚠️ Multiple customers in QR zone (Mike Rodriguez - edge case with wait)

**Must-Have Capabilities:**

**Computer Vision (Accuracy Focus):**
- YOLO11n-pose person detection (confidence ≥0.6)
- BoT-SORT multi-object tracking with Kalman filtering
- **Lightweight ReID with 512-dim features (LAB + HOG + texture + edge)**
- **Track buffer: 300 frames (10 seconds @ 30 FPS) for occlusion handling**
- **ReID similarity threshold: 0.1 cosine similarity**
- **Target: ≥90% re-identification accuracy after 10-second occlusion**
- Zone-based detection (QR zone, shelf zone)

**Sensor Integration:**
- ESP32 HX711 weight sensor (±50g accuracy)
- MQTT weight event publication (`my-shop/shelf-1/events`)
- Weight-to-track spatial correlation in shelf zone
- Latency <2 seconds from sensor to dashboard

**User Interaction:**
- QR code confirmation workflow
- Mobile web interface (responsive)
- Manual confirmation for new tracks
- Dashboard with real-time monitoring

**System Reliability:**
- 30 FPS sustained processing
- 5+ simultaneous customer tracking
- Auto-recovery from camera/MQTT disconnect
- Silent failure pattern (non-critical errors don't crash system)

**What's NOT in MVP:**
- ❌ Holding detection (MediaPipe) - defer to Phase 2
- ❌ Multi-camera tracking - single camera sufficient
- ❌ Native mobile app - web interface works
- ❌ Automatic payment - manual receipt-based
- ❌ Face ReID for customer accounts - anonymous tracking
- ❌ Advanced ReID (deep learning) - lightweight features for now

### Post-MVP Features

**Phase 2: Growth (Post-MVP Optimization)**

**CV Accuracy Enhancements:**
- Holding detection (MediaPipe) for item-level pickup confirmation
- Advanced ReID (yolo11n-cls.pt) for improved feature extraction
- Multi-camera handoff for larger store coverage
- Adaptive zone configuration based on camera position

**Feature Additions:**
- Multi-shelf weight sensors (expand beyond single shelf)
- Automatic receipt generation (PDF/email)
- Basic payment integration (Stripe/square)
- Customer account system with return customer recognition
- Mobile app native (replace web-based QR)

**Growth Targets:**
- Multi-camera tracking with ≤10% ID handoff errors
- Multi-shelf weight correlation accuracy ≥90%
- Error rate reduced from ≤10% (MVP) to ≤5% (Growth)
- Mobile app adoption ≥60% of customers

**Phase 3: Expansion (Production Scale)**

**Advanced Capabilities:**
- Chain-wide deployment with centralized management
- AI shopping assistant (product recommendations)
- Predictive restocking (inventory alerts from shopping patterns)
- Loss prevention (anomalous behavior detection)
- Customer analytics (heat maps, dwell time, product affinity)
- Self-learning ReID (model improves with each customer)

**Production Targets:**
- 100+ stores deployed
- Error rate ≤2% (production validated)
- 99.9% uptime across chain
- Zero checkout staff required
- Real-time inventory accuracy ≥98%

### Risk Mitigation Strategy

**Technical Risks:**

| Risk | Mitigation |
|------|------------|
| **ReID accuracy <90% in production** | Start with controlled environment (good lighting, low occlusion); tune similarity threshold; collect real-world data for model improvement |
| **Cannot sustain 30 FPS with 5+ tracks** | GPU acceleration; frame skipping as fallback; reduce camera resolution if needed |
| **Track buffer insufficient for occlusion** | Increase buffer to 450 frames (15 seconds); add proximity-based track merging |
| **MQTT packet loss causes missed items** | Local MQTT broker; QoS 1 for delivery confirmation; message queuing on ESP32 |

**Market Risks:**

| Risk | Mitigation |
|------|------------|
| **High dispute rate from false charges** | Manual confirmation on first visit; clear dispute workflow; staff review of video+weight logs; refund policy |
| **Customers don't trust autonomous checkout** | Transparent consent process; show cart in real-time on phone; easy opt-out to manual checkout |
| **Store adoption too slow** | Focus on single-store proof-of-concept; document ROI (labor savings vs system costs); case study marketing |

**Resource Risks:**

| Risk | Mitigation |
|------|------------|
| **Limited CV engineering resources** | Start with existing lightweight ReID; defer advanced ReID to Phase 2; use GPU to compensate for algorithm complexity |
| **Hardware budget constraints** | Single-camera deployment; CPU-only mode (slower FPS acceptable for testing); use existing ESP32 hardware |
| **Calibration time under-estimated** | Document calibration process; create calibration scripts; allow 1-2 days for sensor tuning per shelf |

**Contingency Approach:**
- **If CV accuracy <85%:** Increase ReID feature dimensions, add deep learning classifier, reduce simultaneous customer target to 3
- **If error rate >15%:** Implement manual review workflow, add second camera for verification, pause autonomous charging
- **If cannot achieve 30 FPS:** Accept 15-20 FPS for MVP (reduced accuracy but functional), add GPU in Phase 2

---

## Functional Requirements

### Customer Tracking & Identification

**FR1:** System can detect persons entering the shopping area using YOLO pose estimation with configurable confidence threshold

**FR2:** System can assign unique tracking IDs to detected persons and maintain tracking across video frames

**FR3:** System can track up to 5 simultaneous customers within the camera field of view without ID confusion

**FR4:** System can extract 512-dimensional appearance features (LAB color + HOG + texture + edge) from each tracked person

**FR5:** System can re-identify a customer after up to 10 seconds of occlusion using cosine similarity matching with 0.1 threshold

**FR6:** System can maintain track buffer of 300 frames (10 seconds @ 30 FPS) for each active customer to enable occlusion recovery

**FR7:** System can detect when a tracked person enters or exits predefined spatial zones (QR zone, shelf zone)

**FR8:** System can determine when only one customer is present in the QR confirmation zone

**FR9:** System can process video at sustained 30 FPS with no dropped frames during business hours

**FR10:** System can auto-recover tracking state after camera disconnection within 10 seconds

### Item Detection & Cart Management

**FR11:** System can receive MQTT weight change events from ESP32 sensors with timestamp and weight delta

**FR12:** System can correlate weight change events with customer tracks based on spatial presence in shelf zone

**FR13:** System can maintain shopping cart for each tracked customer listing detected items with timestamps

**FR14:** System can apply ±50g weight threshold to filter false positive weight events

**FR15:** System can update customer cart in real-time when correlated weight events occur

**FR16:** System can detect and alert when ESP32 weight sensor is offline for more than 60 seconds

**FR17:** System can calibrate HX711 weight sensor with tare value and scale ratio for accurate gram measurements

**FR18:** System can log all MQTT weight events with timestamps for audit and dispute resolution

### User Interaction & Confirmation

**FR19:** Customer can scan QR code using mobile device to initiate identity confirmation

**FR20:** Customer can view their current shopping cart in real-time via mobile web interface

**FR21:** Customer can confirm their identity and finalize shopping cart via mobile web interface

**FR22:** System can generate unique customer ID (e.g., CUST_XXXX) upon first QR confirmation

**FR23:** System can require manual staff confirmation for new tracks before allowing autonomous checkout

**FR24:** System can display "Please wait" message when multiple customers are present in QR zone

**FR25:** System can lock and finalize customer cart upon QR confirmation and prevent further modifications

**FR26:** Customer can exit store after confirmation without manual checkout

**FR27:** Customer can receive digital receipt (email) with itemized list, timestamps, and total amount

### System Monitoring & Management

**FR28:** Store manager can view real-time dashboard showing active customers, their tracking states, and shopping carts

**FR29:** Store manager can view system health indicators including camera status, MQTT connection status, and sensor status

**FR30:** Store manager can view daily metrics including total customers processed, accuracy rate, and system uptime

**FR31:** Store manager can receive alerts for offline sensors, camera failures, or tracking anomalies

**FR32:** Store manager can access historical tracking events log with timestamps and zone transitions

**FR33:** Store manager can access historical MQTT weight events log for audit purposes

**FR34:** Store manager can export customer transaction logs for end-of-day reconciliation

**FR35:** Store manager can review disputed transactions with video logs and weight event correlation

**FR36:** System administrator can access troubleshooting documentation and configuration utilities

**FR37:** System administrator can calibrate weight sensors via calibration scripts

**FR38:** System administrator can test MQTT connectivity and inspect published messages

**FR39:** System administrator can update zone configuration as percentage-based coordinates

**FR40:** System administrator can configure tracker parameters (ReID threshold, track buffer size, detection confidence)

### Data Management & Privacy

**FR41:** Customer can provide opt-in consent for tracking upon store entry via signage and mobile interface

**FR42:** Customer can request deletion of their tracking data and biometric features under GDPR/CCPA

**FR43:** System can pseudonymize customer data using internal customer IDs (CUST_XXXX) instead of real names

**FR44:** System can delete ReID feature vectors after customer exits and receipt is generated

**FR45:** System can retain video logs for configurable period (7-30 days) and automatically expire after retention period

**FR46:** System can retain transaction logs for 2-7 years for tax and audit compliance

**FR47:** System can blur faces in stored video logs for privacy (optional, configurable)

**FR48:** System can encrypt all data at rest using AES-256 encryption

**FR49:** System can encrypt all data in transit using TLS 1.3 for web traffic and MQTTS for sensor data

**FR50:** System can log all administrative actions with timestamp, user identity, and action details for audit

**FR51:** System can authenticate dashboard users with username/password and role-based access (admin, manager, viewer)

**FR52:** System can authenticate MQTT clients using username/password or client certificates (production requirement)

**FR53:** System can gracefully degrade to manual checkout mode if camera or sensor failures occur

**FR54:** System can recover from crash within 5 minutes and resume tracking with preserved state

**FR55:** System can backup customer transaction logs daily and retain for audit purposes

---

## Non-Functional Requirements

### Performance

**Frame Processing:**
- System processes video at minimum 30 FPS with no dropped frames during business hours
- YOLO pose inference completes within 33ms per frame (GPU) or 50ms per frame (CPU)
- ReID feature extraction completes within 5ms per track update

**Response Time:**
- Weight change events appear on dashboard within 2 seconds of sensor detection
- Mobile web interface loads within 3 seconds on 4G mobile connection
- QR confirmation completes within 10 seconds from scan to confirmation

**Concurrent Users:**
- System tracks minimum 5 simultaneous customers with no performance degradation
- Dashboard supports 10 concurrent manager viewers without lag

**Resource Utilization:**
- CPU usage remains below 80% during peak tracking periods
- RAM usage remains below 16GB with 5 active tracks (300 frame buffers)
- GPU memory usage remains below 8GB when processing at 30 FPS

### Security

**Encryption:**
- All data at rest encrypted using AES-256 encryption
- All data in transit encrypted using TLS 1.3 for web traffic
- All MQTT messages encrypted using MQTTS in production deployment

**Authentication:**
- Dashboard requires username/password authentication with role-based access control (admin, manager, viewer)
- MQTT clients authenticate using username/password or client certificates in production
- API endpoints authenticate using bearer tokens or API keys

**Authorization:**
- Role-based access control enforces minimum privilege principle
- Admins can modify configuration, managers can view and export, viewers can read-only
- Customer data access restricted to authorized personnel only

**Privacy:**
- Biometric ReID feature vectors deleted after customer exit and receipt generation
- Customer data pseudonymized using internal customer IDs (CUST_XXXX)
- Video logs optionally blur faces for privacy (configurable)
- Customer consent obtained before tracking begins

**Audit:**
- All administrative actions logged with timestamp, user identity, and action details
- All customer transaction logs retained for 2-7 years for audit compliance
- Failed authentication attempts logged and monitored for suspicious activity

**Compliance:**
- System supports GDPR right to access and right to erasure requests
- System supports CCPA opt-out and data deletion requests
- System compliant with PCI DSS when payment integration is added

### Scalability

**Customer Throughput:**
- System processes minimum 1 customer per minute sustained throughput
- System supports 100 customers per day in MVP deployment
- System supports 500 customers per day in production deployment

**Store Expansion:**
- Architecture supports multi-store deployment with centralized management
- Each store operates independently with local processing (edge computing)
- Central management console can monitor 100+ stores simultaneously

**Hardware Scaling:**
- System supports adding cameras to existing store without architectural changes
- System supports adding weight sensors to additional shelves via MQTT topic expansion
- System supports GPU upgrade for improved performance without software changes

**Data Growth:**
- System handles 1TB of video logs per store per month with retention policy
- System handles 10,000+ customer records per store per year
- Database queries remain responsive (<1 second) with 100,000+ transaction records

### Reliability

**Uptime:**
- System maintains 99% uptime during store business hours (8-hour day = ~5 minutes downtime acceptable)
- System maintains 99.9% uptime in production deployment across all stores

**Fault Tolerance:**
- System auto-recovers from camera disconnection within 10 seconds
- System auto-recovers from MQTT disconnection within 30 seconds
- System continues operation with degraded functionality if single sensor fails

**Data Integrity:**
- No loss of customer transaction data under normal operation
- No loss of tracking events during temporary network interruptions (local buffering)
- No corruption of video logs or audit trails

**Disaster Recovery:**
- System recovers from crash within 5 minutes and resumes tracking with preserved state
- Customer transaction logs backed up daily to offsite storage
- System configuration backed up before each change

**Graceful Degradation:**
- If camera fails, system alerts staff and falls back to manual checkout mode
- If MQTT fails, system queues weight events locally for later processing
- If tracking accuracy degrades below threshold, system alerts staff for manual review

### Integration

**MQTT Integration:**
- System subscribes to weight sensor topics using MQTT v3.1.1 protocol
- System supports configurable broker address, port, topic, and QoS level
- System handles MQTT connection failures with auto-reconnect and message queuing

**Web Integration:**
- Dashboard accessible via standard HTTP/HTTPS browser
- Mobile web interface responsive across device sizes (phone, tablet, desktop)
- System supports Server-Sent Events (SSE) for real-time dashboard updates

**Hardware Integration:**
- System supports standard USB (UVC) cameras with auto-detection on startup
- System supports RTSP camera streams with configurable URL
- System supports ESP32 HX711 sensors with configurable calibration data

**Future Integration:**
- Architecture supports future POS system integration via REST API
- Architecture supports future payment processor integration via webhooks
- Architecture supports future inventory management system integration via batch export

**Data Exchange:**
- System exports customer transaction logs in CSV format for external systems
- System exposes REST API for querying customer status and cart contents
- System publishes tracking events to configurable webhooks for external monitoring

