# Manual Confirmation System with Validation

## Overview
The system now implements **Option A: Individual Confirmation** with **Intelligent Validation** for new customer IDs. Every new detection requires:
1. **Automatic data collection** (appearance features, confidence scores)
2. **Validation** (minimum quality checks)
3. **Manual confirmation** (only when validation passes)

## How It Works

### Track States
- **PENDING (Collecting)**: New detection, collecting data (Orange box, 0-79% validation)
- **PENDING (Ready)**: Validated, ready for confirmation (Green box, ≥80% validation)
- **CONFIRMED**: Manually confirmed customer (Green box with customer ID)

### Validation Requirements

Before a track can be confirmed, it must pass these checks:

| Check | Requirement | Purpose |
|-------|-------------|---------|
| **Sample Count** | ≥5 frames | Enough observations for reliable identification |
| **Feature Quality** | ≥30% valid | Appearance features extracted successfully |
| **Confidence** | Average ≥0.5 | Detection is reliable |
| **Consistency** | Low variance | Person's appearance is consistent |

**Overall Score**: Average of all checks must be ≥80%

### Visual Indicators

**Stage 1: Collecting (Orange, < 80%)**
```
┌──────────────────────────────┐
│ PENDING_0001                 │ ← Orange box (collecting)
│ Samples: 2/5                 │
│ [▓▓▓▓░░░░░░] 45%            │ ← Progress bar
│ Collecting...                │
└──────────────────────────────┘
```

**Stage 2: Ready (Green, ≥ 80%)**
```
┌──────────────────────────────┐
│ PENDING_0001                 │ ← Green box (validated)
│ Samples: 5/5                 │
│ [▓▓▓▓▓▓▓▓▓▓] 95%            │ ← Full progress bar
│ READY (Press 'c')            │
└──────────────────────────────┘
```

**Stage 3: Confirmed**
```
┌──────────────────────────────┐
│ CUST_0001                    │ ← Green box (confirmed)
│ G:0 I:0                      │ ← Gesture & Items count
└──────────────────────────────┘
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `c` | Confirm selected pending track |
| `1-9` | Select pending track by number |
| `q` | Quit |
| `s` | Save logs |
| `i` | Show statistics |

### Workflow

1. **New Detection (Frame 0-1)**
   - System detects a person
   - Creates PENDING track with orange box
   - Shows `PENDING_XXXX` ID
   - Prints: `⏳ Pending | PENDING_0001 (Track 5) - Collecting info...`
   - **Status**: Collecting (0%)

2. **Data Collection (Frame 1-5)**
   - System automatically collects:
     - Appearance features (LAB color, HOG, texture)
     - Detection confidence scores
     - Bounding boxes
   - Progress bar fills up: 20% → 40% → 60% → 80%
   - Box color changes: Orange → Green (when ready)

3. **Validation Complete (Frame 5+)**
   - All checks passed (≥80% validation score)
   - Box turns green
   - Shows: `READY (Press 'c')`
   - Prints: `✓ Ready | PENDING_0001 - Can confirm now (Press 'c')`

4. **Manual Confirmation**
   - User presses `c` to confirm selected pending track
   - Or presses `1-9` to select specific track, then `c`
   - **Validation check runs**: If < 80%, confirmation is blocked
   - If passed: System assigns permanent `CUST_XXXX` ID
   - Prints: 
     ```
     ✅ Confirmed | CUST_0001 (Track 5)
        Validation: 95% | Samples: 5 | Conf: 0.78
     ```

5. **Insufficient Information**
   - User tries to confirm before validation passes
   - System blocks confirmation
   - Prints:
     ```
     ❌ Cannot confirm PENDING_0001 - Insufficient information:
        • Need 5 samples, got 3
        Validation score: 60% (need ≥80%)
     ```

6. **Re-Identification (Automatic)**
   - If person was seen before (within 5 seconds)
   - System uses ReID to match automatically
   - **No validation or confirmation needed**
   - Prints: `🔄 ReID | CUST_0001 (New Track 7)`

7. **Auto-Timeout**
   - Pending tracks older than 10 seconds are removed
   - Prints: `⏱️  Timeout | PENDING_0001 removed (no confirmation)`

## On-Screen Display

### Top Bar
```
Active: 2 | Pending: 1 | Occluded: 0 | Total: 2
FPS: 28.5
```

### Pending Panel (if pending tracks exist)
```
PENDING TRACKS:
> 1. PENDING_0001 (3.2s) ✓95%  ← Selected (yellow), Ready (green check)
  2. PENDING_0002 (1.5s) ⏳45%  ← Not selected, Collecting (hourglass)
```

**Color Coding:**
- **Yellow**: Currently selected track
- **Green**: Validated and ready to confirm
- **Orange**: Still collecting information

**Icons:**
- **✓**: Validation passed (≥80%)
- **⏳**: Still collecting data (<80%)

## Features

### ✅ Implemented
1. **Manual Confirmation System**
   - Pending state for new detections
   - Individual confirmation workflow
   
2. **Visual Indicators**
   - Orange boxes for PENDING tracks
   - Green boxes for CONFIRMED customers
   - Confirmation prompts on bounding boxes
   
3. **Keyboard Controls**
   - `c` for confirmation
   - `1-9` for selection
   
4. **Auto-Timeout**
   - 10-second timeout for unconfirmed tracks
   
5. **ReID Integration**
   - Automatic re-identification for returning customers
   - No confirmation needed for re-entries

## Validation Details

### What Gets Validated?

1. **Sample Count (Weight: 25%)**
   - Checks: Number of frames with features extracted
   - Pass: ≥5 samples collected
   - Why: Need multiple observations for reliable identification

2. **Feature Quality (Weight: 25%)**
   - Checks: Percentage of valid (non-null) features
   - Pass: ≥30% of samples have valid features
   - Why: Poor lighting/angles may fail feature extraction

3. **Detection Confidence (Weight: 25%)**
   - Checks: Average YOLO detection confidence
   - Pass: ≥0.5 average confidence
   - Why: Low confidence = uncertain detection

4. **Feature Consistency (Weight: 25%)**
   - Checks: Variance of appearance features across samples
   - Pass: Low variance (consistent appearance)
   - Why: High variance = unstable tracking or multiple people

**Overall Score** = Average of all 4 checks

### Why Validation Matters

❌ **Without Validation:**
- False detections get confirmed
- Poor quality tracks assigned IDs
- Inconsistent ReID matching
- Wasted effort tracking noise

✅ **With Validation:**
- Only high-quality tracks confirmed
- Reliable appearance features for ReID
- Reduced false positives
- Better tracking accuracy

### Configuration

```python
# In RetailCustomerTracker.__init__()

# Validation requirements
self.min_samples_required = 5      # Need 5 feature samples
self.min_confidence_avg = 0.5       # Average confidence >= 0.5
self.min_feature_quality = 0.3      # At least 30% valid features

# Timeouts
self.pending_timeout = 10.0         # Auto-remove pending after 10 seconds
self.max_lost_time = 5.0            # ReID window (5 seconds)

# ReID settings
self.feature_gallery_size = 10      # Keep last 10 features per track
```

### Adjusting Validation Strictness

**More Lenient** (accept lower quality):
```python
self.min_samples_required = 3       # Fewer samples (faster, less reliable)
self.min_confidence_avg = 0.4       # Lower confidence threshold
self.min_feature_quality = 0.2      # Accept more failed extractions
```

**More Strict** (higher quality):
```python
self.min_samples_required = 10      # More samples (slower, more reliable)
self.min_confidence_avg = 0.6       # Higher confidence threshold
self.min_feature_quality = 0.5      # Require more valid features
```

## Usage Example

```python
from main_tracker import RetailCustomerTracker

tracker = RetailCustomerTracker(
    detection_model='yolo11n.pt',
    tracker_config='botsort_reid.yaml'
)

# In main loop:
# 1. New person detected → PENDING_0001 appears (orange)
# 2. User presses 'c' → becomes CUST_0001 (green)
# 3. Person leaves, then returns → automatically matched to CUST_0001
```

## Benefits

✅ **Quality Assurance**: Only validated tracks can be confirmed  
✅ **Full Control**: No automatic new IDs without confirmation  
✅ **Individual Verification**: Confirm each person separately  
✅ **Visual Feedback**: Real-time validation progress with progress bars  
✅ **Intelligent Blocking**: Cannot confirm insufficient data  
✅ **Prevents False IDs**: Automatic validation filters out bad detections  
✅ **Auto-Cleanup**: Old pending tracks timeout automatically  
✅ **Reliable ReID**: High-quality features ensure better re-identification  

## Trade-offs

⚠️ **Manual Interaction Required**: User must press 'c' for each new person  
⚠️ **Attention Needed**: Can miss confirmations if not watching  
⚠️ **Slower Workflow**: 5+ frames needed before confirmation possible  
⚠️ **Not for High Traffic**: Not suitable for 10+ simultaneous people  

## Best For

- **Low to Medium Traffic**: 1-5 people at a time
- **High Accuracy Requirements**: Critical tracking scenarios  
- **Quality Over Speed**: When false positives are costly
- **Training/Testing**: Validating detection quality
- **Controlled Environments**: Where manual verification is acceptable
- **Retail Analytics**: Accurate customer counting and behavior analysis

