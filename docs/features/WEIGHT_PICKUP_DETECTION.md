# Weight-Based Pickup Detection System

## Overview

Hệ thống tự động phát hiện và ping khách hàng đã CONFIRMED khi họ lấy hàng từ kệ, sử dụng kết hợp:
- **Weight Sensor (ESP32 + HX711)**: Phát hiện thay đổi trọng lượng
- **AI Vision**: Xác định ai đang ở gần kệ và có tay duỗi về phía kệ
- **Shopping Cart**: Tự động cập nhật giỏ hàng của khách hàng

---

## Architecture

```
ESP32 (Weight Sensor)
    ↓ MQTT: "CHANGE:-480"
    ↓
Python CV System
    ↓ Subscribe MQTT
    ↓
Find CONFIRMED customers in shelf zone
    ↓
Rank by: Proximity (60%) + Hand Position (40%)
    ↓
Ping best candidate → Update shopping cart
```

---

## Configuration

### MQTT Settings
- **Broker**: `test.mosquitto.org` (port 1883)
- **Topic**: `my-shop/shelf-1/events`
- **Payload Format**: `"CHANGE:-480"` (weight change in grams)

### Shelf Zone
- **Default Position**: Left-middle area (0-50% width, 30-90% height)
- **Overlap Threshold**: 30% of person must be in zone
- **Calibration**: Adjust `shelf_zone_percent` in `RetailCustomerTracker.__init__()`

### Scoring Weights
- **Proximity**: 60% (distance to shelf center)
- **Hand Position**: 40% (hand reaching toward shelf)
- **Combined Score**: `0.6 * proximity + 0.4 * hand`

---

## Features Implemented

### ✅ Phase 1: Setup MQTT & Shelf Zone
- MQTT client connection and subscription
- Shelf zone definition and visualization
- Weight event reception and parsing

### ✅ Phase 2: Matching Logic
- Find confirmed customers in shelf zone
- Rank by proximity and hand position
- Select best candidate

### ✅ Phase 3: Ping Mechanism
- Update shopping cart with picked up items
- Rate limiting (max 1 ping per 2 seconds)
- Event logging

---

## Shopping Cart Structure

```python
customer['shopping_cart'] = [
    {
        'weight_g': 480,
        'timestamp': '2024-01-15T10:30:45.123Z',
        'shelf_id': 'shelf-1',
        'confidence': 0.85
    },
    # ... more items
]

customer['pickup_count'] = 2
customer['last_pickup_time'] = datetime(...)
customer['items_detected'] = {'item_1', 'item_2'}
```

---

## Event Logging

### Item Picked Up Event
```json
{
    "type": "item_picked_up",
    "customer_id": "CUST_0001",
    "track_id": 5,
    "weight_change_g": -480,
    "item_weight_g": 480,
    "timestamp": "2024-01-15T10:30:45.123Z",
    "shelf_id": "shelf-1",
    "confidence": 0.85,
    "proximity_score": 0.9,
    "hand_score": 0.8
}
```

### Unmatched Event
```json
{
    "type": "unmatched_weight_event",
    "weight_change_g": -480,
    "timestamp": "2024-01-15T10:30:45.123Z",
    "shelf_id": "shelf-1",
    "reason": "no_customer_in_zone"
}
```

---

## Testing

### Prerequisites
1. ESP32 running and sending MQTT events
2. Camera system running and tracking customers
3. At least one CONFIRMED customer (via QR scan)

### Test Flow
1. **Start System**: Run `python src/main_tracker.py`
2. **Confirm Customer**: Use QR scan to confirm a customer
3. **Move Customer to Shelf Zone**: Customer should be in shelf zone (cyan rectangle)
4. **Trigger Weight Event**: ESP32 sends weight change event
5. **Verify Ping**: Check console logs for ping confirmation
6. **Check Shopping Cart**: Verify `customer['shopping_cart']` has new item

### Expected Console Output
```
⚖️  Weight Event: -480g at 10:30:45.123
   🔍 Looking for customer who picked up item (480g)...
   ✅ Found 1 candidate(s), best: CUST_0001 (score: 85%)
   ✅ Pinged CUST_0001 (Track 5)
      Item: 480g
      Confidence: 85%
      Shopping cart: 1 items
```

---

## Calibration

### Shelf Zone Position
Adjust in `src/main_tracker.py`:
```python
self.shelf_zone_percent = {
    'x1_percent': 0.0,    # Left edge
    'y1_percent': 0.3,    # 30% from top
    'x2_percent': 0.5,    # 50% from left
    'y2_percent': 0.9     # 90% from top
}
```

**How to calibrate:**
1. Run system with camera
2. Identify shelf position in frame
3. Adjust percentages to match shelf location
4. Test with person in/out of zone

### Scoring Weights
Adjust in `_rank_customers_by_pickup_likelihood()`:
```python
combined_score = 0.6 * proximity_score + 0.4 * hand_score
```

**Tuning:**
- If proximity too important → Reduce to 0.5, increase hand to 0.5
- If hand detection unreliable → Reduce to 0.3, increase proximity to 0.7

---

## Edge Cases Handled

### Multiple Customers in Zone
- **Solution**: Rank all candidates, ping highest score
- **Logic**: Combined score determines priority

### No Customer in Zone
- **Solution**: Log as unmatched event
- **Reason**: Customer moved away or false positive

### Rate Limiting
- **Limit**: Max 1 ping per customer per 2 seconds
- **Purpose**: Avoid duplicate pings for rapid pickups

### Timing Delay
- **Window**: Events matched within current frame
- **Future**: Could add time window matching (last 3 seconds)

---

## Dependencies

- `paho-mqtt>=1.6.0` (added to requirements.txt)
- All existing dependencies (opencv, numpy, ultralytics, etc.)

---

## Next Steps (Phase 4: Testing & Refinement)

1. **Test MQTT Connection**: Verify events received correctly
2. **Calibrate Shelf Zone**: Adjust position based on camera view
3. **Test Matching Logic**: Verify correct customer selected
4. **Test Shopping Cart**: Verify items added correctly
5. **Tune Scoring Weights**: Optimize based on accuracy
6. **Handle Edge Cases**: Test multiple customers, no match scenarios

---

## Troubleshooting

### MQTT Not Connecting
- Check internet connection
- Verify `test.mosquitto.org` is accessible
- Check firewall settings
- Try alternative broker

### No Customers Found
- Verify customer is CONFIRMED (not PENDING)
- Check customer is in shelf zone (cyan rectangle visible)
- Verify `last_box` is not None

### Wrong Customer Pinged
- Check proximity and hand scores in logs
- Adjust scoring weights if needed
- Verify shelf zone position is correct

### Shopping Cart Not Updating
- Check rate limiting (2 second limit)
- Verify customer still exists in `self.customers`
- Check console logs for errors

---

## Notes

- Only CONFIRMED customers are pinged (PENDING tracks ignored)
- Weight increase events (item return) are currently skipped
- Hand position detection requires YOLO pose keypoints
- System works best with clear view of shelf and customers



