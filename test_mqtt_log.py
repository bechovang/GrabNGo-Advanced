"""
Test MQTT log display on dashboard
"""

import sys
import os
import time
import threading
import requests

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_mqtt_log():
    """Test MQTT log display with simulated events"""
    print("="*60)
    print("MQTT LOG DASHBOARD TEST")
    print("="*60)
    
    # Initialize tracker
    print("1. Initializing tracker...")
    from main_tracker import RetailCustomerTracker
    tracker = RetailCustomerTracker(
        detection_model='yolo11n-pose.pt',
        tracker_config='config/botsort_reid.yaml'
    )
    
    # Initialize MQTT
    print("2. Initializing MQTT...")
    tracker._init_mqtt()
    
    # Add mock customer in shelf zone
    print("3. Adding mock customer...")
    mock_customer_id = f"TEST_{int(time.time())}"
    mock_track_id = 999
    
    customer_data = {
        'track_id': mock_track_id,
        'customer_id': mock_customer_id,
        'first_seen': time.time(),
        'last_seen': time.time(),
        'last_box': [200, 200, 400, 600],  # Box in shelf zone
        'keypoints': None,
        'confirmed': True,
        'shopping_cart': [],
        'pickup_count': 0,
        'last_pickup_time': None
    }
    
    tracker.customers[mock_track_id] = customer_data
    tracker._update_shelf_zone((720, 1280))  # height, width
    
    # Start web server
    print("4. Starting web server...")
    try:
        from web_server import run_server
        server_thread = threading.Thread(
            target=run_server,
            args=(tracker, '0.0.0.0', 8080, False),
            daemon=True
        )
        server_thread.start()
        print("   ✓ Web server started: http://localhost:8080")
        print("   ✓ Dashboard: http://localhost:8080/dashboard")
    except Exception as e:
        print(f"   ✗ Web server error: {e}")
        return False
    
    print("\n5. Test sequence:")
    print("   - Open http://localhost:8080/dashboard in your browser")
    print("   - Check MQTT Log section (blue box)")
    print("   - Simulating weight events in 5 seconds...")
    
    # Wait a bit for dashboard to load
    time.sleep(5)
    
    # Simulate weight events
    test_events = [
        ("CHANGE:-150", "Item picked up"),
        ("CHANGE:+100", "Item returned"),
        ("CHANGE:-75", "Item picked up"),
        ("CHANGE:-200", "Item picked up"),
    ]
    
    print("\n6. Simulating MQTT events...")
    for i, (payload, description) in enumerate(test_events):
        print(f"   Event {i+1}: {description}")
        
        # Simulate MQTT message
        class FakeMsg:
            def __init__(self, topic, payload):
                self.topic = topic
                self.payload = payload.encode('utf-8')
        
        fake_msg = FakeMsg("my-shop/shelf-1/events", payload)
        tracker._on_mqtt_message(None, None, fake_msg)
        
        # Wait a bit between events
        time.sleep(3)
    
    print("\n" + "="*60)
    print("MQTT LOG TEST COMPLETED!")
    print("="*60)
    print("\nExpected results on dashboard:")
    print("1. Green entries in MQTT Log for items picked up")
    print("2. Blue entries for items returned")
    print("3. Yellow entries for unmatched events (if any)")
    print("4. Shopping cart updated with test items")
    print("\nCheck the dashboard at http://localhost:8080/dashboard")
    print("Press Ctrl+C to stop the test system")
    
    try:
        # Keep system running for manual testing
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping test system...")
    
    return True

def main():
    test_mqtt_log()

if __name__ == '__main__':
    main()