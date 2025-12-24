"""
Test complete flow with simulated MQTT messages
"""

import sys
import os
import time
import threading
import paho.mqtt.client as mqtt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def simulate_mqtt_message(topic, payload):
    """Simulate an MQTT message directly to the tracker"""
    print(f"Simulating MQTT message: {payload}")
    
    # Create a fake message object
    class FakeMsg:
        def __init__(self, topic, payload):
            self.topic = topic
            self.payload = payload.encode('utf-8')
    
    # Initialize tracker to test
    from main_tracker import RetailCustomerTracker
    tracker = RetailCustomerTracker(
        detection_model='yolo11n-pose.pt',
        tracker_config='config/botsort_reid.yaml'
    )
    
    # Add a mock customer in shelf zone
    mock_customer_id = f"MOCK_{int(time.time())}"
    mock_track_id = 999
    
    # Create mock customer data
    customer_data = {
        'track_id': mock_track_id,
        'customer_id': mock_customer_id,
        'first_seen': time.time(),
        'last_seen': time.time(),
        'last_box': [200, 200, 400, 600],  # Box in shelf zone
        'keypoints': None,  # No keypoints for simplicity
        'confirmed': True,
        'shopping_cart': [],
        'pickup_count': 0,
        'last_pickup_time': None
    }
    
    # Add to customers
    tracker.customers[mock_track_id] = customer_data
    
    # Initialize shelf zone
    tracker._update_shelf_zone((720, 1280))  # height, width
    
    # Simulate receiving the message
    fake_msg = FakeMsg(topic, payload)
    tracker._on_mqtt_message(None, None, fake_msg)
    
    # Check if customer was pinged
    pickup_events = [e for e in tracker.events if e.get('type') == 'item_picked_up']
    if pickup_events:
        print(f"SUCCESS! Customer was pinged!")
        for event in pickup_events:
            customer_id = event.get('customer_id', 'unknown')
            weight = event.get('weight_change_g', 0)
            print(f"   - Customer {customer_id} picked up {abs(weight)}g item")
        return True
    else:
        print("Customer was not pinged")
        return False

def test_mqtt_handler():
    """Test the MQTT message handling logic"""
    print("Testing MQTT message handling...")
    
    # Test cases
    test_cases = [
        ("my-shop/shelf-1/events", "CHANGE:-100"),  # Item picked up
        ("my-shop/shelf-1/events", "CHANGE:-200"),  # Item picked up
    ]
    
    success_count = 0
    for topic, payload in test_cases:
        print(f"\nTesting: {payload}")
        if simulate_mqtt_message(topic, payload):
            success_count += 1
        time.sleep(1)  # Avoid rate limiting
    
    print(f"\nResults: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_camera_and_mqtt():
    """Test camera and MQTT integration with simulated messages"""
    print("\nTesting camera and MQTT integration...")
    
    try:
        # Initialize tracker
        from main_tracker import RetailCustomerTracker
        tracker = RetailCustomerTracker(
            detection_model='yolo11n-pose.pt',
            tracker_config='config/botsort_reid.yaml'
        )
        
        # Initialize MQTT
        tracker._init_mqtt()
        
        # Open camera
        print("Opening camera...")
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Failed to open camera, continuing with MQTT test only...")
        else:
            print("Camera opened successfully!")
            
            # Process a few frames
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    result, _, active_tracks = tracker.process_frame(
                        frame,
                        conf=0.5,
                        iou=0.7,
                        return_annotated=False
                    )
                    print(f"Frame {i+1}: Processed successfully")
                time.sleep(0.5)
            
            cap.release()
        
        # Add a mock customer
        mock_customer_id = f"MOCK_{int(time.time())}"
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
        
        # Initialize shelf zone
        tracker._update_shelf_zone((720, 1280))  # height, width
        
        # Check initial stats
        initial_events = len(tracker.events)
        
        # Simulate MQTT message
        print("\nSimulating MQTT message...")
        class FakeMsg:
            def __init__(self, topic, payload):
                self.topic = topic
                self.payload = payload.encode('utf-8')
        
        fake_msg = FakeMsg("my-shop/shelf-1/events", "CHANGE:-150")
        tracker._on_mqtt_message(None, None, fake_msg)
        
        # Check if event was processed
        final_events = len(tracker.events)
        if final_events > initial_events:
            print("Event was processed!")
            
            # Check if customer was pinged
            pickup_events = [e for e in tracker.events if e.get('type') == 'item_picked_up']
            if pickup_events:
                print("Customer was pinged!")
                for event in pickup_events:
                    print(f"   - Customer {event.get('customer_id')} picked up {abs(event.get('weight_change_g', 0))}g item")
                    print(f"   - Shopping cart: {customer_data['shopping_cart']}")
                return True
        
        print("Event was not processed or customer was not pinged")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("COMPLETE FLOW TEST")
    print("="*50)
    
    # Test 1: MQTT message handling
    test1_result = test_mqtt_handler()
    
    # Test 2: Camera and MQTT integration
    test2_result = test_camera_and_mqtt()
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print(f"MQTT Handler Test: {'PASSED' if test1_result else 'FAILED'}")
    print(f"Camera + MQTT Test: {'PASSED' if test2_result else 'FAILED'}")
    
    if test1_result and test2_result:
        print("\nOVERALL RESULT: PASSED!")
        print("The system is working correctly. ESP32 needs to be checked separately.")
    else:
        print("\nOVERALL RESULT: FAILED!")
        print("Check the system configuration and try again.")

if __name__ == '__main__':
    main()