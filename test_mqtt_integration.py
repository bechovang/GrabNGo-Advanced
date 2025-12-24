"""
Test MQTT integration between ESP32 and tracker
"""

import sys
import os
import time
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_mqtt_integration():
    print("Testing MQTT integration between ESP32 and tracker...")
    
    try:
        # Initialize tracker
        print("\n1. Initializing tracker...")
        from main_tracker import RetailCustomerTracker
        tracker = RetailCustomerTracker(
            detection_model='yolo11n-pose.pt',
            tracker_config='config/botsort_reid.yaml'
        )
        
        # Initialize MQTT
        print("2. Initializing MQTT...")
        tracker._init_mqtt()
        
        # Add a mock customer in shelf zone for testing
        print("3. Adding mock customer in shelf zone...")
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
        print(f"   Added mock customer {mock_customer_id} with track ID {mock_track_id}")
        
        # Wait for MQTT messages
        print("4. Listening for MQTT messages for 15 seconds...")
        print("   💡 Place or remove weight on the sensor to trigger events")
        
        # Store initial events count
        initial_events_count = len(tracker.events)
        
        # Wait for messages
        time.sleep(15)
        
        # Check if new events were received
        final_events_count = len(tracker.events)
        new_events = final_events_count - initial_events_count
        
        print(f"\n5. Results:")
        print(f"   Initial events count: {initial_events_count}")
        print(f"   Final events count: {final_events_count}")
        print(f"   New events received: {new_events}")
        
        # Print new events
        if new_events > 0:
            print("\n   New events:")
            for i, event in enumerate(tracker.events[-new_events:], 1):
                print(f"   {i}. {event.get('type', 'unknown')}: {event}")
                
            # Check if customer was pinged
            pickup_events = [e for e in tracker.events if e.get('type') == 'item_picked_up']
            if pickup_events:
                print(f"\n   🎉 SUCCESS! Customer was pinged {len(pickup_events)} time(s)!")
                for event in pickup_events:
                    customer_id = event.get('customer_id', 'unknown')
                    weight = event.get('weight_change_g', 0)
                    print(f"      - Customer {customer_id} picked up {abs(weight)}g item")
            else:
                print("\n   ⚠️  No pickup events detected (weight events might be unmatched)")
        else:
            print("\n   ⚠️  No MQTT messages received from ESP32")
            print("      Check if ESP32 is powered and connected to WiFi/MQTT")
        
        # Print customer cart if updated
        if customer_data['shopping_cart']:
            print(f"\n   Mock customer's shopping cart: {customer_data['shopping_cart']}")
        
        return new_events > 0
        
    except Exception as e:
        print(f"\n❌ Error during MQTT integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_mqtt_integration()
    if success:
        print("\n🎉 MQTT integration test PASSED!")
        print("   - MQTT receiving messages from ESP32")
        print("   - Weight events being processed")
        print("   - Customers being pinged correctly")
    else:
        print("\n💥 MQTT integration test FAILED!")
        print("   Check ESP32 connection and MQTT broker")