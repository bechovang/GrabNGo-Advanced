"""
Test MQTT integration with tracker without camera
"""

import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main_tracker import RetailCustomerTracker

def main():
    print("Creating tracker instance...")
    tracker = RetailCustomerTracker(camera_source=None)  # No camera
    
    print("Initializing MQTT...")
    tracker._init_mqtt()
    
    print("Listening for MQTT messages for 10 seconds...")
    time.sleep(10)
    
    print("\nDone!")
    # Print any received events
    if tracker.events:
        print(f"Received {len(tracker.events)} events:")
        for i, event in enumerate(tracker.events[-5:], 1):  # Show last 5 events
            print(f"  {i}. {event.get('type', 'unknown')}: {event}")
    else:
        print("No events received")

if __name__ == '__main__':
    main()