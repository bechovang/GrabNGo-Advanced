"""
Simple test to check if tracker can be initialized
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Importing RetailCustomerTracker...")
    from main_tracker import RetailCustomerTracker
    print("✅ Import successful")
    
    print("\nInitializing tracker (this may take a while to load YOLO models)...")
    tracker = RetailCustomerTracker(
        detection_model='yolo11n-pose.pt',
        tracker_config='config/botsort_reid.yaml'
    )
    print("✅ Tracker initialized successfully")
    
    print("\nInitializing MQTT...")
    tracker._init_mqtt()
    print("✅ MQTT initialized successfully")
    
    print("\nGetting initial stats...")
    stats = tracker.get_stats()
    print(f"✅ Stats: {stats}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()