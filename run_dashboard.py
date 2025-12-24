"""
Run Dashboard Only (No Camera) - For MQTT Monitoring
"""

import sys
import os
import time
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("="*60)
    print("SMART RETAIL DASHBOARD (MQTT Only)")
    print("="*60)
    print("\nThis mode runs the dashboard without camera.")
    print("Perfect for monitoring MQTT events and customer data.\n")
    
    # Initialize tracker
    print("1. Initializing tracker...")
    from main_tracker import RetailCustomerTracker
    tracker = RetailCustomerTracker(
        detection_model='yolo11n-pose.pt',
        tracker_config='config/botsort_reid.yaml'
    )
    print("   ✓ Tracker initialized")
    
    # Initialize MQTT
    print("\n2. Initializing MQTT...")
    tracker._init_mqtt()
    print("   ✓ MQTT initialization complete")
    
    # Start web server in background thread
    print("\n3. Starting web server...")
    try:
        from web_server import run_server
        server_thread = threading.Thread(
            target=run_server,
            args=(tracker, '0.0.0.0', 8080, False),
            daemon=True
        )
        server_thread.start()
        time.sleep(1)  # Give server time to start
        print("   ✓ Web server started")
        print("   ✓ Dashboard: http://localhost:8080/dashboard")
        print("   ✓ Mobile app: http://localhost:8080")
    except Exception as e:
        print(f"   ✗ Web server error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("SYSTEM RUNNING")
    print("="*60)
    print("\n📊 Dashboard: http://localhost:8080/dashboard")
    print("📱 Mobile QR Scanner: http://localhost:8080")
    print("\n💡 This mode is for MQTT monitoring only.")
    print("   No camera tracking - dashboard will show MQTT events.")
    print("\n⌨️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        # Keep running and show periodic status
        last_status_time = time.time()
        while True:
            time.sleep(5)
            
            # Show status every 30 seconds
            current_time = time.time()
            if current_time - last_status_time >= 30:
                stats = tracker.get_stats()
                mqtt_status = "Connected" if tracker.mqtt_connected else "Disconnected"
                print(f"[{time.strftime('%H:%M:%S')}] Status: "
                      f"MQTT={mqtt_status}, "
                      f"Events={stats['total_events']}, "
                      f"Customers={stats['total_customers']}")
                last_status_time = current_time
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutting down...")
        print("✅ System stopped.")

if __name__ == '__main__':
    main()
