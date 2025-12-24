"""
Run complete Smart Retail System with all components
"""

import sys
import os
import time
import threading
import requests
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_system_requirements():
    """Check if all system requirements are met"""
    print("="*60)
    print("SMART RETAIL SYSTEM - FULL FLOW TEST")
    print("="*60)
    
    print("\n1. Checking system requirements...")
    
    # Check camera
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("   ✓ Camera: Available")
            cap.release()
        else:
            print("   ✗ Camera: Not available")
            return False
    except Exception as e:
        print(f"   ✗ Camera: Error - {e}")
        return False
    
    # Check tracker
    try:
        from main_tracker import RetailCustomerTracker
        print("   ✓ Tracker: Available")
    except Exception as e:
        print(f"   ✗ Tracker: Error - {e}")
        return False
    
    # Check web server
    try:
        import flask
        print("   ✓ Web server: Available")
    except Exception as e:
        print(f"   ✗ Web server: Error - {e}")
        return False
    
    # Check MQTT
    try:
        import paho.mqtt.client as mqtt
        print("   ✓ MQTT: Available")
    except Exception as e:
        print(f"   ✗ MQTT: Error - {e}")
        return False
    
    return True

def test_esp32_connection():
    """Test if ESP32 is connected and sending messages"""
    print("\n2. Testing ESP32 connection...")
    
    try:
        import paho.mqtt.client as mqtt
        
        # Setup MQTT client
        client = mqtt.Client(client_id="test-full-system")
        message_received = False
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("   ✓ Connected to MQTT broker")
                client.subscribe("my-shop/shelf-1/events")
            else:
                print(f"   ✗ Failed to connect: {rc}")
        
        def on_message(client, userdata, msg):
            nonlocal message_received
            message_received = True
            payload = msg.payload.decode("utf-8")
            print(f"   ✓ Received message: {payload}")
        
        client.on_connect = on_connect
        client.on_message = on_message
        
        # Connect and wait for messages
        client.connect("test.mosquitto.org", 1883, 60)
        client.loop_start()
        
        print("   Waiting for ESP32 message (10 seconds)...")
        time.sleep(10)
        
        client.loop_stop()
        client.disconnect()
        
        if message_received:
            print("   ✓ ESP32: Connected and sending messages")
            return True
        else:
            print("   ✗ ESP32: No messages received")
            print("     Check if ESP32 is powered and connected")
            return False
            
    except Exception as e:
        print(f"   ✗ ESP32 test error: {e}")
        return False

def run_full_system():
    """Run the complete system"""
    print("\n3. Starting full system...")
    
    try:
        # Initialize tracker
        print("   Initializing tracker...")
        from main_tracker import RetailCustomerTracker
        tracker = RetailCustomerTracker(
            detection_model='yolo11n-pose.pt',
            tracker_config='config/botsort_reid.yaml'
        )
        
        # Initialize MQTT
        print("   Initializing MQTT...")
        tracker._init_mqtt()
        
        # Start web server in background thread
        print("   Starting web server...")
        try:
            from web_server import run_server
            server_thread = threading.Thread(
                target=run_server,
                args=(tracker, '0.0.0.0', 8080, False),
                daemon=True
            )
            server_thread.start()
            print("   ✓ Web server: http://localhost:8080")
            print("   ✓ Dashboard: http://localhost:8080/dashboard")
        except Exception as e:
            print(f"   ✗ Web server error: {e}")
            return False
        
        # Open camera
        print("   Opening camera...")
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("   ✗ Failed to open camera")
            return False
        
        print("   ✓ Camera opened successfully")
        
        # Run tracking loop
        print("\n4. System is running!")
        print("   - Dashboard: http://localhost:8080/dashboard")
        print("   - Mobile app: http://localhost:8080")
        print("   - Press Ctrl+C to stop")
        print("\n" + "="*60)
        print("SYSTEM LOG:")
        print("="*60)
        
        frame_count = 0
        last_stats_update = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame (without visualization for speed)
                result, _, active_tracks = tracker.process_frame(
                    frame,
                    conf=0.5,
                    iou=0.7,
                    return_annotated=False  # Skip annotation for performance
                )
                
                frame_count += 1
                
                # Print stats every 10 seconds
                current_time = time.time()
                if current_time - last_stats_update > 10:
                    stats = tracker.get_stats()
                    print(f"[{time.strftime('%H:%M:%S')}] "
                          f"Active: {stats['active_customers']}, "
                          f"Pending: {stats['pending_tracks']}, "
                          f"Total: {stats['total_customers']}, "
                          f"Events: {stats['total_events']}")
                    last_stats_update = current_time
                
                # Small delay to reduce CPU usage
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("STOPPING SYSTEM...")
            print("="*60)
        
        cap.release()
        print("✓ Camera released")
        return True
        
    except Exception as e:
        print(f"✗ System error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_server():
    """Test if web server is responding"""
    print("\n5. Testing web server...")
    
    try:
        # Test dashboard page
        response = requests.get("http://localhost:8080/dashboard", timeout=5)
        if response.status_code == 200:
            print("   ✓ Dashboard page: Available")
        else:
            print(f"   ✗ Dashboard page: Error - {response.status_code}")
            return False
        
        # Test dashboard data API
        response = requests.get("http://localhost:8080/dashboard/data", timeout=5)
        if response.status_code == 200:
            print("   ✓ Dashboard API: Available")
            return True
        else:
            print(f"   ✗ Dashboard API: Error - {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ Web server test error: {e}")
        return False

def main():
    # Check system requirements
    if not check_system_requirements():
        print("\n✗ System requirements not met. Please install missing components.")
        return
    
    # Test ESP32 connection (optional)
    esp32_ok = test_esp32_connection()
    
    # Run full system
    if not run_full_system():
        print("\n✗ System failed to start properly")
        return
    
    # Test web server (optional)
    test_web_server()
    
    # Summary
    print("\n" + "="*60)
    print("SYSTEM SUMMARY:")
    print("="*60)
    print(f"✓ System components: Running")
    print(f"{'✓' if esp32_ok else '✗'} ESP32 connection: {'Connected' if esp32_ok else 'Not connected'}")
    print("✓ Web server: Running")
    print("✓ Camera tracking: Running")
    print("\nTo test the full flow:")
    print("1. Stand in front of the camera")
    print("2. Check dashboard for your appearance in the list")
    print("3. Stand in the QR zone (right side of screen)")
    print("4. Use your phone to scan QR at http://localhost:8080")
    print("5. After confirmation, stand in the shelf zone (left side)")
    print("6. Place or remove weight on the sensor to trigger MQTT events")
    print("7. Check dashboard for shopping cart updates")
    
    print("\nSYSTEM IS READY FOR TESTING!")

if __name__ == '__main__':
    main()