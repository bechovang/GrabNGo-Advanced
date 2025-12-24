"""
Simple MQTT connection test
"""

import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_mqtt():
    """Test MQTT connection"""
    print("="*50)
    print("SIMPLE MQTT CONNECTION TEST")
    print("="*50)
    
    try:
        import paho.mqtt.client as mqtt
        
        print("1. Creating MQTT client...")
        client = mqtt.Client(client_id="simple-test")
        connected = False
        
        def on_connect(client, userdata, flags, rc):
            nonlocal connected
            if rc == 0:
                connected = True
                print("   Connected to MQTT broker")
                client.subscribe("my-shop/shelf-1/events")
                print("   Subscribed to topic: my-shop/shelf-1/events")
            else:
                print(f"   Failed to connect: {rc}")
        
        def on_message(client, userdata, msg):
            payload = msg.payload.decode("utf-8")
            print(f"   Message received: {payload}")
        
        client.on_connect = on_connect
        client.on_message = on_message
        
        print("2. Connecting to MQTT broker...")
        client.connect("test.mosquitto.org", 1883, 60)
        client.loop_start()
        
        print("3. Waiting for messages (15 seconds)...")
        time.sleep(15)
        
        client.loop_stop()
        client.disconnect()
        
        if connected:
            print("   Result: SUCCESS - Connected to MQTT broker")
            return True
        else:
            print("   Result: FAILED - Could not connect to MQTT broker")
            return False
            
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    result = test_mqtt()
    
    print("\n" + "="*50)
    print("TEST RESULT")
    print("="*50)
    
    if result:
        print("SUCCESS: MQTT connection is working")
        print("\nNext steps:")
        print("1. Check if ESP32 is sending messages")
        print("2. Place weight >50g on ESP32 sensor")
        print("3. Run dashboard with 'python run_full_system.py'")
        print("4. Check dashboard for MQTT Log section")
    else:
        print("FAILED: MQTT connection not working")
        print("\nPossible causes:")
        print("1. No internet connection")
        print("2. Firewall blocking port 1883")
        print("3. MQTT broker (test.mosquitto.org) is down")

if __name__ == '__main__':
    main()