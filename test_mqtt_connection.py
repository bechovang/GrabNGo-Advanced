"""
Test MQTT connection to diagnose connection issues
"""

import sys
import socket
import time

def test_network_connectivity(broker="test.mosquitto.org", port=1883):
    """Test if we can reach the MQTT broker."""
    print(f"🔍 Testing network connectivity to {broker}:{port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((broker, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Network connection successful!")
            return True
        else:
            print(f"❌ Cannot connect to {broker}:{port}")
            print(f"   Error code: {result}")
            print(f"   Possible causes:")
            print(f"   - Firewall blocking port {port}")
            print(f"   - Network connectivity issues")
            print(f"   - Broker is down")
            return False
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def test_mqtt_connection(broker="test.mosquitto.org", port=1883, topic="my-shop/shelf-1/events"):
    """Test MQTT connection using paho-mqtt."""
    print(f"\n🔍 Testing MQTT connection...")
    
    # Check if paho-mqtt is installed
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("❌ paho-mqtt is not installed!")
        print("   Install with: pip install paho-mqtt")
        return False
    
    print(f"✅ paho-mqtt is installed")
    
    connection_result = {"connected": False, "error": None}
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            connection_result["connected"] = True
            print(f"✅ MQTT connected successfully!")
            print(f"   Subscribing to topic: {topic}")
            client.subscribe(topic)
        else:
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_messages.get(rc, f"Unknown error code {rc}")
            connection_result["error"] = f"Code {rc}: {error_msg}"
            print(f"❌ MQTT connection failed: {error_msg}")
    
    def on_message(client, userdata, msg):
        print(f"📨 Received message on {msg.topic}: {msg.payload.decode()}")
    
    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print(f"⚠️  Unexpected disconnection (code {rc})")
    
    try:
        import uuid
        client_id = f"test-client-{uuid.uuid4().hex[:8]}"
        client = mqtt.Client(client_id=client_id)
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        print(f"   Connecting to {broker}:{port}...")
        print(f"   Client ID: {client_id}")
        
        # Try to connect
        client.connect_async(broker, port, 60)
        client.loop_start()
        
        # Wait for connection (max 10 seconds)
        for i in range(20):
            time.sleep(0.5)
            if connection_result["connected"]:
                print(f"✅ Connection established!")
                time.sleep(2)  # Wait a bit more to see if subscription works
                client.loop_stop()
                client.disconnect()
                return True
            if connection_result["error"]:
                print(f"❌ Connection failed: {connection_result['error']}")
                client.loop_stop()
                return False
        
        print(f"⏱️  Connection timeout (waited 10 seconds)")
        client.loop_stop()
        client.disconnect()
        return False
        
    except Exception as e:
        print(f"❌ MQTT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("MQTT CONNECTION TEST")
    print("="*60)
    
    broker = "test.mosquitto.org"
    port = 1883
    topic = "my-shop/shelf-1/events"
    
    # Test 1: Network connectivity
    print("\n[TEST 1] Network Connectivity")
    print("-" * 60)
    network_ok = test_network_connectivity(broker, port)
    
    if not network_ok:
        print("\n⚠️  Network test failed. MQTT cannot connect.")
        print("   Solutions:")
        print("   1. Check your internet connection")
        print("   2. Check if firewall is blocking port 1883")
        print("   3. Try using a different MQTT broker")
        print("   4. Use a local MQTT broker (localhost)")
        sys.exit(1)
    
    # Test 2: MQTT connection
    print("\n[TEST 2] MQTT Connection")
    print("-" * 60)
    mqtt_ok = test_mqtt_connection(broker, port, topic)
    
    if mqtt_ok:
        print("\n✅ All tests passed! MQTT should work in the dashboard.")
    else:
        print("\n❌ MQTT connection test failed.")
        print("   The dashboard will show MQTT as disconnected.")
        print("\n   Troubleshooting:")
        print("   1. Check if paho-mqtt is installed: pip install paho-mqtt")
        print("   2. Check your network/firewall settings")
        print("   3. Try a different MQTT broker")
        print("   4. Check if the broker requires authentication")
