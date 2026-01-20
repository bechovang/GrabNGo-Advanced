"""
MQTT Client Module for Weight-Based Pickup Detection
Handles MQTT connection and weight event subscriptions.
"""

import uuid
import socket
import time
from datetime import datetime
from collections import deque


class MQTTClient:
    """MQTT client wrapper for weight event subscriptions."""
    
    def __init__(self, broker="test.mosquitto.org", topic="my-shop/shelf-1/events", 
                 on_weight_event=None):
        """
        Initialize MQTT client.
        
        Args:
            broker: MQTT broker address
            topic: MQTT topic to subscribe to
            on_weight_event: Callback function(weight_change_g, timestamp) when weight event received
        """
        self.broker = broker
        self.topic = topic
        self.on_weight_event = on_weight_event
        self.mqtt_client = None
        self.connected = False
        self.recent_weight_events = deque(maxlen=10)  # Last 10 events
        
        # Check if paho-mqtt is available
        try:
            import paho.mqtt.client as mqtt
            self._mqtt_module = mqtt
        except ImportError:
            self._mqtt_module = None
            print("⚠️  paho-mqtt not installed. Install with: pip install paho-mqtt")
            print("   Weight-based pickup detection will be disabled")
    
    def connect(self):
        """Initialize and connect to MQTT broker."""
        if self._mqtt_module is None:
            return False
        
        try:
            # Test network connectivity first
            print(f"🔍 Testing connection to {self.broker}:1883...")
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(5)  # 5 second timeout
                result = test_socket.connect_ex((self.broker, 1883))
                test_socket.close()
                
                if result != 0:
                    print(f"⚠️  Cannot reach {self.broker}:1883 (connection refused or timeout)")
                    print(f"   This might be due to:")
                    print(f"   - Firewall blocking port 1883")
                    print(f"   - Network connectivity issues")
                    print(f"   - Broker is down")
                    print(f"   💡 Tip: Try using a local MQTT broker or check your network")
                    self.connected = False
                    return False
            except Exception as e:
                print(f"⚠️  Network test failed: {e}")
                print(f"   Cannot verify connectivity to {self.broker}")
                # Continue anyway, let MQTT client handle the connection
            
            # Create client
            client_id = f"cv-system-{uuid.uuid4().hex[:8]}"
            self.mqtt_client = self._mqtt_module.Client(client_id=client_id)
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
            self.mqtt_client.on_disconnect = self._on_disconnect
            
            # Set connection timeout and connect
            print(f"   Attempting to connect to {self.broker}:1883...")
            try:
                self.mqtt_client.connect_async(self.broker, 1883, 60)
                self.mqtt_client.loop_start()  # Start background thread
                
                print(f"✅ MQTT client initialized (connecting to {self.broker}...)")
                print(f"   Waiting for connection... (this may take a few seconds)")
                
                # Wait a bit for connection to establish
                time.sleep(2)
                
                # Check connection status
                if not self.connected:
                    print(f"⚠️  MQTT connection pending... (checking again in 3 seconds)")
                    # Give it more time
                    time.sleep(3)
                    if not self.connected:
                        print(f"❌ MQTT connection timeout - broker may be unreachable")
                        print(f"   Current status: connected = {self.connected}")
                        return False
                return True
            except Exception as conn_error:
                print(f"❌ Failed to start MQTT connection: {conn_error}")
                self.connected = False
                return False
                    
        except Exception as e:
            print(f"⚠️  MQTT initialization failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("   Weight-based pickup detection will be disabled")
            self.connected = False
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when MQTT client connects."""
        if rc == 0:
            self.connected = True
            # Subscribe to weight events topic
            result = client.subscribe(self.topic)
            if result[0] == 0:
                print(f"✅ MQTT connected to {self.broker}")
                print(f"   Subscribed to: {self.topic}")
            else:
                print(f"⚠️  MQTT connected but subscription failed with code {result[0]}")
        else:
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_messages.get(rc, f"Unknown error code {rc}")
            print(f"❌ MQTT connection failed with code {rc}: {error_msg}")
            print(f"   Broker: {self.broker}:1883")
            print(f"   Topic: {self.topic}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when MQTT client disconnects."""
        self.connected = False
        if rc != 0:
            print(f"⚠️  MQTT disconnected unexpectedly (code {rc})")
        else:
            print("ℹ️  MQTT disconnected")
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT weight change events."""
        try:
            topic_str = msg.topic.decode('utf-8') if isinstance(msg.topic, bytes) else msg.topic
            message_str = msg.payload.decode('utf-8') if isinstance(msg.payload, bytes) else str(msg.payload)
            
            if topic_str == self.topic:
                # Parse: "CHANGE:-480"
                if message_str.startswith("CHANGE:"):
                    weight_change = int(message_str.split(":")[1])
                    timestamp = datetime.now()
                    
                    # Store event
                    self.recent_weight_events.append({
                        'weight_change_g': weight_change,
                        'timestamp': timestamp
                    })
                    
                    # Call callback if provided
                    if self.on_weight_event:
                        self.on_weight_event(weight_change, timestamp)
        except Exception as e:
            # Silent error handling (no print)
            pass
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.connected = False

