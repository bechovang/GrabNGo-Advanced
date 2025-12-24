import paho.mqtt.client as mqtt
import time

# Configuration - same as in main_tracker.py
MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC_WEIGHT = "my-shop/shelf-1/events"

message_received = False

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(MQTT_TOPIC_WEIGHT)
        print(f"Subscribed to topic: {MQTT_TOPIC_WEIGHT}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    global message_received
    message_received = True
    payload_str = msg.payload.decode("utf-8")
    print(f"Received message: {payload_str}")

# Create MQTT client
client = mqtt.Client(client_id="test-listener")

# Setup callbacks
client.on_connect = on_connect
client.on_message = on_message

try:
    # Connect to broker
    client.connect(MQTT_BROKER, 1883, 60)
    
    # Start the loop in a non-blocking way
    client.loop_start()
    
    # Wait for 10 seconds to see if any messages arrive
    print("Listening for 10 seconds...")
    time.sleep(10)
    
    client.loop_stop()
    client.disconnect()
    
    if not message_received:
        print("No messages received from ESP32")
        print("ESP32 might not be connected or not sending messages")
    else:
        print("Messages received successfully!")
        
except Exception as e:
    print(f"Error: {e}")