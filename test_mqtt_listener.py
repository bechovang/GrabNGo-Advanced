#!/usr/bin/env python3
"""
Test MQTT listener to check if ESP32 is sending messages
Connects to the same broker and topic as the main system
"""

import paho.mqtt.client as mqtt
import time
import json

# Configuration - same as in main_tracker.py
MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC_WEIGHT = "my-shop/shelf-1/events"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
        client.subscribe(MQTT_TOPIC_WEIGHT)
        print(f"Subscribed to topic: {MQTT_TOPIC_WEIGHT}")
        print("\nWaiting for messages from ESP32...")
        print("Place or remove weight on the sensor to trigger messages")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        # Decode message
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8")
        
        print(f"\nReceived message:")
        print(f"   Topic: {topic}")
        print(f"   Payload: {payload_str}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse weight change if message format is correct
        if payload_str.startswith("CHANGE:"):
            try:
                weight_change = int(payload_str[7:])  # Remove "CHANGE:" prefix
                print(f"   Weight change: {weight_change}g")
                
                if weight_change < 0:
                    print(f"   -> Item picked up ({abs(weight_change)}g)")
                else:
                    print(f"   -> Item placed ({weight_change}g)")
            except ValueError:
                print(f"   -> Invalid weight value")
        else:
            print(f"   -> Unknown message format")
    except Exception as e:
        print(f"   Error processing message: {e}")

def main():
    print("Starting MQTT Listener Test")
    print(f"   Broker: {MQTT_BROKER}")
    print(f"   Topic: {MQTT_TOPIC_WEIGHT}")
    print()
    
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
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping listener...")
            client.loop_stop()
            client.disconnect()
            print("Disconnected from MQTT broker")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()