"""
Check dashboard MQTT connection
"""

import requests
import time

def test_dashboard_mqtt():
    """Test if dashboard can receive MQTT events"""
    print("="*50)
    print("DASHBOARD MQTT CHECK")
    print("="*50)
    
    print("1. Testing dashboard API...")
    
    try:
        # Test if dashboard API is accessible
        response = requests.get("http://localhost:8080/dashboard/data", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            mqtt_events = data.get('mqtt_events', [])
            print(f"   ✓ Dashboard API accessible")
            print(f"   ✓ MQTT events received: {len(mqtt_events)}")
            
            if mqtt_events:
                print("\nRecent MQTT events:")
                for i, event in enumerate(mqtt_events[:3], 1):
                    print(f"   {i}. {event}")
            else:
                print("\nNo MQTT events received yet")
                print("\nPossible reasons:")
                print("   1. ESP32 not sending messages")
                print("   2. ESP32 not connected to WiFi")
                print("   3. No weight changes on the sensor")
        else:
            print("   ✗ Dashboard API not accessible")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    
    print("If dashboard is not showing MQTT events:")
    print("1. Open dashboard: http://localhost:8080/dashboard")
    print("2. Check connection status indicator (should be green)")
    print("3. Check MQTT Log section (should show events)")
    print("4. Test with: python simple_mqtt_test.py")
    
    return True

def main():
    test_dashboard_mqtt()

if __name__ == '__main__':
    main()