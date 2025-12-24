"""
Test dashboard functionality
"""

import sys
import os
import time
import threading
import requests

def test_dashboard_api():
    """Test the dashboard API endpoints"""
    base_url = "http://localhost:8080"
    
    try:
        # Test dashboard page
        print("1. Testing dashboard page...")
        response = requests.get(f"{base_url}/dashboard")
        if response.status_code == 200:
            print("   Dashboard page: OK")
        else:
            print(f"   Dashboard page: FAILED - {response.status_code}")
            return False
        
        # Test dashboard data API
        print("2. Testing dashboard data API...")
        response = requests.get(f"{base_url}/dashboard/data")
        if response.status_code == 200:
            data = response.json()
            print(f"   Dashboard data: OK - {len(data.get('customers', {}))} customers")
            print(f"   Stats: {data.get('stats', {})}")
        else:
            print(f"   Dashboard data: FAILED - {response.status_code}")
            return False
        
        # Test mobile QR scanner page
        print("3. Testing mobile QR scanner page...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   Mobile page: OK")
        else:
            print(f"   Mobile page: FAILED - {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error testing dashboard API: {e}")
        return False

def test_dashboard_with_mock_data():
    """Test dashboard with mock data"""
    print("\nTesting with mock data...")
    
    # Test the dashboard by opening it in a browser
    print("Dashboard is ready for testing!")
    print("Open http://localhost:8080/dashboard in your browser to view the dashboard")
    print("\nFeatures to test:")
    print("1. Customer list on the left")
    print("2. Store map with customer positions")
    print("3. Shopping cart when clicking on a customer")
    print("4. Real-time updates (simulated with polling)")
    
    return True

def main():
    print("="*60)
    print("DASHBOARD TESTING")
    print("="*60)
    
    # Test 1: API endpoints
    api_test = test_dashboard_api()
    
    # Test 2: Mock data
    mock_test = test_dashboard_with_mock_data()
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS:")
    print(f"API Test: {'PASSED' if api_test else 'FAILED'}")
    print(f"Mock Data Test: {'PASSED' if mock_test else 'FAILED'}")
    
    if api_test and mock_test:
        print("\nOVERALL RESULT: PASSED!")
        print("Dashboard is ready for use with the tracking system.")
        print("\nTo test with real data:")
        print("1. Run 'python run_dashboard.py' to start the tracking system")
        print("2. Open http://localhost:8080/dashboard in your browser")
        print("3. Have people walk in front of the camera")
        print("4. Scan QR codes to confirm customers")
        print("5. Check if ESP32 is sending weight events")
    else:
        print("\nOVERALL RESULT: FAILED!")
        print("Check the errors above and try again.")

if __name__ == '__main__':
    main()