"""
Test script to check if web server is running and accessible.
Run this to diagnose web server connection issues.
"""

import requests
import socket
import sys

def get_local_ip():
    """Get local IP address."""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def test_server(host='localhost', port=8080):
    """Test if web server is accessible."""
    print("=" * 60)
    print("🔍 Testing Web Server Connection")
    print("=" * 60)
    print()
    
    # Get local IP
    local_ip = get_local_ip()
    
    print(f"📡 Local IP Address: {local_ip}")
    print(f"🌐 Testing server at: http://{host}:{port}")
    print()
    
    # Test 1: Check if port is open
    print("1️⃣  Checking if port is open...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"   ✅ Port {port} is OPEN")
        else:
            print(f"   ❌ Port {port} is CLOSED or not accessible")
            print(f"   💡 Make sure web server is running!")
            return False
    except Exception as e:
        print(f"   ❌ Error checking port: {e}")
        return False
    
    # Test 2: Try to connect to server
    print()
    print("2️⃣  Testing HTTP connection...")
    try:
        url = f"http://{host}:{port}/"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Server is responding!")
            print(f"   📄 Response length: {len(response.text)} bytes")
        else:
            print(f"   ⚠️  Server responded with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to server!")
        print(f"   💡 Possible causes:")
        print(f"      - Web server is not running")
        print(f"      - Firewall is blocking port {port}")
        print(f"      - Wrong IP address")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Connection timeout!")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: Test API endpoint
    print()
    print("3️⃣  Testing API endpoint...")
    try:
        url = f"http://{host}:{port}/qr_zone_status"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"   ✅ API endpoint is working!")
            data = response.json()
            print(f"   📊 Response: {data}")
        else:
            print(f"   ⚠️  API returned status: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  API test failed: {e}")
        print(f"   (This is OK if tracker is not initialized)")
    
    # Summary
    print()
    print("=" * 60)
    print("📋 Summary")
    print("=" * 60)
    print(f"✅ Server is accessible at:")
    print(f"   - http://localhost:{port}")
    print(f"   - http://127.0.0.1:{port}")
    print(f"   - http://{local_ip}:{port}")
    print()
    print("📱 To access from mobile phone:")
    print(f"   1. Make sure phone and computer are on same WiFi")
    print(f"   2. Open browser on phone")
    print(f"   3. Go to: http://{local_ip}:{port}")
    print()
    
    return True

if __name__ == '__main__':
    host = 'localhost'
    port = 8080
    
    # Check command line arguments
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    success = test_server(host, port)
    
    if not success:
        print()
        print("❌ Server is not accessible!")
        print()
        print("🔧 Troubleshooting steps:")
        print("   1. Make sure you ran: python main.py")
        print("   2. Check if you see '✅ Web server started' message")
        print("   3. Check Windows Firewall settings")
        print("   4. Try running web server separately:")
        print("      python web_server.py")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        sys.exit(0)

