"""
Get network IP address for accessing dashboard from other devices
"""

import socket
import requests

def get_local_ip():
    """Get the local IP address"""
    try:
        # Create a socket and connect to a remote server
        # This doesn't actually send data, just determines the interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            # Fallback method
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "127.0.0.1"  # localhost

def check_port_open(ip, port):
    """Check if a port is open on the given IP"""
    try:
        url = f"http://{ip}:{port}"
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    print("="*50)
    print("NETWORK IP INFORMATION")
    print("="*50)
    
    ip = get_local_ip()
    port = 8080
    
    print(f"Local IP: {ip}")
    print(f"Port: {port}")
    
    # Check if server is running
    server_running = check_port_open(ip, port)
    
    if server_running:
        print("\nServer Status: RUNNING")
    else:
        print("\nServer Status: NOT RUNNING or PORT BLOCKED")
    
    print("\nAccess URLs:")
    print(f"- Local: http://localhost:{port}")
    print(f"- Network: http://{ip}:{port}")
    print(f"- Dashboard: http://{ip}:{port}/dashboard")
    
    if not server_running:
        print("\nTo start the server, run:")
        print("  python run_full_system.py")
    
    print("\nQR Code Content:")
    print(f"http://{ip}:{port}")
    
    print("\nNote: Make sure devices are on the same network")
    print("      and firewall allows port 8080")

if __name__ == '__main__':
    main()