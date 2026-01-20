"""
Web Server for QR Code Confirmation System
Flask app initialization and server runner.
"""

from flask import Flask
from flask_cors import CORS
import os

# Get the directory of this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_static_dir = os.path.join(_current_dir, 'static')

# Initialize Flask app with static folder
app = Flask(__name__, static_folder=_static_dir, static_url_path='')
CORS(app)  # Enable CORS for mobile web access

# Global reference to tracker (will be set by main)
tracker = None

def run_server(tracker_instance, host='0.0.0.0', port=8080, debug=False):
    """Run Flask server with tracker instance."""
    # Import routes here to avoid circular import
    from . import routes
    routes.set_tracker(tracker_instance)  # Set tracker in routes module
    
    print(f"\n🌐 Starting Web Server...")
    print(f"   URL: http://{host}:{port}")
    print(f"   Dashboard: http://{host}:{port}/dashboard")
    print(f"   QR Zone Status: http://{host}:{port}/qr_zone_status")
    print(f"   Confirm Endpoint: http://{host}:{port}/confirm")
    print(f"   Mobile App: http://{host}:{port}/")
    print(f"   (Access from mobile: http://<your-ip>:{port}/)\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    # For testing without tracker
    print("⚠️  Running web server without tracker (for testing)")
    print("   Use run_server(tracker_instance) from main script")
    app.run(host='0.0.0.0', port=8080, debug=True)


