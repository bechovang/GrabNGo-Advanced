"""
Web Server for QR Code Confirmation System
Provides API endpoints for mobile web app to check QR zone status and confirm customers.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile web access

# Global reference to tracker (will be set by main)
tracker = None

@app.route('/')
def index():
    """Serve mobile web app."""
    return send_from_directory('.', 'mobile_qr_scanner.html')

@app.route('/qr_zone_status', methods=['GET'])
def get_qr_zone_status():
    """Return QR zone status for mobile web."""
    if tracker is None:
        return jsonify({
            'zone_active': False,
            'pending_id': None,
            'pending_count': 0,
            'error': 'Tracker not initialized'
        }), 500
    
    # Check zone status
    zone_active, pending_id, pending_count = tracker._check_qr_zone()
    
    return jsonify({
        'zone_active': zone_active and pending_count == 1,
        'pending_id': pending_id,
        'pending_count': pending_count,
        'message': 'Ready to scan' if (zone_active and pending_count == 1) else 
                   f'Multiple people in zone ({pending_count})' if pending_count > 1 else
                   'Please stand in QR zone'
    })

@app.route('/confirm', methods=['POST'])
def confirm_customer():
    """Receive QR scan confirmation from mobile web."""
    print(f"\n📱 POST /confirm received")
    
    if tracker is None:
        print(f"   ❌ Tracker not initialized")
        return jsonify({
            'status': 'error',
            'message': 'Tracker not initialized'
        }), 500
    
    data = request.json
    print(f"   Request data: {data}")
    
    if not data or 'customer_id' not in data:
        print(f"   ❌ Missing customer_id in request")
        return jsonify({
            'status': 'error',
            'message': 'Missing customer_id in request'
        }), 400
    
    customer_id = data.get('customer_id')
    pending_id = data.get('pending_id')  # Optional, will use zone_active_pending if not provided
    
    print(f"   Processing confirmation: customer_id={customer_id}, pending_id={pending_id}")
    
    # Confirm using tracker
    success, message = tracker.confirm_pending_with_customer_id(customer_id, pending_id)
    
    print(f"   Confirmation result: success={success}, message={message}")
    
    if success:
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'pending_id': pending_id or tracker.zone_active_pending,
            'message': message
        })
    else:
        return jsonify({
            'status': 'error',
            'message': message
        }), 400

@app.route('/pending', methods=['GET'])
def get_pending_tracks():
    """Get list of active pending tracks (for debugging)."""
    if tracker is None:
        return jsonify({'pending_tracks': []})
    
    pending_list = []
    for track_id, pending in tracker.pending_tracks.items():
        pending_list.append({
            'pending_id': pending.get('pending_id'),
            'track_id': int(track_id),
            'age_seconds': (time.time() - pending['first_seen'].timestamp()) if 'first_seen' in pending else 0
        })
    
    return jsonify({
        'pending_tracks': pending_list,
        'count': len(pending_list)
    })

def run_server(tracker_instance, host='0.0.0.0', port=8080, debug=False):
    """Run Flask server with tracker instance."""
    global tracker
    tracker = tracker_instance
    
    print(f"\n🌐 Starting Web Server...")
    print(f"   URL: http://{host}:{port}")
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

