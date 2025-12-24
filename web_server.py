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

@app.route('/dashboard', methods=['GET'])
def dashboard_page():
    """Serve the dashboard page."""
    return send_from_directory('.', 'dashboard.html')

@app.route('/dashboard/data', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data including customers and statistics."""
    if tracker is None:
        return jsonify({
            'customers': {},
            'stats': {
                'total_customers': 0,
                'active_customers': 0,
                'items_taken': 0,
                'avg_time': '0m'
            }
        })
    
    # Prepare customers data
    customers_data = {}
    
    # Add confirmed customers
    for track_id, customer in tracker.customers.items():
        duration = (time.time() - customer['first_seen'].timestamp()) if 'first_seen' in customer else 0
        
        customers_data[f"customer_{track_id}"] = {
            'track_id': track_id,
            'customer_id': customer.get('customer_id', 'UNKNOWN'),
            'confirmed': customer.get('confirmed', False),
            'first_seen': customer['first_seen'].isoformat() if 'first_seen' in customer else None,
            'last_seen': customer['last_seen'].isoformat() if 'last_seen' in customer else None,
            'duration': duration,
            'last_box': customer.get('last_box'),
            'shopping_cart': customer.get('shopping_cart', []),
            'pickup_count': customer.get('pickup_count', 0)
        }
    
    # Add pending customers
    for pending_id, pending in tracker.pending_tracks.items():
        duration = (time.time() - pending['first_seen'].timestamp()) if 'first_seen' in pending else 0
        
        customers_data[f"pending_{pending_id}"] = {
            'track_id': int(pending_id),
            'customer_id': pending.get('pending_id', 'UNKNOWN'),
            'confirmed': False,
            'first_seen': pending['first_seen'].isoformat() if 'first_seen' in pending else None,
            'last_seen': pending['last_seen'].isoformat() if 'last_seen' in pending else None,
            'duration': duration,
            'last_box': pending.get('last_box'),
            'shopping_cart': [],
            'pickup_count': 0
        }
    
    # Calculate statistics
    total_customers = len(customers_data)
    active_customers = len([c for c in customers_data.values() if c['confirmed']])
    
    # Calculate total items taken
    items_taken = 0
    for customer in customers_data.values():
        items_taken += len(customer['shopping_cart'])
    
    # Calculate average time
    if customers_data:
        total_duration = sum(customer['duration'] for customer in customers_data.values())
        avg_duration = total_duration / len(customers_data)
        avg_minutes = int(avg_duration / 60)
        avg_seconds = int(avg_duration % 60)
        avg_time = f"{avg_minutes}m {avg_seconds}s" if avg_minutes > 0 else f"{avg_seconds}s"
    else:
        avg_time = "0m"
    
    # Get recent MQTT events (last 20)
    mqtt_events = []
    if tracker and hasattr(tracker, 'events'):
        # Get only recent MQTT-related events
        for event in tracker.events:
            if event.get('type') in ['item_picked_up', 'unmatched_weight_event']:
                event_copy = event.copy()
                if isinstance(event_copy.get('timestamp'), str):
                    pass  # Already formatted
                elif hasattr(event_copy.get('timestamp'), 'isoformat'):
                    event_copy['timestamp'] = event_copy['timestamp'].isoformat()
                elif event_copy.get('timestamp'):
                    event_copy['timestamp'] = str(event_copy['timestamp'])
                mqtt_events.append(event_copy)
        
        # Sort by timestamp (newest first) and limit to 20
        mqtt_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        mqtt_events = mqtt_events[:20]

    # Get zone settings
    qr_zone_percent = getattr(tracker, 'qr_zone_percent', {
        'x1_percent': 0.7,
        'y1_percent': 0.0,
        'x2_percent': 1.0,
        'y2_percent': 1.0
    })
    
    shelf_zone_percent = getattr(tracker, 'shelf_zone_percent', {
        'x1_percent': 0.0,
        'y1_percent': 0.3,
        'x2_percent': 0.5,
        'y2_percent': 0.9
    })
    
    # Get MQTT connection status
    mqtt_connected = getattr(tracker, 'mqtt_connected', False)
    mqtt_broker = getattr(tracker, 'mqtt_broker', 'unknown')
    mqtt_topic = getattr(tracker, 'mqtt_topic_weight', 'unknown')
    
    # Debug logging (only print occasionally to avoid spam)
    import time
    if not hasattr(get_dashboard_data, '_last_debug_time'):
        get_dashboard_data._last_debug_time = 0
    
    current_time = time.time()
    if current_time - get_dashboard_data._last_debug_time > 10:  # Every 10 seconds
        print(f"[Dashboard] MQTT Status: connected={mqtt_connected}, broker={mqtt_broker}, topic={mqtt_topic}")
        get_dashboard_data._last_debug_time = current_time
    
    mqtt_status = {
        'connected': mqtt_connected,
        'broker': mqtt_broker,
        'topic': mqtt_topic
    }
    
    # Return dashboard data
    return jsonify({
        'customers': customers_data,
        'stats': {
            'total_customers': total_customers,
            'active_customers': active_customers,
            'items_taken': items_taken,
            'avg_time': avg_time
        },
        'mqtt_events': mqtt_events,
        'mqtt_status': mqtt_status,
        'zones': {
            'qr_zone': qr_zone_percent,
            'shelf_zone': shelf_zone_percent
        }
    })

@app.route('/dashboard/zones', methods=['GET'])
def get_zone_settings():
    """Get current zone settings."""
    if tracker is None:
        return jsonify({'error': 'Tracker not initialized'}), 500
    
    qr_zone_percent = getattr(tracker, 'qr_zone_percent', {
        'x1_percent': 0.7,
        'y1_percent': 0.0,
        'x2_percent': 1.0,
        'y2_percent': 1.0
    })
    
    shelf_zone_percent = getattr(tracker, 'shelf_zone_percent', {
        'x1_percent': 0.0,
        'y1_percent': 0.3,
        'x2_percent': 0.5,
        'y2_percent': 0.9
    })
    
    return jsonify({
        'qr_zone': qr_zone_percent,
        'shelf_zone': shelf_zone_percent
    })

@app.route('/dashboard/zones', methods=['POST'])
def update_zone_settings():
    """Update zone settings."""
    if tracker is None:
        return jsonify({'error': 'Tracker not initialized'}), 500
    
    try:
        data = request.json
        
        # Update QR zone settings
        if 'qr_zone' in data:
            qr_zone = data['qr_zone']
            tracker.qr_zone_percent = {
                'x1_percent': qr_zone.get('x1_percent', 0.7),
                'y1_percent': qr_zone.get('y1_percent', 0.0),
                'x2_percent': qr_zone.get('x2_percent', 1.0),
                'y2_percent': qr_zone.get('y2_percent', 1.0)
            }
            
            # Update zone position based on new percentages
            if hasattr(tracker, 'frame_shape') and tracker.frame_shape:
                tracker._update_qr_zone(tracker.frame_shape)
        
        # Update Shelf zone settings
        if 'shelf_zone' in data:
            shelf_zone = data['shelf_zone']
            tracker.shelf_zone_percent = {
                'x1_percent': shelf_zone.get('x1_percent', 0.0),
                'y1_percent': shelf_zone.get('y1_percent', 0.3),
                'x2_percent': shelf_zone.get('x2_percent', 0.5),
                'y2_percent': shelf_zone.get('y2_percent', 0.9)
            }
            
            # Update zone position based on new percentages
            if hasattr(tracker, 'frame_shape') and tracker.frame_shape:
                tracker._update_shelf_zone(tracker.frame_shape)
        
        return jsonify({'success': True, 'message': 'Zone settings updated'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

