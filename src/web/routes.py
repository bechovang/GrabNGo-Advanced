"""
Web Server Routes
API endpoints for dashboard and QR scanner.
"""

from flask import request, jsonify, send_from_directory
import time
import os

# Import app from server module
# Import here to avoid circular dependency
from .server import app

# Tracker reference (set by server)
_tracker = None

def set_tracker(tracker_instance):
    """Set tracker instance for routes."""
    global _tracker
    _tracker = tracker_instance

@app.route('/')
def index():
    """Serve mobile web app."""
    import os
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web', 'static')
    return send_from_directory(static_dir, 'mobile_qr_scanner.html')

@app.route('/qr_zone_status', methods=['GET'])
def get_qr_zone_status():
    """Return QR zone status for mobile web."""
    if _tracker is None:
        return jsonify({
            'zone_active': False,
            'pending_id': None,
            'pending_count': 0,
            'error': 'Tracker not initialized'
        }), 500
    
    # Check zone status
    zone_active, pending_id, pending_count = _tracker._check_qr_zone()
    
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
    
    if _tracker is None:
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
    success, message = _tracker.confirm_pending_with_customer_id(customer_id, pending_id)
    
    print(f"   Confirmation result: success={success}, message={message}")
    
    if success:
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'pending_id': pending_id or _tracker.zone_active_pending,
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
    if _tracker is None:
        return jsonify({'pending_tracks': []})
    
    pending_list = []
    for track_id, pending in _tracker.pending_tracks.items():
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
    import os
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web', 'static')
    return send_from_directory(static_dir, 'dashboard.html')

@app.route('/dashboard/data', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data including customers and statistics."""
    # Try to load shared stats if tracker is not available or has no data
    try:
        from src.utils.stats_manager import StatsManager
        stats_manager = StatsManager()
        shared_stats = stats_manager.load_stats()
        shared_customers = stats_manager.load_customers_data()
        shared_mqtt_events = stats_manager.load_mqtt_events()
    except ImportError:
        stats_manager = None
        shared_stats = {}
        shared_customers = {}
        shared_mqtt_events = []
    
    if _tracker is None:
        # Use shared stats if available
        if shared_stats:
            return jsonify({
                'customers': shared_customers,
                'stats': {
                    'total_customers': shared_stats.get('total_customers', 0),
                    'active_customers': shared_stats.get('active_customers', 0),
                    'items_taken': shared_stats.get('items_taken', 0),
                    'avg_time': shared_stats.get('avg_time', '0m')
                },
                'mqtt_events': shared_mqtt_events,
                'mqtt_status': {
                    'connected': False,
                    'broker': 'unknown',
                    'topic': 'unknown'
                }
            })
        else:
            return jsonify({
                'customers': {},
                'stats': {
                    'total_customers': 0,
                    'active_customers': 0,
                    'items_taken': 0,
                    'avg_time': '0m'
                },
                'mqtt_events': [],
                'mqtt_status': {
                    'connected': False,
                    'broker': 'unknown',
                    'topic': 'unknown'
                }
            })
    
    # Prepare customers data
    customers_data = {}
    
    # Add confirmed customers
    for track_id, customer in _tracker.customers.items():
        # Use entry_time instead of first_seen
        entry_time = customer.get('entry_time') or customer.get('first_seen')
        if entry_time:
            if hasattr(entry_time, 'timestamp'):
                duration = time.time() - entry_time.timestamp()
            elif isinstance(entry_time, str):
                try:
                    from dateutil import parser
                    entry_dt = parser.parse(entry_time)
                    duration = time.time() - entry_dt.timestamp()
                except:
                    duration = 0
            else:
                duration = 0
        else:
            duration = 0
        
        # Check if confirmed (use state or confirmed field)
        state = customer.get('state')
        is_confirmed = False
        if state:
            if hasattr(state, 'name'):
                is_confirmed = state.name == 'CONFIRMED'
            elif str(state) == 'TrackState.CONFIRMED' or str(state) == 'CONFIRMED':
                is_confirmed = True
        else:
            is_confirmed = customer.get('confirmed', False)
        
        # Get last_seen (use last_detection_time)
        last_detection_time = customer.get('last_detection_time') or customer.get('last_seen')
        last_seen_str = None
        if last_detection_time:
            if hasattr(last_detection_time, 'isoformat'):
                last_seen_str = last_detection_time.isoformat()
            elif isinstance(last_detection_time, str):
                last_seen_str = last_detection_time
        
        # Get first_seen string
        first_seen_str = None
        if entry_time:
            if hasattr(entry_time, 'isoformat'):
                first_seen_str = entry_time.isoformat()
            elif isinstance(entry_time, str):
                first_seen_str = entry_time
        
        customers_data[f"customer_{track_id}"] = {
            'track_id': track_id,
            'customer_id': customer.get('customer_id', 'UNKNOWN'),
            'confirmed': is_confirmed,
            'first_seen': first_seen_str,
            'last_seen': last_seen_str,
            'duration': duration,
            'last_box': customer.get('last_box'),
            'shopping_cart': customer.get('shopping_cart', []),
            'pickup_count': customer.get('pickup_count', 0)
        }
    
    # Add pending customers
    for pending_id, pending in _tracker.pending_tracks.items():
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
    if _tracker and hasattr(_tracker, 'events'):
        # Get only recent MQTT-related events
        for event in _tracker.events:
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

    # If tracker has no customers, try to use shared stats
    if not customers_data and shared_stats:
        customers_data = shared_customers
        total_customers = shared_stats.get('total_customers', 0)
        active_customers = shared_stats.get('active_customers', 0)
        items_taken = shared_stats.get('items_taken', 0)
        avg_time = shared_stats.get('avg_time', '0m')
        if not mqtt_events:
            mqtt_events = shared_mqtt_events
    
    # Get MQTT connection status
    mqtt_connected = getattr(_tracker, 'mqtt_connected', False) if _tracker else False
    mqtt_broker = getattr(_tracker, 'mqtt_broker', 'unknown') if _tracker else 'unknown'
    mqtt_topic = getattr(_tracker, 'mqtt_topic_weight', 'unknown') if _tracker else 'unknown'
    
    mqtt_status = {
        'connected': mqtt_connected,
        'broker': mqtt_broker,
        'topic': mqtt_topic
    }
    
    # Return dashboard data
    try:
        response_data = {
            'customers': customers_data,
            'stats': {
                'total_customers': total_customers,
                'active_customers': active_customers,
                'items_taken': items_taken,
                'avg_time': avg_time
            },
            'mqtt_events': mqtt_events,
            'mqtt_status': mqtt_status
        }
        return jsonify(response_data)
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'customers': {},
            'stats': {
                'total_customers': 0,
                'active_customers': 0,
                'items_taken': 0,
                'avg_time': '0m'
            },
            'mqtt_events': [],
            'mqtt_status': {
                'connected': False,
                'broker': 'unknown',
                'topic': 'unknown'
            }
        }), 500

