"""
Shared Stats Manager
Allows multiple processes to share tracking statistics via JSON file.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path


class StatsManager:
    """Manages shared statistics between processes."""
    
    def __init__(self, stats_file='data/shared_stats.json'):
        """
        Initialize stats manager.
        
        Args:
            stats_file: Path to shared stats JSON file
        """
        self.stats_file = stats_file
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure stats file directory exists."""
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
    
    def save_stats(self, stats_data):
        """
        Save stats to shared file.
        
        Args:
            stats_data: Dictionary with stats (customers, events, etc.)
        """
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'stats': stats_data,
                'last_updated': time.time()
            }
            
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception as e:
            # Silent fail - don't break main process
            pass
    
    def load_stats(self):
        """
        Load stats from shared file.
        
        Returns:
            dict: Stats data or empty dict if file doesn't exist
        """
        try:
            if not os.path.exists(self.stats_file):
                return {
                    'total_customers': 0,
                    'active_customers': 0,
                    'items_taken': 0,
                    'avg_time': '0m',
                    'total_events': 0,
                    'last_updated': None
                }
            
            with open(self.stats_file, 'r') as f:
                data = json.load(f)
                return data.get('stats', {})
        except Exception as e:
            # Return default stats on error
            return {
                'total_customers': 0,
                'active_customers': 0,
                'items_taken': 0,
                'avg_time': '0m',
                'total_events': 0,
                'last_updated': None
            }
    
    def save_customers_data(self, customers_data):
        """
        Save customers data to shared file.
        
        Args:
            customers_data: Dictionary of customers
        """
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'customers': customers_data,
                'last_updated': time.time()
            }
            
            customers_file = self.stats_file.replace('shared_stats.json', 'shared_customers.json')
            with open(customers_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            # Silent fail
            pass
    
    def load_customers_data(self):
        """
        Load customers data from shared file.
        
        Returns:
            dict: Customers data or empty dict if file doesn't exist
        """
        try:
            customers_file = self.stats_file.replace('shared_stats.json', 'shared_customers.json')
            if not os.path.exists(customers_file):
                return {}
            
            with open(customers_file, 'r') as f:
                data = json.load(f)
                return data.get('customers', {})
        except Exception as e:
            return {}
    
    def save_mqtt_events(self, mqtt_events):
        """
        Save MQTT events to shared file.
        
        Args:
            mqtt_events: List of MQTT events
        """
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'events': mqtt_events[-20:],  # Keep last 20 events
                'last_updated': time.time()
            }
            
            events_file = self.stats_file.replace('shared_stats.json', 'shared_mqtt_events.json')
            with open(events_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            # Silent fail
            pass
    
    def load_mqtt_events(self):
        """
        Load MQTT events from shared file.
        
        Returns:
            list: MQTT events or empty list if file doesn't exist
        """
        try:
            events_file = self.stats_file.replace('shared_stats.json', 'shared_mqtt_events.json')
            if not os.path.exists(events_file):
                return []
            
            with open(events_file, 'r') as f:
                data = json.load(f)
                return data.get('events', [])
        except Exception as e:
            return []

