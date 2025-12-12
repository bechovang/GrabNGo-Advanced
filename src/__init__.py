"""
Smart Retail Tracking System
Core modules for customer tracking with BoT-SORT and ReID
"""

from .main_tracker import RetailCustomerTracker, LightweightReID, TrackState

# HoldingDetector is optional (requires mediapipe)
try:
    from .holding_detector import HoldingDetector
except ImportError:
    HoldingDetector = None

__all__ = ['RetailCustomerTracker', 'LightweightReID', 'TrackState', 'HoldingDetector']

