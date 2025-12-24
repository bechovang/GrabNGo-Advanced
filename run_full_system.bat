@echo off
echo ================================================
echo     SMART RETAIL SYSTEM - FULL FLOW
echo ================================================
echo.
echo Getting network information...
python get_network_ip.py

echo.
echo This script will:
echo 1. Check system requirements
echo 2. Test ESP32 connection
echo 3. Start tracking system with dashboard
echo.
echo Press Ctrl+C to stop at any time
echo ================================================
echo.

python run_full_system.py

echo.
echo ================================================
echo System stopped. Press any key to exit.
echo ================================================
pause >nul