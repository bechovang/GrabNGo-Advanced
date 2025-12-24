@echo off
echo ================================================
echo     SMART RETAIL DASHBOARD STARTUP
echo ================================================
echo.
echo Starting Smart Retail Tracking System with Dashboard...
echo.
echo Dashboard: http://localhost:8080/dashboard
echo Mobile QR Scanner: http://localhost:8080
echo.
echo Press Ctrl+C to stop the system
echo ================================================
echo.

python run_dashboard.py

echo.
echo ================================================
echo System stopped. Press any key to exit.
echo ================================================
pause >nul