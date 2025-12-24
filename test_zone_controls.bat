@echo off
echo ================================================
echo     ZONE CONTROLS TEST
echo ================================================
echo.
echo This will test the zone controls functionality.
echo.
echo 1. Starting tracking system...
echo 2. Open dashboard
echo 3. Click the "Zones" button to adjust zones
echo.
echo Press Ctrl+C to stop
echo ================================================
echo.

python run_full_system.py

echo.
echo ================================================
echo Test stopped. Press any key to exit.
echo ================================================
pause >nul