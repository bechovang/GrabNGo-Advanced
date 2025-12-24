@echo off
echo ================================================
echo     MQTT LOG DASHBOARD TEST
echo ================================================
echo.
echo This will:
echo 1. Start the tracking system
echo 2. Add a mock customer in shelf zone
echo 3. Simulate MQTT weight events
echo 4. Display the events in the dashboard log
echo.
echo Open http://localhost:8080/dashboard in your browser
echo   to see the MQTT Log section (blue box)
echo.
echo Press Ctrl+C to stop the test
echo ================================================
echo.

python test_mqtt_log.py

echo.
echo ================================================
echo Test stopped. Press any key to exit.
echo ================================================
pause >nul