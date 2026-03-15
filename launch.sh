#!/bin/bash
# Wait for desktop to fully load
sleep 20

# Navigate to project directory
cd /home/pi/stenosis_detector

# Activate virtual environment
source venv/bin/activate

# Run application
python3 main.py

# Exit cleanly
exit 0
