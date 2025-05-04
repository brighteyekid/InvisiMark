#!/bin/bash

# InvisiMark Steganography Fix Test Script
echo "===== InvisiMark Steganography Fix Tester ====="
echo "This script will test if the steganography module is working correctly."

# Check if virtual environment exists, create if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Make sure python3-venv is installed."
        echo "Try: sudo apt-get install python3-venv"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
if [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing required packages for testing..."
    pip install numpy opencv-python scipy pywt qrcode pillow cryptography
    # Mark as installed
    touch venv/.requirements_installed
fi

echo "Running test script..."
echo "---------------------"

# Run the test script
python test_fix.py

# Deactivate virtual environment
deactivate 