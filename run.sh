#!/bin/bash

# InvisiMark Quick Start Script
echo "===== InvisiMark Quick Start ====="
echo "This script will help you run InvisiMark with the correct environment setup."

# Check if virtual environment exists
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

# Install dependencies if needed
if [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Reinstall PyQt5 with proper options to fix platform plugin issues
    echo "Reinstalling PyQt5 to fix potential platform plugin issues..."
    pip uninstall -y PyQt5
    pip install PyQt5 --config-settings --confirm-license= --verbose
    
    # Mark requirements as installed
    touch venv/.requirements_installed
fi

# Check for Qt dependencies
echo "Checking Qt dependencies..."
if ! dpkg -s libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 &>/dev/null; then
    echo "Some Qt dependencies may be missing. Installing them requires sudo privileges."
    echo "Would you like to install them now? (y/n)"
    read -r answer
    if [[ "$answer" == "y" ]]; then
        sudo apt-get install -y libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-randr0 libxcb-xkb1 libxkbcommon-x11-0
    fi
fi

# Set Qt environment variables to avoid plugin errors
export QT_QPA_PLATFORM_PLUGIN_PATH=""
export QT_DEBUG_PLUGINS=1

echo "Starting InvisiMark..."
echo "---------------------"

# Run the application
python main.py

# If application failed, offer to run demo
if [ $? -ne 0 ]; then
    echo "The GUI application encountered an error."
    echo "Would you like to run the command-line demo instead? (y/n)"
    read -r run_demo
    if [[ "$run_demo" == "y" ]]; then
        echo "Running command-line demo..."
        python demo.py
    fi
fi

# Deactivate virtual environment
deactivate 