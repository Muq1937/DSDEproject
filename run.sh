#!/bin/bash
# Quick start script for Bangkok Traffy Dashboard

echo "========================================="
echo "Bangkok Traffy Dashboard - Quick Start"
echo "========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/installed.flag" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/installed.flag
    echo "✓ Dependencies installed"
    echo ""
fi

# Display options
echo "Select mode:"
echo "  1) Dashboard (interactive web app)"
echo "  2) Generate maps"
echo "  3) Generate network graph"
echo "  4) Generate all visualizations"
echo ""
read -p "Enter choice [1-4]: " choice

# Set sample size
read -p "Enter sample fraction (0.01 to 1.0) [default: 0.1]: " sample
sample=${sample:-0.1}

# Set CSV file
read -p "Enter CSV filename [default: bangkok_traffy_30.csv]: " csvfile
csvfile=${csvfile:-bangkok_traffy_30.csv}

echo ""
echo "Starting with:"
echo "  CSV: $csvfile"
echo "  Sample: $(echo "$sample * 100" | bc)%"
echo ""

# Run based on choice
case $choice in
    1)
        echo "Starting dashboard on http://localhost:8050"
        echo "Press Ctrl+C to stop"
        python app.py dashboard --csv "$csvfile" --sample "$sample"
        ;;
    2)
        echo "Generating maps..."
        python app.py maps --csv "$csvfile" --sample "$sample"
        ;;
    3)
        echo "Generating network graph..."
        python app.py network --csv "$csvfile" --sample "$sample"
        ;;
    4)
        echo "Generating all visualizations..."
        python app.py all --csv "$csvfile" --sample "$sample"
        ;;
    *)
        echo "Invalid choice. Exiting."
        deactivate
        exit 1
        ;;
esac

echo ""
echo "Done! Deactivating virtual environment..."
deactivate
