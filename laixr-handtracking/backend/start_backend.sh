#!/bin/bash

echo "🚀 Starting LAIXR Hand Tracking Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create uploads and results directories
mkdir -p uploads results

# Start the server
echo "🎯 Starting FastAPI server on http://localhost:8000"
echo "📊 Dashboard will be available at http://localhost:3002"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py 