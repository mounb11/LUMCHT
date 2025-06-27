#!/bin/bash

echo "--- LAIXR Handtracking Setup ---"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    echo "   Download from: https://www.python.org/"
    exit 1
fi

echo "✅ Prerequisites found"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install frontend dependencies"
    exit 1
fi

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment and install backend dependencies
echo "📦 Installing backend dependencies..."
source venv/bin/activate
pip install -r backend/requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install backend dependencies"
    exit 1
fi

# Make start script executable
chmod +x start.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application, run:"
echo "  ./start.sh"
echo "" 