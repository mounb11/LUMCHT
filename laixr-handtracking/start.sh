#!/bin/bash

# Script to start the LAIXR Handtracking application (backend and frontend)

echo "--- Starting LAIXR Handtracking Application ---"

# Function to kill all background processes on exit
cleanup() {
    echo ""
    echo "--- Shutting down servers ---"
    kill $BACKEND_PID
    pkill -f "next dev" # More reliable way to kill the Next.js dev server
    echo "All processes stopped."
    exit 0
}

# Trap CTRL+C (interrupt signal)
trap cleanup INT

# --- 1. Start Backend Server ---
echo "[1/2] Starting Python backend..."
cd backend
uvicorn main:app &
BACKEND_PID=$!
cd ..
echo "Backend server started with PID: ${BACKEND_PID}"


# --- 2. Start Frontend Server ---
echo "[2/2] Starting Next.js frontend..."
npm run dev &

# --- Wait for servers to be ready ---
echo ""
echo "Servers are starting. Please wait a moment..."
sleep 5 # Give servers a moment to initialize

echo ""
echo "--- LAIXR Application is Running ---"
echo "Backend available at:  http://localhost:8000"
echo "Frontend available at: http://localhost:3000"
echo ""
echo "Press CTRL+C in this terminal to stop both servers."
echo ""

# Wait indefinitely until the script is interrupted
wait 