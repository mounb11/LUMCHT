# Script to start the Python backend
Write-Host "--- Starting Python Backend Server ---" -ForegroundColor Cyan
Write-Host "This terminal will show the backend logs."
Write-Host "URL: http://localhost:8000"
Write-Host "----------------------------------------"

# Activate the virtual environment
& ".\venv\Scripts\Activate.ps1"

# Start the FastAPI server using Uvicorn
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir ./backend 