@echo off
echo --- Starting LAIXR Handtracking Application ---

echo [1/2] Starting Python backend...
cd backend
start /b uvicorn main:app
cd ..
echo Backend server started

echo [2/2] Starting Next.js frontend...
start /b npm run dev

echo.
echo Servers are starting. Please wait a moment...
timeout /t 5 /nobreak >nul

echo.
echo --- LAIXR Application is Running ---
echo Backend available at:  http://localhost:8000
echo Frontend available at: http://localhost:3000
echo.
echo Press any key to stop the servers...
echo.

pause 